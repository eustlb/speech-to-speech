import threading
from queue import Queue
import queue
import sounddevice as sd
import numpy as np
import gradio as gr
import time
from dataclasses import dataclass, field
import websocket
import threading
import librosa
import io
import ssl
from pydub import AudioSegment
import asyncio
from starlette.endpoints import WebSocketEndpoint
from starlette.routing import Route, WebSocketRoute
from starlette.applications import Starlette
from starlette.responses import JSONResponse

@dataclass
class AppState:
    stream: np.ndarray | None = None
    sampling_rate: int = 0
    pause_detected: bool = False
    started_talking: bool =  False
    stopped: bool = False
    conversation: list = field(default_factory=list)
    session_state: str = "idle"

@dataclass
class AudioStreamingClientArguments:
    sample_rate: int = field(default=48000, metadata={"help": "Audio sample rate in Hz."})
    api_url: str = field(default="https://yxfmjcvuzgi123sw.us-east-1.aws.endpoints.huggingface.cloud", metadata={"help": "The URL of the API endpoint."})
    auth_token: str = field(default="your_auth_token", metadata={"help": "Authentication token for the API."})
    

class AudioStreamingClient:
    def __init__(self, args: AudioStreamingClientArguments):
        self.args = args
        self.stop_event = threading.Event()
        self.send_queue = Queue()
        self.recv_queue = Queue()
        self.session_id = None
        self.headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.args.auth_token}",
            "Content-Type": "application/json"
        }
        self.ws_ready = threading.Event()
        self.session_state = "idle"
        self.should_stop = threading.Event()

    def start(self):
        print("Starting audio streaming...")
        ws_url = self.args.api_url.replace("http", "ws") + "/ws"

        self.ws = websocket.WebSocketApp(
            ws_url,
            header=[f"{key}: {value}" for key, value in self.headers.items()],
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )

        ws_thread = threading.Thread(target=self.ws.run_forever, kwargs={'sslopt': {"cert_reqs": ssl.CERT_NONE}})
        ws_thread.start()
        
        # Wait for the WebSocket to be ready
        self.ws_ready.wait()
        self.start_audio_streaming()

    def start_audio_streaming(self):
        self.send_thread = threading.Thread(target=self.send_audio_thread)
        self.send_thread.start()

    def on_open(self, ws):
        print("WebSocket connection opened.")
        self.ws_ready.set()  # Signal that the WebSocket is ready

    def on_message(self, ws, message):
        print(f" =============== Received message =============")
        self.should_stop.set()
        # message is bytes
        if message == b'DONE':
            print("LISTENING")
            # self.session_state = "listen"
            # self.state = gr.State(value=AppState(session_state="listen"))
        else:
            
            self.session_state = "processing"
            # self.state = gr.State(value=AppState(session_state="processing"))
            print(len(message))
            audio_np = np.frombuffer(message, dtype=np.int16)
            if len(audio_np) > 0:
                print("PROCESSING, {}".format(len(audio_np) / self.args.sample_rate))
                self.recv_queue.put(audio_np)

    def on_error(self, ws, error):
        print(f"WebSocket error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        print("WebSocket connection closed.")

    def on_shutdown(self):
        self.stop_event.set()
        self.send_thread.join()
        # self.play_thread.join()
        self.ws.close()
        if hasattr(self, 'input_stream'):
            self.input_stream.stop()
            self.input_stream.close()
        if hasattr(self, 'output_stream'):
            self.output_stream.stop()
            self.output_stream.close()
        print("Service shutdown.")

    def send_audio_thread(self):
        while not self.should_stop.is_set():
            if not self.send_queue.empty():
                chunk = self.send_queue.get()
                if self.session_state != "processing":
                    print("sending")
                    self.ws.send(chunk.astype(np.int16).tobytes(), opcode=websocket.ABNF.OPCODE_BINARY)
                else:
                    self.ws.send([], opcode=websocket.ABNF.OPCODE_BINARY)  # handshake 
            time.sleep(0.01)

    def gradio_interface(self):

        def send_audio(audio):
            sr, data = audio

            # Resample the audio data to 16000 Hz if necessary
            if sr != 16000:
                data = data.astype(np.float32) / 32768.0
                data = librosa.resample(data, orig_sr=sr, target_sr=16000)
                data = (data * 32768.0).astype(np.int16)
            self.send_queue.put(data)

            if self.session_state == "processing":
                return gr.Audio(recording=False)
            return gr.Audio(recording=True)
        
        def from_queue_to_bytes():
            while True:
                if not self.recv_queue.empty(): 
                    audio = self.recv_queue.get()
                    audio_buffer = io.BytesIO()
                    segment = AudioSegment(
                        audio.tobytes(),
                        frame_rate=16000,
                        sample_width=2,
                        channels=1,
                    )
                    segment.export(audio_buffer, format="mp3", bitrate="320k")
                    audio = audio_buffer.getvalue()
                    yield audio  

        def recv_audio():
            generator = from_queue_to_bytes()
            for audio in generator:

                print(f"yielding audio {len(audio)}")
                yield audio

        def start_recording():
            return gr.Audio(recording=True)

        with gr.Blocks() as demo:
            with gr.Row():
                with gr.Column():
                    input_audio = gr.Audio(
                        label="Input Audio", 
                        sources="microphone", 
                        type="numpy"
                    )
                with gr.Column():
                    output_audio = gr.Audio(
                        label="Output Audio", 
                        streaming=True, 
                        autoplay=True
                    )

            input_stream = input_audio.stream(
                send_audio,
                [input_audio],
                [input_audio],
                stream_every=1,
                time_limit=30,
            )

            when_stop_recording = input_audio.stop_recording(
                recv_audio,
                [],
                [output_audio],   
            )

            when_response_finished = output_audio.stop(
                start_recording,
                [],
                [input_audio],
            )

        demo.launch()

# if __name__ == "__main__":
    # import argparse

    # parser = argparse.ArgumentParser(description="Audio Streaming Client")
    # parser.add_argument("--sample_rate", type=int, default=16000, help="Audio sample rate in Hz. Default is 16000.")
    # parser.add_argument("--chunk_size", type=int, default=1024, help="The size of audio chunks in samples. Default is 1024.")
    # parser.add_argument("--api_url", type=str, required=True, help="The URL of the API endpoint.")
    # parser.add_argument("--auth_token", type=str, required=True, help="Authentication token for the API.")

    # args = parser.parse_args()

client_args = AudioStreamingClientArguments(api_url = "ws://localhost:8765", auth_token = "your_token")
client = AudioStreamingClient(client_args)
client.start()
client.start_audio_streaming()
client.gradio_interface()
