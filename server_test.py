import asyncio
from starlette.endpoints import WebSocketEndpoint
from starlette.routing import Route, WebSocketRoute
from starlette.applications import Starlette
import numpy as np
import base64

SAMPLE_RATE = 16000  # Assuming 16kHz sample rate
ONE_SECOND_SAMPLES = SAMPLE_RATE
CHUNK_SIZE = 1024  # Adjust as needed

class AudioBuffer:
    def __init__(self):
        self.buffer = np.array([], dtype=np.int16)
    
    def add_chunk(self, chunk):
        self.buffer = np.append(self.buffer, chunk)
    
    def get_duration(self):
        return len(self.buffer) / SAMPLE_RATE
    
    def clear(self):
        self.buffer = np.array([], dtype=np.int16)
    
    def get_audio(self):
        return self.buffer.tobytes()

class WebSocketPredictEndpoint(WebSocketEndpoint):
    encoding = "bytes"
    FIRST_RECEIVE = True

    async def on_connect(self, websocket):
        await websocket.accept()
        print("connected")
        self.audio_buffer = AudioBuffer()

    async def on_receive(self, websocket, data):
        audio_chunk = np.frombuffer(data, dtype=np.int16) 
        self.audio_buffer.add_chunk(audio_chunk)
        
        # Check if we have accumulated 5 seconds of audio
        if self.audio_buffer.get_duration() >= 5.0:
            print("5 seconds accumulated, sending back")
            audio_data = self.audio_buffer.get_audio()
            await websocket.send_bytes(audio_data)
            self.audio_buffer.clear()

    async def on_disconnect(self, websocket, close_code):
        pass

async def health(request):
    return JSONResponse({"status": "ok"})

app = Starlette(
    debug=False,
    routes=[
        Route("/", health, methods=["GET"]),
        Route("/health", health, methods=["GET"]),
        WebSocketRoute("/ws", WebSocketPredictEndpoint)
    ]
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)