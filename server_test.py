import asyncio
from starlette.endpoints import WebSocketEndpoint
from starlette.routing import Route, WebSocketRoute
from starlette.applications import Starlette
from starlette.responses import JSONResponse
import numpy as np
import time
from asyncio import Queue

SAMPLE_RATE = 16000  # Assuming 48kHz sample rate
CHUNK_SIZE = 1024 * 10  # Adjust as needed
BUFFER_DELAY = 10  # Delay in seconds to accumulate packets

class WebSocketPredictEndpoint(WebSocketEndpoint):
    encoding = "bytes"

    async def on_connect(self, websocket):
        await websocket.accept()
        print("connected")
        self.audio_queue = Queue()
        self.start_event = asyncio.Event()
        self.start_time_task = None
        self.send_task = asyncio.create_task(self.send_audio(websocket))

    async def on_receive(self, websocket, data):
        if data == b"DONE":
            await websocket.send_bytes(b"DONE")
            return
        audio = np.frombuffer(data, dtype=np.int16)
        audio_chunks = [audio[i:i+CHUNK_SIZE] for i in range(0, len(audio), CHUNK_SIZE)]
        if len(audio) > 0:
            for chunk in audio_chunks:
                print(f"received {len(chunk) / SAMPLE_RATE} seconds")
                await self.audio_queue.put(chunk)
        else:
            await websocket.send_bytes(b"DONE")

        # If this is the first packet, start the delay timer
        if not self.start_event.is_set() and not self.start_time_task:
            self.start_time_task = asyncio.create_task(self.delay_start())

    async def delay_start(self):
        await asyncio.sleep(BUFFER_DELAY)
        self.start_event.set()
        print(f"Buffering for {BUFFER_DELAY} seconds completed. Starting to send audio.")

    async def send_audio(self, websocket):
        await self.start_event.wait()
        while True:
            if not self.audio_queue.empty():
                audio_chunk = await self.audio_queue.get()
                await websocket.send_bytes(audio_chunk.tobytes())
         
    async def on_disconnect(self, websocket, close_code):
        self.send_task.cancel()
        if self.start_time_task:
            self.start_time_task.cancel()
        try:
            await self.send_task
        except asyncio.CancelledError:
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