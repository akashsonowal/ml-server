import asyncio
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from uvicorn import Config, Server
from typing import Callable

from .batching import BatchQueue

class MLServe:
    def __init__(self, handle: Callable):
        self.app = FastAPI(title="MLServe: A scalable ML Server", docs_url="/")
        self.loop = asyncio.new_event_loop()
        self.config = Config(app=self.app, loop=self.loop, host="0.0.0.0", port=8080, log_level="info")
        self.server = Server(self.config)
        self.queue = BatchQueue(max_batch_size=16, batch_wait_timeout_s=1, handle_batch_func=handle)

        self.app.add_api_route("/health", self.health, methods=["GET"])
        self.app.add_api_route("/endpoint", self.api, methods=["POST"])

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    async def health(self):
        return Response(status_code=200)

    async def api(self, request: Request):
        self.queue.process_batch(request)
        return {"message": "Hello from /endpoint"}

    def run_server(self):
        asyncio.run(self.server.serve())