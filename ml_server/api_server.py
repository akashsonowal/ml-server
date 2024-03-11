import asyncio
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from uvicorn import Config, Server
from typing import Callable
import logging

from .batching import BatchQueue

logger = logging.getLogger(__name__)

class MLServe:
    def __init__(self, handle: Callable, max_batch_size: int = 16, batch_wait_timeout_s: int = 0.05):
        self.app = FastAPI(title="MLServe: A scalable ML Server", docs_url="/")
        self.loop = asyncio.new_event_loop()
        self.config = Config(app=self.app, loop=self.loop, host="0.0.0.0", port=8080, log_level="info")
        self.server = Server(self.config)
        self.queue = BatchQueue(max_batch_size, batch_wait_timeout_s, handle)

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
        try:
            data = await request.json() 
            result = await self.queue.process_request(data)
            return {"result": result}
        except Exception as e:
            logger.exception("Error processing request")
            raise HTTPException(status_code=500, detail="Internal Server Error")
    
    @self.app.on_event("startup")
    async def startup_event(self):
        asyncio.create_task(self.queue.process_batches())

    def run_server(self):
        asyncio.run(self.server.serve())
