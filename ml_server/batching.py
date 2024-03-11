import asyncio
import uuid
import time

from typing import Callable
from pydantic import BaseModel

class BatchQueue:
    def __init__(self, max_batch_size: int, batch_wait_timeout_s: int, handle_batch_func: Callable):
        self.queue = asyncio.Queue()
        self.max_batch_size = max_batch_size
        self.batch_wait_timeout_s = batch_wait_timeout_s
        self.requests_available_event = asyncio.Event()

        self.func = handle_batch_func
        self._responses = {}
         
    def put(self, request):
        self.queue.put_nowait(request)
        self.requests_available_event.set()
    
    async def consume(self):
        batch = []
        batch.append(await self.queue.get())

        max_batch_size = self.max_batch_size
        batch_wait_timeout_s = self.batch_wait_timeout_s

        batch_start_time = time.time()
        while True:
            remaining_batch_time_s = max(
                batch_wait_timeout_s - (time.time() - batch_start_time), 0
            )
            try:
                await asyncio.wait_for(
                    self.requests_available_event.wait(), remaining_batch_time_s
                )
            except asyncio.TimeoutError:
                pass
            
            while len(batch) < max_batch_size and not self.queue.empty():
                batch.append(self.queue.get_nowait())

            if self.queue.empty():
                self.requests_available_event.clear()

            if (
                time.time() - batch_start_time >= batch_wait_timeout_s
                or len(batch) >= max_batch_size
            ):
                break
        
        return batch
    
    async def process_request(self, data: BaseModel):
        request_id = uuid.uuid4().hex
        self.put((request_id, data))
        while True:
            if request_id in self._responses:
                result = self._responses.pop(request_id)
                return result
            await asyncio.sleep(0.001)
    
    async def process_batches(self):
        batch = await self.consume()
        assert len(batch) > 0
        data = self.func(batch)
        for item in data:
            self._responses[item["uid"]] = item
