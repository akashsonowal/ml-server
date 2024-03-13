import logging
import time
import asyncio

from typing import Any, Callable, List

logger = logging.getLogger(__name__)

class BatchQueue:
    def __init__(self, max_batch_size: int, batch_wait_timeout_s: int, handle_batch_func: Callable):
        self.queue = asyncio.Queue()
        self.max_batch_size = max_batch_size
        self.batch_wait_timeout_s = batch_wait_timeout_s

        self._loop = asyncio.get_event_loop()
        self._requests_available_event = asyncio.Event()

        self._handle_batch_task = None

        if handle_batch_func is not None:
            self._handle_batch_task = self._loop.create_task(
                self._process_batches(handle_batch_func)
            )
        
    def put(self, request):
        self.queue.put_nowait(request)
        self._requests_available_event.set()
    
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
                    self._requests_available_event.wait(), remaining_batch_time_s
                )
            except asyncio.TimeoutError:
                pass
            
            while len(batch) < max_batch_size and not self.queue.empty():
                batch.append(self.queue.get_nowait())

            if self.queue.empty():
                self._requests_available_event.clear()

            if (
                time.time() - batch_start_time >= batch_wait_timeout_s
                or len(batch) >= max_batch_size
            ):
                break
        return batch
    
    async def process_request(self, item):
        future = self._loop.create_future()
        self.put((item, future))
        return await future
    
    async def _process_batches(self, func):
        while not self._loop.is_closed():
            await self._process_batch(func)

    async def _process_batch(self, func: Callable) -> None:
        batch = await self.consume()
        assert len(batch) > 0

        items, futures = zip(*batch)

        try:
            results = func(items)

            for result, future in zip(results, futures):
                if not future.done():
                    future.set_result(result)
        
        except Exception as e:
            for future in futures:
                if not future.done():
                    future.set_exception(e)
