import asyncio

from typing import Callable

class BatchQueue:
    def __init__(self, max_batch_size: int, batch_wait_timeout_s: int, handle_batch_func: Callable):
        self._queue = asyncio.Queue()
        self.max_batch_size = max_batch_size
        self.batch_wait_timeout_s = batch_wait_timeout_s
        self.handle = handle_batch_func
    
    def get(self):
        pass
    
    def put(self, request):
        self.queue.put_nowait(request)
    
    def process_batch(self):
        pass 