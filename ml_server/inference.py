from .api_server import MLServe

class InferenceEngine(MLServe):
    def __init__(self):
        super().__init__(self.handler)
        self.model = None
        pass 
    
    def handler(self):
        pass