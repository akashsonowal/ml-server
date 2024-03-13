import logging
import os.path
import urllib

from typing import List
from PIL import Image

import torch
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from pydantic import BaseModel
from fastapi import UploadFile

from ..api_server import MLServe

class ResponseSchema(BaseModel):
    label: str
    score: float

class ServeImageClassifier(MLServe):
    INPUT_SCHEMA = UploadFile

    def __init__(self, model_name, batch_size, timeout):
        super().__init__(
                handle=self.predict,
                input_schema=ServeImageClassifier.INPUT_SCHEMA,
                response_schema=ResponseSchema,
                max_batch_size=batch_size,
                batch_wait_timeout_s=timeout
            )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self._model = timm.create_model(model_name, pretrained=True).to(device)
        self._model.eval()

        config = resolve_data_config({}, model=self._model)
        self.transform = create_transform(**config)

        if not os.path.isfile("imagenet_classes.txt"):
            logging.info("Downloading Imagenet classes")
            url, filename = (
                "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt",
                "imagenet_classes.txt",
            )
            urllib.request.urlretrieve(url, filename)
        with open("imagenet_classes.txt") as f:
            self.categories = [s.strip() for s in f.readlines()]

    def process_image(self, filename):
        from PIL import Image

        img = Image.open(filename).convert("RGB")
        tensor = self.transform(img)  # transform and add batch dimension
        return tensor
    
    @torch.no_grad()
    def predict(self, batch: List[Image.Image]) -> List[int]:
        tensors = []

        for file in batch:
            tensor = self.process_image(file.file)
            tensors.append(tensor)
        tensors = torch.stack(tensors).to(self.device)

        out = self._model(tensors)
        probs: torch.Tensor = torch.nn.functional.softmax(out, dim=1)  # batchX1000
        max_probs, indices = probs.max(1)  # batch x 1

        results = [
            ResponseSchema(label=self.categories[idx], score=p.cpu().item())
            for idx, p in zip(indices, max_probs)
        ]
        return results