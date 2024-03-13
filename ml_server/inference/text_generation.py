import os
import torch
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login

from typing import List, Optional

from ..api_server import MLServe

DEFAULT_MODEL = "meta-llama/Llama-2-7b-chat-hf"
login(token="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")

MAX_MAX_NEW_TOKENS = 2048
DEFAULT_MAX_NEW_TOKENS = 1024
MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "4096"))

device = "cuda" if torch.cuda.is_available() else "cpu"

def generate_prompt(message, tokenizer):
    conversation = []
    system_prompt = "You are a medical assistant who helps doctors prepare detailed SOAP Notes. In the response, please don't start with 'Sure' or 'Here' and please be very detailed. And also please don't include any extra comments in the bottom."
    conversation.append({"role": "system", "content": system_prompt})
    conversation.append({"role": "user", "content": message})
    return tokenizer.apply_chat_template(conversation, tokenize=False)

class PromptRequest(BaseModel):
    prompt: str = "Generate a short story about a mysterious adventure."
    temperature: float = 1
    max_tokens: int = 1024
    stop: List[str] = []


class ResponseModel(BaseModel):
    prompt: str
    prompt_token_ids: Optional[List] = None  # The token IDs of the prompt.
    text: str  # The output sequences of the request.
    finished: bool  # Whether the whole request is finished.


class ServeTextGenerator(MLServe):
    def __init__(self, model_path=DEFAULT_MODEL, batch_size=1, timeout=0.05):
        super().__init__(
                handle=self.handle,
                input_schema=PromptRequest,
                response_schema=ResponseModel,
                max_batch_size=batch_size,
                batch_wait_timeout_s=timeout
            )
        torch.set_default_device(device)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            use_safetensors=True,
            quantization_config=bnb_config,
            trust_remote_code=True,
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.tokenizer.use_default_system_prompt = False

    @torch.inference_mode()
    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.2,
    ) -> List[str]:
        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True, return_attention_mask=False
        ).to(device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
        return self.tokenizer.batch_decode(
            outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

    def handle(self, batch: List[PromptRequest]) -> List[ResponseModel]:
        outputs = self.generate(
            prompts=[generate_prompt(item.prompt, self.tokenizer) for item in batch],
            max_new_tokens=batch[0].max_tokens,
            temperature=batch[0].temperature,
        )
        responses = [
            ResponseModel(prompt=item.prompt, text=output, finished=True)
            for item, output in zip(batch, outputs)
        ]
        return responses