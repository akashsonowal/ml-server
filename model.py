from transformers import AutoTokenizer, AutoModelForCausalLM

class ServeModel:
    def initialize(self):
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-13b")
        model = AutoModelForCausalLM.from_pretrained("facebook/opt-13b")
