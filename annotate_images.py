#%%
from transformers import pipeline, LlavaForConditionalGeneration, BitsAndBytesConfig
from PIL import Image    
import torch

model_id = "xtuner/llava-llama-3-8b-v1_1-transformers"
config = BitsAndBytesConfig(
    load_in_8bit = True
)

model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True,
    quantization_config = config
)

# %%

image = Image.open("Intel/llava-llama-3-8b")

prompt = ("<|start_header_id|>user<|end_header_id|>\n\n<image>\nWhat are these?<|eot_id|>"
          "<|start_header_id|>assistant<|end_header_id|>\n\n")
outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 200})
print(outputs)