from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import torch

model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)

print(type(tokenizer).__name__)

text = "I am polishing my AI Engineeering"

tokens = tokenizer.tokenize(text)

token_ids = tokenizer.encode(text, add_special_tokens=False)

print("Tokens:", tokens)
print("Token IDs:", token_ids)
print("Token count:", len(token_ids))
print("Decoded:", tokenizer.decode(token_ids))

model = AutoModelForCausalLM.from_pretrained(model_id)
model = model.to("cpu")
model.eval()

print("Model class:", type(model).__name__)
print("Device:", next(model.parameters()).device)

parameter_count = sum(p.numel() for p in model.parameters())  # `numel()` counts the numerical values in each parameter tensor; sum() totals them.
name, parameter = next(model.named_parameters()) # named_parameters() lets us inspect a parameter’s name and tensor.


print("Total parameters:", f"{parameter_count:,}")
print("First parameter name:", name)
print("Shape:", tuple(parameter.shape))
print("Data type:", parameter.dtype)
print("First five values:", parameter.detach().reshape(-1)[:5]) # detach().reshape(-1)[:5] gives us five values to view without tracking gradients for that inspection.
print("Embedding parameters: ", parameter.numel())

inputs = tokenizer(
    text,
    return_tensors="pt",
    add_special_tokens=False,
)

with torch.inference_mode():
    outputs = model(**inputs)

print("Input IDs Shape: ", tuple(inputs["input_ids"].shape))
print("Logits shape:", tuple(outputs.logits.shape))

next_token_logits = outputs.logits[:, -1, :]
print(next_token_logits.shape)


print("EOS token:", tokenizer.eos_token)
print("EOS token ID:", tokenizer.eos_token_id)
print("Generation EOS ID:", model.generation_config.eos_token_id)

next_token_id = next_token_logits.argmax(dim=-1)

print("Predicted token ID:", next_token_id.item())
print("Predicted token:", tokenizer.decode(next_token_id))