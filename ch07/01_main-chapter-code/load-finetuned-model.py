#!/usr/bin/env python
# coding: utf-8

# <table style="width:100%">
# <tr>
# <td style="vertical-align:middle; text-align:left;">
# <font size="2">
# Supplementary code for the <a href="http://mng.bz/orYv">Build a Large Language Model From Scratch</a> book by <a href="https://sebastianraschka.com">Sebastian Raschka</a><br>
# <br>Code repository: <a href="https://github.com/rasbt/LLMs-from-scratch">https://github.com/rasbt/LLMs-from-scratch</a>
# </font>
# </td>
# <td style="vertical-align:middle; text-align:left;">
# <a href="http://mng.bz/orYv"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/cover-small.webp" width="100px"></a>
# </td>
# </tr>
# </table>

# # Load And Use Finetuned Model

# This notebook contains minimal code to load the finetuned model that was instruction finetuned and saved in chapter 7 via [ch07.ipynb](ch07.ipynb).

# In[1]:


from importlib.metadata import version

pkgs = [
    "tiktoken",    # Tokenizer
    "torch",       # Deep learning library
]
for p in pkgs:
    print(f"{p} version: {version(p)}")


# In[2]:


from pathlib import Path

finetuned_model_path = Path("gpt2-medium355M-sft.pth")
if not finetuned_model_path.exists():
    print(
        f"Could not find '{finetuned_model_path}'.\n"
        "Run the `ch07.ipynb` notebook to finetune and save the finetuned model."
    )


# In[3]:


from previous_chapters import GPTModel


BASE_CONFIG = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "drop_rate": 0.0,        # Dropout rate
    "qkv_bias": True         # Query-key-value bias
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

CHOOSE_MODEL = "gpt2-medium (355M)"

BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
model = GPTModel(BASE_CONFIG)


# In[4]:


import torch

model.load_state_dict(torch.load("gpt2-medium355M-sft.pth", map_location=torch.device("cpu")))
model.eval();


# In[5]:


import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")


# In[6]:


prompt = """Below is an instruction that describes a task. Write a response 
that appropriately completes the request.

### Instruction:
Convert the active sentence to passive: 'The chef cooks the meal every day.'
"""


# In[7]:


from previous_chapters import (
    generate,
    text_to_token_ids,
    token_ids_to_text
)

def extract_response(response_text, input_text):
    return response_text[len(input_text):].replace("### Response:", "").strip()

torch.manual_seed(123)

token_ids = generate(
    model=model,
    idx=text_to_token_ids(prompt, tokenizer),
    max_new_tokens=35,
    context_size=BASE_CONFIG["context_length"],
    eos_id=50256
)

response = token_ids_to_text(token_ids, tokenizer)
response = extract_response(response, prompt)
print(response)

