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

# ## FLOPS Analysis

# - FLOPs (Floating Point Operations Per Second) measure the computational complexity of neural network models by counting the number of floating-point operations executed
# - High FLOPs indicate more intensive computation and energy consumption

# In[1]:


# pip install -r requirements-extra.txt


# In[2]:


from importlib.metadata import version

import matplotlib
import torch

print("thop version:", version("thop"))
print("torch version:", version("torch"))


# In[3]:


import torch
from thop import profile

from previous_chapters import GPTModel


BASE_CONFIG = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "drop_rate": 0.0,        # Dropout rate
    "qkv_bias": True         # Query-key-value bias
}

model_configs = {
    "gpt-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_tensor = torch.randint(0, 50257, (2, 1024)).to(device)

for size in model_configs:
    BASE_CONFIG.update(model_configs[size])

    model = GPTModel(BASE_CONFIG).bfloat16()
    model.to(device)

    # MACS = multiply-accumulate operations
    # MACS are typically counted as two FLOPS (one multiply and one accumulate)
    macs, params = profile(model, inputs=(input_tensor,), verbose=False)
    flops = 2*macs
    print(f"{size:18}: {flops:.1e} FLOPS")

    del model
    torch.cuda.empty_cache()

