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
# 

# # Chapter 2 Exercise solutions

# # Exercise 2.1

# In[1]:


import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")


# In[2]:


integers = tokenizer.encode("Akwirw ier")
print(integers)


# In[3]:


for i in integers:
    print(f"{i} -> {tokenizer.decode([i])}")


# In[4]:


tokenizer.encode("Ak")


# In[5]:


tokenizer.encode("w")


# In[6]:


tokenizer.encode("ir")


# In[7]:


tokenizer.encode("w")


# In[8]:


tokenizer.encode(" ")


# In[9]:


tokenizer.encode("ier")


# In[10]:


tokenizer.decode([33901, 86, 343, 86, 220, 959])


# # Exercise 2.2

# In[11]:


import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(txt, batch_size=4, max_length=256, stride=128):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size)

    return dataloader


with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

tokenizer = tiktoken.get_encoding("gpt2")
encoded_text = tokenizer.encode(raw_text)

vocab_size = 50257
output_dim = 256
max_len = 4
context_length = max_len

token_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)


# In[12]:


dataloader = create_dataloader(raw_text, batch_size=4, max_length=2, stride=2)

for batch in dataloader:
    x, y = batch
    break

x


# In[13]:


dataloader = create_dataloader(raw_text, batch_size=4, max_length=8, stride=2)

for batch in dataloader:
    x, y = batch
    break

x

