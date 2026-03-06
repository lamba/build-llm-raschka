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

# # Chapter 3 Exercise solutions

# # Exercise 3.1

# In[5]:


import torch

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

d_in, d_out = 3, 2


# In[58]:


import torch.nn as nn

class SelfAttention_v1(nn.Module):

    def __init__(self, d_in, d_out):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value

        attn_scores = queries @ keys.T # omega
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        context_vec = attn_weights @ values
        return context_vec

torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)


# In[59]:


class SelfAttention_v2(nn.Module):

    def __init__(self, d_in, d_out):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=False)
        self.W_key   = nn.Linear(d_in, d_out, bias=False)
        self.W_value = nn.Linear(d_in, d_out, bias=False)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=1)

        context_vec = attn_weights @ values
        return context_vec

torch.manual_seed(123)
sa_v2 = SelfAttention_v2(d_in, d_out)


# In[60]:


sa_v1.W_query = torch.nn.Parameter(sa_v2.W_query.weight.T)
sa_v1.W_key = torch.nn.Parameter(sa_v2.W_key.weight.T)
sa_v1.W_value = torch.nn.Parameter(sa_v2.W_value.weight.T)


# In[61]:


sa_v1(inputs)


# In[62]:


sa_v2(inputs)


# # Exercise 3.2

# If we want to have an output dimension of 2, as earlier in single-head attention, we can have to change the projection dimension `d_out` to 1:

# ```python
# torch.manual_seed(123)
# 
# d_out = 1
# mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads=2)
# 
# context_vecs = mha(batch)
# 
# print(context_vecs)
# print("context_vecs.shape:", context_vecs.shape)
# ```

# ```
# tensor([[[-9.1476e-02,  3.4164e-02],
#          [-2.6796e-01, -1.3427e-03],
#          [-4.8421e-01, -4.8909e-02],
#          [-6.4808e-01, -1.0625e-01],
#          [-8.8380e-01, -1.7140e-01],
#          [-1.4744e+00, -3.4327e-01]],
# 
#         [[-9.1476e-02,  3.4164e-02],
#          [-2.6796e-01, -1.3427e-03],
#          [-4.8421e-01, -4.8909e-02],
#          [-6.4808e-01, -1.0625e-01],
#          [-8.8380e-01, -1.7140e-01],
#          [-1.4744e+00, -3.4327e-01]]], grad_fn=<CatBackward0>)
# context_vecs.shape: torch.Size([2, 6, 2])
# ```

# # Exercise 3.3

# ```python
# context_length = 1024
# d_in, d_out = 768, 768
# num_heads = 12
# 
# mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads)
# ```

# Optionally, the number of parameters is as follows:

# ```python
# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)
# 
# count_parameters(mha)
# ```

# ```
# 2360064  # (2.36 M)
# ```

# The GPT-2 model has 117M parameters in total, but as we can see, most of its parameters are not in the multi-head attention module itself.
