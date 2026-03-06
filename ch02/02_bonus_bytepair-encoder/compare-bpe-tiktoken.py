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

# - Install the additional package requirements for this bonus notebook by uncommenting and running the following cell:

# In[1]:


# pip install -r requirements-extra.txt


# # Comparing Various Byte Pair Encoding (BPE) Implementations

# <br>
# &nbsp;
# 
# ## Using BPE from `tiktoken`

# In[2]:


from importlib.metadata import version

print("tiktoken version:", version("tiktoken"))


# In[3]:


import tiktoken

tik_tokenizer = tiktoken.get_encoding("gpt2")

text = "Hello, world. Is this-- a test?"


# In[4]:


integers = tik_tokenizer.encode(text, allowed_special={"<|endoftext|>"})

print(integers)


# In[5]:


strings = tik_tokenizer.decode(integers)

print(strings)


# In[6]:


print(tik_tokenizer.n_vocab)


# <br>
# &nbsp;
# 
# ## Using the original BPE implementation used in GPT-2

# In[7]:


from bpe_openai_gpt2 import get_encoder, download_vocab


# In[8]:


download_vocab()


# In[9]:


orig_tokenizer = get_encoder(model_name="gpt2_model", models_dir=".")


# In[10]:


integers = orig_tokenizer.encode(text)

print(integers)


# In[11]:


strings = orig_tokenizer.decode(integers)

print(strings)


# <br>
# &nbsp;
# 
# ## Using the BPE via Hugging Face transformers

# In[12]:


import transformers

transformers.__version__


# In[13]:


from transformers import GPT2Tokenizer

hf_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


# In[14]:


hf_tokenizer(strings)["input_ids"]


# <br>
# &nbsp;
# 
# ## A quick performance benchmark

# In[15]:


with open('../01_main-chapter-code/the-verdict.txt', 'r', encoding='utf-8') as f:
    raw_text = f.read()


# In[16]:


get_ipython().run_line_magic('timeit', 'orig_tokenizer.encode(raw_text)')


# In[17]:


get_ipython().run_line_magic('timeit', 'tik_tokenizer.encode(raw_text)')


# In[18]:


get_ipython().run_line_magic('timeit', 'hf_tokenizer(raw_text)["input_ids"]')


# In[19]:


get_ipython().run_line_magic('timeit', 'hf_tokenizer(raw_text, max_length=5145, truncation=True)["input_ids"]')

