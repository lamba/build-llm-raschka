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

# # Understanding the Difference Between Embedding Layers and Linear Layers

# - Embedding layers in PyTorch accomplish the same as linear layers that perform matrix multiplications; the reason we use embedding layers is computational efficiency
# - We will take a look at this relationship step by step using code examples in PyTorch

# In[1]:


import torch

print("PyTorch version:", torch.__version__)


# <br>
# &nbsp;
# 
# ## Using nn.Embedding

# In[2]:


# Suppose we have the following 3 training examples,
# which may represent token IDs in a LLM context
idx = torch.tensor([2, 3, 1])

# The number of rows in the embedding matrix can be determined
# by obtaining the largest token ID + 1.
# If the highest token ID is 3, then we want 4 rows, for the possible
# token IDs 0, 1, 2, 3
num_idx = max(idx)+1

# The desired embedding dimension is a hyperparameter
out_dim = 5


# - Let's implement a simple embedding layer:

# In[3]:


# We use the random seed for reproducibility since
# weights in the embedding layer are initialized with
# small random values
torch.manual_seed(123)

embedding = torch.nn.Embedding(num_idx, out_dim)


# We can optionally take a look at the embedding weights:

# In[4]:


embedding.weight


# - We can then use the embedding layers to obtain the vector representation of a training example with ID 1:

# In[5]:


embedding(torch.tensor([1]))


# - Below is a visualization of what happens under the hood:

# <img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/embeddings-and-linear-layers/1.png" width="400px">

# - Similarly, we can use embedding layers to obtain the vector representation of a training example with ID 2:

# In[6]:


embedding(torch.tensor([2]))


# <img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/embeddings-and-linear-layers/2.png" width="400px">

# - Now, let's convert all the training examples we have defined previously:

# In[7]:


idx = torch.tensor([2, 3, 1])
embedding(idx)


# - Under the hood, it's still the same look-up concept:

# <img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/embeddings-and-linear-layers/3.png" width="450px">

# <br>
# &nbsp;
# 
# ## Using nn.Linear

# - Now, we will demonstrate that the embedding layer above accomplishes exactly the same as `nn.Linear` layer on a one-hot encoded representation in PyTorch
# - First, let's convert the token IDs into a one-hot representation:

# In[8]:


onehot = torch.nn.functional.one_hot(idx)
onehot


# - Next, we initialize a `Linear` layer, which caries out a matrix multiplication $X W^\top$:

# In[9]:


torch.manual_seed(123)
linear = torch.nn.Linear(num_idx, out_dim, bias=False)
linear.weight


# - Note that the linear layer in PyTorch is also initialized with small random weights; to directly compare it to the `Embedding` layer above, we have to use the same small random weights, which is why we reassign them here:

# In[10]:


linear.weight = torch.nn.Parameter(embedding.weight.T.detach())


# - Now we can use the linear layer on the one-hot encoded representation of the inputs:

# In[11]:


linear(onehot.float())


# As we can see, this is exactly the same as what we got when we used the embedding layer:

# In[12]:


embedding(idx)


# - What happens under the hood is the following computation for the first training example's token ID:

# <img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/embeddings-and-linear-layers/4.png" width="450px">

# - And for the second training example's token ID:

# <img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/embeddings-and-linear-layers/5.png" width="450px">

# - Since all but one index in each one-hot encoded row are 0 (by design), this matrix multiplication is essentially the same as a look-up of the one-hot elements
# - This use of the matrix multiplication on one-hot encodings is equivalent to the embedding layer look-up but can be inefficient if we work with large embedding matrices, because there are a lot of wasteful multiplications by zero
