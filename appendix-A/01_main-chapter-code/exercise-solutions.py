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

# ## Exercise A.1

# The [Python Setup Tips](../../setup/01_optional-python-setup-preferences/README.md) document in this repository contains additional recommendations and tips to set up your Python environment.
# 

# ## Exercise A.2

# The [Installing Libraries Used In This Book document](../../setup/02_installing-python-libraries/README.md) and [directory](../../setup/02_installing-python-libraries/) contains utilities to check whether your environment is set up correctly.

# ## Exercise A.3

# In[2]:


import torch

class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.layers = torch.nn.Sequential(

            # 1st hidden layer
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),

            # 2nd hidden layer
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),

            # output layer
            torch.nn.Linear(20, num_outputs),
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits


# In[3]:


model = NeuralNetwork(2, 2)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of trainable model parameters:", num_params)


# ## Exercise A.4

# In[1]:


import torch

a = torch.rand(100, 200)
b = torch.rand(200, 300)


# In[2]:


get_ipython().run_line_magic('timeit', 'a @ b')


# In[3]:


a, b = a.to("cuda"), b.to("cuda")


# In[4]:


get_ipython().run_line_magic('timeit', 'a @ b')

