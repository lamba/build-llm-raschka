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

# # Appendix A: Introduction to PyTorch (Part 1)

# ## A.1 What is PyTorch

# In[1]:


import torch

print(torch.__version__)


# In[2]:


print(torch.cuda.is_available())


# <img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/appendix-a_compressed/1.webp" width="400px">

# <img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/appendix-a_compressed/2.webp" width="300px">
# 
# <img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/appendix-a_compressed/3.webp" width="300px">
# 
# <img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/appendix-a_compressed/4.webp" width="500px">
# 
# <img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/appendix-a_compressed/5.webp" width="500px">

# ## A.2 Understanding tensors

# <img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/appendix-a_compressed/6.webp" width="400px">

# ### A.2.1 Scalars, vectors, matrices, and tensors

# In[3]:


import torch
import numpy as np

# create a 0D tensor (scalar) from a Python integer
tensor0d = torch.tensor(1)

# create a 1D tensor (vector) from a Python list
tensor1d = torch.tensor([1, 2, 3])

# create a 2D tensor from a nested Python list
tensor2d = torch.tensor([[1, 2], 
                         [3, 4]])

# create a 3D tensor from a nested Python list
tensor3d_1 = torch.tensor([[[1, 2], [3, 4]], 
                           [[5, 6], [7, 8]]])

# create a 3D tensor from NumPy array
ary3d = np.array([[[1, 2], [3, 4]], 
                  [[5, 6], [7, 8]]])
tensor3d_2 = torch.tensor(ary3d)  # Copies NumPy array
tensor3d_3 = torch.from_numpy(ary3d)  # Shares memory with NumPy array


# In[4]:


ary3d[0, 0, 0] = 999
print(tensor3d_2) # remains unchanged


# In[5]:


print(tensor3d_3) # changes because of memory sharing


# ### A.2.2 Tensor data types

# In[6]:


tensor1d = torch.tensor([1, 2, 3])
print(tensor1d.dtype)


# In[7]:


floatvec = torch.tensor([1.0, 2.0, 3.0])
print(floatvec.dtype)


# In[8]:


floatvec = tensor1d.to(torch.float32)
print(floatvec.dtype)


# ### A.2.3 Common PyTorch tensor operations

# In[9]:


tensor2d = torch.tensor([[1, 2, 3], 
                         [4, 5, 6]])
tensor2d


# In[10]:


tensor2d.shape


# In[11]:


tensor2d.reshape(3, 2)


# In[12]:


tensor2d.view(3, 2)


# In[13]:


tensor2d.T


# In[14]:


tensor2d.matmul(tensor2d.T)


# In[15]:


tensor2d @ tensor2d.T


# ## A.3 Seeing models as computation graphs

# <img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/appendix-a_compressed/7.webp" width="600px">

# In[16]:


import torch.nn.functional as F

y = torch.tensor([1.0])  # true label
x1 = torch.tensor([1.1]) # input feature
w1 = torch.tensor([2.2]) # weight parameter
b = torch.tensor([0.0])  # bias unit

z = x1 * w1 + b          # net input
a = torch.sigmoid(z)     # activation & output

loss = F.binary_cross_entropy(a, y)
print(loss)


# ## A.4 Automatic differentiation made easy

# <img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/appendix-a_compressed/8.webp" width="600px">

# In[17]:


import torch.nn.functional as F
from torch.autograd import grad

y = torch.tensor([1.0])
x1 = torch.tensor([1.1])
w1 = torch.tensor([2.2], requires_grad=True)
b = torch.tensor([0.0], requires_grad=True)

z = x1 * w1 + b 
a = torch.sigmoid(z)

loss = F.binary_cross_entropy(a, y)

grad_L_w1 = grad(loss, w1, retain_graph=True)
grad_L_b = grad(loss, b, retain_graph=True)

print(grad_L_w1)
print(grad_L_b)


# In[18]:


loss.backward()

print(w1.grad)
print(b.grad)


# ## A.5 Implementing multilayer neural networks

# <img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/appendix-a_compressed/9.webp" width="500px">

# In[19]:


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


# In[20]:


model = NeuralNetwork(50, 3)


# In[21]:


print(model)


# In[22]:


num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of trainable model parameters:", num_params)


# In[23]:


print(model.layers[0].weight)


# In[24]:


torch.manual_seed(123)

model = NeuralNetwork(50, 3)
print(model.layers[0].weight)


# In[25]:


print(model.layers[0].weight.shape)


# In[26]:


torch.manual_seed(123)

X = torch.rand((1, 50))
out = model(X)
print(out)


# In[27]:


with torch.no_grad():
    out = model(X)
print(out)


# In[28]:


with torch.no_grad():
    out = torch.softmax(model(X), dim=1)
print(out)


# ## A.6 Setting up efficient data loaders

# <img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/appendix-a_compressed/10.webp" width="600px">

# In[29]:


X_train = torch.tensor([
    [-1.2, 3.1],
    [-0.9, 2.9],
    [-0.5, 2.6],
    [2.3, -1.1],
    [2.7, -1.5]
])

y_train = torch.tensor([0, 0, 0, 1, 1])


# In[30]:


X_test = torch.tensor([
    [-0.8, 2.8],
    [2.6, -1.6],
])

y_test = torch.tensor([0, 1])


# In[31]:


from torch.utils.data import Dataset


class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y

    def __getitem__(self, index):
        one_x = self.features[index]
        one_y = self.labels[index]        
        return one_x, one_y

    def __len__(self):
        return self.labels.shape[0]

train_ds = ToyDataset(X_train, y_train)
test_ds = ToyDataset(X_test, y_test)


# In[32]:


len(train_ds)


# In[33]:


from torch.utils.data import DataLoader

torch.manual_seed(123)

train_loader = DataLoader(
    dataset=train_ds,
    batch_size=2,
    shuffle=True,
    num_workers=0
)


# In[34]:


test_ds = ToyDataset(X_test, y_test)

test_loader = DataLoader(
    dataset=test_ds,
    batch_size=2,
    shuffle=False,
    num_workers=0
)


# In[35]:


for idx, (x, y) in enumerate(train_loader):
    print(f"Batch {idx+1}:", x, y)


# In[36]:


train_loader = DataLoader(
    dataset=train_ds,
    batch_size=2,
    shuffle=True,
    num_workers=0,
    drop_last=True
)


# In[37]:


for idx, (x, y) in enumerate(train_loader):
    print(f"Batch {idx+1}:", x, y)


# <img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/appendix-a_compressed/11.webp" width="600px">

# ## A.7 A typical training loop

# In[38]:


import torch.nn.functional as F


torch.manual_seed(123)
model = NeuralNetwork(num_inputs=2, num_outputs=2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

num_epochs = 3

for epoch in range(num_epochs):

    model.train()
    for batch_idx, (features, labels) in enumerate(train_loader):

        logits = model(features)

        loss = F.cross_entropy(logits, labels) # Loss function

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ### LOGGING
        print(f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
              f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
              f" | Train/Val Loss: {loss:.2f}")

    model.eval()
    # Optional model evaluation


# In[39]:


model.eval()

with torch.no_grad():
    outputs = model(X_train)

print(outputs)


# In[40]:


torch.set_printoptions(sci_mode=False)
probas = torch.softmax(outputs, dim=1)
print(probas)

predictions = torch.argmax(probas, dim=1)
print(predictions)


# In[41]:


predictions = torch.argmax(outputs, dim=1)
print(predictions)


# In[42]:


predictions == y_train


# In[43]:


torch.sum(predictions == y_train)


# In[44]:


def compute_accuracy(model, dataloader):

    model = model.eval()
    correct = 0.0
    total_examples = 0

    for idx, (features, labels) in enumerate(dataloader):

        with torch.no_grad():
            logits = model(features)

        predictions = torch.argmax(logits, dim=1)
        compare = labels == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)

    return (correct / total_examples).item()


# In[45]:


compute_accuracy(model, train_loader)


# In[46]:


compute_accuracy(model, test_loader)


# ## A.8 Saving and loading models

# In[47]:


torch.save(model.state_dict(), "model.pth")


# In[48]:


model = NeuralNetwork(2, 2) # needs to match the original model exactly
model.load_state_dict(torch.load("model.pth"))


# ## A.9 Optimizing training performance with GPUs

# ### A.9.1 PyTorch computations on GPU devices

# See [code-part2.ipynb](code-part2.ipynb)

# ### A.9.2 Single-GPU training

# See [code-part2.ipynb](code-part2.ipynb)

# ### A.9.3 Training with multiple GPUs

# See [DDP-script.py](DDP-script.py)
