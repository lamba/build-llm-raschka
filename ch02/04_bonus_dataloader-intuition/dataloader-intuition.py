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

# ## Data sampling with a sliding window with number data

# In[ ]:


from importlib.metadata import version
import torch

print("torch version:", version("torch"))


# To understand the dataloader, which using a sliding window approach, more intuitive, we can consider a dataset that consists of digits only:
# 
# ```
# 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 ... 1000
# ```

# In[64]:


with open("number-data.txt", "w", encoding="utf-8") as f:
    for number in range(1001):
        f.write(f"{number} ")


# Next, we make a small modification to the `token_ids`: instead of using a tokenizer, we parse the integers directly from the text file:

# In[65]:


from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Modification
        # token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        token_ids = [int(i) for i in txt.strip().split()]

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


# In[66]:


def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):

    # Initialize the tokenizer
    # tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = None

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader


# Let's test the dataloader with a batch size of 1 for an LLM with a context size of 4:

# In[67]:


with open("number-data.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()


# In[68]:


dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)

data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)


# In[69]:


second_batch = next(data_iter)
print(second_batch)


# In[70]:


third_batch = next(data_iter)
print(third_batch)


# In[71]:


for batch in dataloader:
    pass

last_batch = batch
print(last_batch)


# Now, let's look at the batched inputs:

# In[75]:


dataloader = create_dataloader_v1(raw_text, batch_size=2, max_length=4, stride=4, shuffle=False)

for inputs, targets in dataloader:
    pass

print("Inputs:\n", inputs)
print("\nTargets:\n", targets)


# Finally, a data loader with shuffling:

# In[76]:


torch.manual_seed(123)
dataloader = create_dataloader_v1(raw_text, batch_size=2, max_length=4, stride=4, shuffle=True)

for inputs, targets in dataloader:
    pass

print("Inputs:\n", inputs)
print("\nTargets:\n", targets)

