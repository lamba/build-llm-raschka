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

# # Generating An Instruction Dataset via Llama 3 and Ollama

# - This notebook uses an 8 billion parameter Llama 3 model through ollama to generate a synthetic dataset using the "hack" proposed in the "Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing" paper ([https://arxiv.org/abs/2406.08464](https://arxiv.org/abs/2406.08464))
# 
# - The generated dataset will be an instruction dataset with "instruction" and "output" field similar to what can be found in Alpaca:
# 
# 
# ```python
# {
#     "instruction": "What is the atomic number of helium?",
#     "output": "The atomic number of helium is 2.",
# },
# ```
# 
# - The code doesn't require a GPU and runs on a laptop (it was tested on a M3 MacBook Air)
# 
# *Note that the instruction datasets created here are for educational purposes. However, it is the users' duty to ensure that their use adheres to the terms of the relevant licensing agreements with Meta AI's Llama 3.*

# In[1]:


from importlib.metadata import version

pkgs = [
    "tqdm",    # Progress bar
]

for p in pkgs:
    print(f"{p} version: {version(p)}")


# ## Installing Ollama and Downloading Llama 3

# - Ollama is an application to run LLMs efficiently
# - It is a wrapper around [llama.cpp](https://github.com/ggerganov/llama.cpp), which implements LLMs in pure C/C++ to maximize efficiency
# - Note that it is a tool for using LLMs to generate text (inference), not training or finetuning LLMs
# - Prior to running the code below, install ollama by visiting [https://ollama.com](https://ollama.com) and following the instructions (for instance, clicking on the "Download" button and downloading the ollama application for your operating system)

# - For macOS and Windows users, click on the ollama application you downloaded; if it prompts you to install the command line usage, say "yes"
# - Linux users can use the installation command provided on the ollama website
# 
# - In general, before we can use ollama from the command line, we have to either start the ollama application or run `ollama serve` in a separate terminal
# 
# <img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/ollama-eval/ollama-serve.webp?1">
# 
# 
# - With the ollama application or `ollama serve` running, in a different terminal, on the command line, execute the following command to try out the 8 billion parameters Llama 3 model (the model, which takes up 4.7 GB of storage space, will be automatically downloaded the first time you execute this command)
# 
# ```bash
# # 8B model
# ollama run llama3
# ```
# 
# 
# The output looks like as follows:
# 
# ```
# $ ollama run llama3
# pulling manifest 
# pulling 6a0746a1ec1a... 100% ▕████████████████▏ 4.7 GB                         
# pulling 4fa551d4f938... 100% ▕████████████████▏  12 KB                         
# pulling 8ab4849b038c... 100% ▕████████████████▏  254 B                         
# pulling 577073ffcc6c... 100% ▕████████████████▏  110 B                         
# pulling 3f8eb4da87fa... 100% ▕████████████████▏  485 B                         
# verifying sha256 digest 
# writing manifest 
# removing any unused layers 
# success 
# ```
# 
# - Note that `llama3` refers to the instruction finetuned 8 billion Llama 3 model
# 
# - Alternatively, you can also use the larger 70 billion parameters Llama 3 model, if your machine supports it, by replacing `llama3` with `llama3:70b`
# 
# - After the download has been completed, you will see a command line prompt that allows you to chat with the model
# 
# - Try a prompt like "What do llamas eat?", which should return an output similar to the following:
# 
# ```
# >>> What do llamas eat?
# Llamas are ruminant animals, which means they have a four-chambered 
# stomach and eat plants that are high in fiber. In the wild, llamas 
# typically feed on:
# 1. Grasses: They love to graze on various types of grasses, including tall 
# grasses, wheat, oats, and barley.
# ```

# - You can end this session using the input `/bye`

# ## Using Ollama's REST API

# - Now, an alternative way to interact with the model is via its REST API in Python via the following function
# - Before you run the next cells in this notebook, make sure that ollama is still running, as described above, via
#   - `ollama serve` in a terminal
#   - the ollama application
# - Next, run the following code cell to query the model

# - First, let's try the API with a simple example to make sure it works as intended:

# In[2]:


import urllib.request
import json

def query_model(prompt, model="llama3", url="http://localhost:11434/api/chat", role="user"):
    # Create the data payload as a dictionary
    data = {
        "model": model,
        "seed": 123,        # for deterministic responses
        "temperature": 1.,   # for deterministic responses
        "top_p": 1,         
        "messages": [
            {"role": role, "content": prompt}
        ]
    }

    # Convert the dictionary to a JSON formatted string and encode it to bytes
    payload = json.dumps(data).encode("utf-8")

    # Create a request object, setting the method to POST and adding necessary headers
    request = urllib.request.Request(url, data=payload, method="POST")
    request.add_header("Content-Type", "application/json")

    # Send the request and capture the response
    response_data = ""
    with urllib.request.urlopen(request) as response:
        # Read and decode the response
        while True:
            line = response.readline().decode("utf-8")
            if not line:
                break
            response_json = json.loads(line)
            response_data += response_json["message"]["content"]

    return response_data


# In[3]:


result = query_model("What do Llamas eat?")
print(result)


# ## Extract Instructions

# - Now, let's use the "hack" proposed in the paper: we provide the empty prompt template `"<|begin_of_text|><|start_header_id|>user<|end_header_id|>"` prompt, which will cause the instruction-finetuned Llama 3 model to generate an instruction

# In[4]:


def extract_instruction(text):
    for content in text.split("\n"):
        if content:
            return content.strip()


# In[5]:


query = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>"

result = query_model(query, role="assistant")
instruction = extract_instruction(result)
print(instruction)


# - As we can see above, surprisingly, the model indeed generated an instruction

# ## Generate Responses

# - Now, the next step is to create the corresponding response, which can be done by simply passing the instruction as input

# In[6]:


response = query_model(instruction, role="user")
print(response)


# ## Generate Dataset

# - We can scale up this approach to an arbitrary number of data samples (you may want to apply some optional filtering length or quality (e.g., using another LLM to rate the generated data)
# - Below, we generate 5 synthetic instruction-response pairs, which takes about 3 minutes on an M3 MacBook Air
# - (To generate a dataset suitable for instruction finetuning, we want to increase this to at least 1k to 50k and perhaps run it on a GPU to generate the examples in a more timely fashion)
# 
# **Tip**
# 
# - You can generate even higher-quality responses by changing `model="llama3"` to `model="llama3:70b"`, however, this will require more computational resources

# In[7]:


from tqdm import tqdm

dataset_size = 5
dataset = []

for i in tqdm(range(dataset_size)):

    result = query_model(query, role="assistant")
    instruction = extract_instruction(result)
    response = query_model(instruction, role="user")
    entry = {
        "instruction": instruction,
        "output": response
    }
    dataset.append(entry)


# In[8]:


with open("instruction-data-llama3-7b.json", "w") as file:
    json.dump(dataset, file, indent=4)


# In[9]:


get_ipython().system('cat instruction-data-llama3-7b.json')

