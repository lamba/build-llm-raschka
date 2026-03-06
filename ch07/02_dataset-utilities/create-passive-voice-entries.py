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

# # Create "Passive Voice" Entries for an Instruction Dataset

# - This notebook uses OpenAI's GPT-4 to create "passive voice" entries for an instruction dataset, as shown in the example below
# 
# ```python
# {  
#    'instruction': 'Identify the verb in the following sentence',
#    'input': 'The cat sleeps on the couch.',
#    'output': 'The verb in the sentence is "sleeps."',
#    'output_2': 'The sentence is "sleeps."'   #  <---- Newly created entry
# }  
# ```

# In[1]:


# pip install -r requirements-extra.txt


# In[2]:


from importlib.metadata import version

pkgs = ["openai",  # OpenAI API
        "tqdm",    # Progress bar
       ]

for p in pkgs:
    print(f"{p} version: {version(p)}")


# ## Test OpenAI API

# - First, let's test if the OpenAI API is correctly set up
# - If you don't have an account yet, you need to create one at https://platform.openai.com/
# - Note that you will also have to transfer some funds to your account as the GPT-4 API is not free (see https://platform.openai.com/settings/organization/billing/overview)
# - Creating the ~200 passive voice entries using the code in this notebook costs about $0.13 (13 cents)

# - First, we need to provide our OpenAI API secret key, which can be found at https://platform.openai.com/api-keys
# - Make sure not to share this key with anyone
# - Add this secret key (`"sk-..."`) to the `config.json` file in this folder

# In[3]:


import json
from openai import OpenAI

# Load API key from a JSON file. 
# Make sure to replace "sk-..." with your actual API key from https://platform.openai.com/api-keys
with open("config.json", "r") as config_file:
    config = json.load(config_file)
    api_key = config["OPENAI_API_KEY"]

client = OpenAI(api_key=api_key)


# - First, let's try the API with a simple example to make sure it works as intended:

# In[4]:


def run_chatgpt(prompt, client, model="gpt-4-turbo"):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return response.choices[0].message.content


# Prepare intput
sentence = "I ate breakfast"
prompt = f"Convert the following sentence to passive voice: '{sentence}'"
run_chatgpt(prompt, client)


# ## Create JSON Entries

# - Next, we load the file we want to modify:

# In[5]:


import json

json_file = "instruction-examples.json"

with open(json_file, "r") as file:
    json_data = json.load(file)

print("Number of entries:", len(json_data))


# - And we try the OpenAI chat API on a small sample first to ensure that it works correctly:

# In[6]:


for entry in json_data[:5]:
    text = entry["output"]
    prompt = f"Without adding any response or explanation, convert the following text to passive voice: {text}"

    print("\nInput:")
    print(">>", text)
    print("\nOutput:")
    print(">>", run_chatgpt(prompt, client))
    print("\n-------------------------")


# - Let's now extend the code to add the generated entries to the `json_data` and add a progress bar:

# In[7]:


from tqdm import tqdm  # a progress bar tool


for i, entry in tqdm(enumerate(json_data[:5]), total=len(json_data[:5])):
    text = entry["output"]
    prompt = f"Without adding any response or explanation, convert the following text to passive voice: {text}"
    json_data[i]["output_2"] = run_chatgpt(prompt, client)


# - One more time, let's make sure that the new entries (`"output_2"`) look ok

# In[8]:


json_data[0]


# - Finally, if everything above looks ok, let's run the conversion to passive voice on our entire json dataset (this takes about 3 minutes):

# In[9]:


for i, entry in tqdm(enumerate(json_data), total=len(json_data)):
    text = entry["output"]
    prompt = f"Without adding any response or explanation, convert the following text to passive voice: {text}"
    json_data[i]["output_2"] = run_chatgpt(prompt, client)


# - After the conversion is completed, we save the file:

# In[10]:


new_json_file = json_file.replace(".json", "-modified.json")


with open(new_json_file, "w") as file:
    json.dump(json_data, file, indent=4)  # "indent" for pretty-printing

