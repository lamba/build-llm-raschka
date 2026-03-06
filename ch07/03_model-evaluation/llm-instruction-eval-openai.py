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

# # Evaluating Instruction Responses Using the OpenAI API

# - This notebook uses OpenAI's GPT-4 API to evaluate responses by a instruction finetuned LLMs based on an dataset in JSON format that includes the generated model responses, for example:
# 
# 
# 
# ```python
# {
#     "instruction": "What is the atomic number of helium?",
#     "input": "",
#     "output": "The atomic number of helium is 2.",               # <-- The target given in the test set
#     "model 1 response": "\nThe atomic number of helium is 2.0.", # <-- Response by an LLM
#     "model 2 response": "\nThe atomic number of helium is 3."    # <-- Response by a 2nd LLM
# },
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
# - Running the experiments and creating the ~200 evaluations using the code in this notebook costs about $0.26 (26 cents) as of this writing

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
        seed=123,
    )
    return response.choices[0].message.content


prompt = f"Respond with 'hello world' if you got this message."
run_chatgpt(prompt, client)


# ## Load JSON Entries

# - Here, we assume that we saved the test dataset and the model responses as a JSON file that we can load as follows:

# In[5]:


json_file = "eval-example-data.json"

with open(json_file, "r") as file:
    json_data = json.load(file)

print("Number of entries:", len(json_data))


# - The structure of this file is as follows, where we have the given response in the test dataset (`'output'`) and responses by two different models (`'model 1 response'` and `'model 2 response'`):

# In[6]:


json_data[0]


# - Below is a small utility function that formats the input for visualization purposes later:

# In[7]:


def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. Write a response that "
        f"appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    instruction_text + input_text

    return instruction_text + input_text


# - Now, let's try the OpenAI API to compare the model responses (we only evaluate the first 5 responses for a visual comparison):

# In[8]:


for entry in json_data[:5]:
    prompt = (f"Given the input `{format_input(entry)}` "
              f"and correct output `{entry['output']}`, "
              f"score the model response `{entry['model 1 response']}`"
              f" on a scale from 0 to 100, where 100 is the best score. "
              )
    print("\nDataset response:")
    print(">>", entry['output'])
    print("\nModel response:")
    print(">>", entry["model 1 response"])
    print("\nScore:")
    print(">>", run_chatgpt(prompt, client))
    print("\n-------------------------")


# - Note that the responses are very verbose; to quantify which model is better, we only want to return the scores:

# In[9]:


from tqdm import tqdm


def generate_model_scores(json_data, json_key, client):
    scores = []
    for entry in tqdm(json_data, desc="Scoring entries"):
        prompt = (
            f"Given the input `{format_input(entry)}` "
            f"and correct output `{entry['output']}`, "
            f"score the model response `{entry[json_key]}`"
            f" on a scale from 0 to 100, where 100 is the best score. "
            f"Respond with the number only."
        )
        score = run_chatgpt(prompt, client)
        try:
            scores.append(int(score))
        except ValueError:
            continue

    return scores


# - Please note that the response scores may vary because OpenAI's GPT models are not deterministic despite setting a random number seed, etc.

# - Let's now apply this evaluation to the whole dataset and compute the average score of each model:

# In[11]:


from pathlib import Path

for model in ("model 1 response", "model 2 response"):

    scores = generate_model_scores(json_data, model, client)
    print(f"\n{model}")
    print(f"Number of scores: {len(scores)} of {len(json_data)}")
    print(f"Average score: {sum(scores)/len(scores):.2f}\n")

    # Optionally save the scores
    save_path = Path("scores") / f"gpt4-{model.replace(' ', '-')}.json"
    with open(save_path, "w") as file:
        json.dump(scores, file)


# - Based on the evaluation above, we can say that the 1st model is substantially better than the 2nd model
