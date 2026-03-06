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

# # Evaluating Instruction Responses Locally Using a Llama 3 Model Via Ollama

# - This notebook uses an 8 billion parameter Llama 3 model through ollama to evaluate responses of instruction finetuned LLMs based on a dataset in JSON format that includes the generated model responses, for example:
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
# 
# - The code doesn't require a GPU and runs on a laptop (it was tested on a M3 MacBook Air)

# In[1]:


from importlib.metadata import version

pkgs = ["tqdm",    # Progress bar
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


def query_model(prompt, model="llama3", url="http://localhost:11434/api/chat"):
    # Create the data payload as a dictionary
    data = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "options": {     # Settings below are required for deterministic responses
            "seed": 123,
            "temperature": 0,
            "num_ctx": 2048
        }
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


result = query_model("What do Llamas eat?")
print(result)


# ## Load JSON Entries

# - Now, let's get to the data evaluation part
# - Here, we assume that we saved the test dataset and the model responses as a JSON file that we can load as follows:

# In[3]:


json_file = "eval-example-data.json"

with open(json_file, "r") as file:
    json_data = json.load(file)

print("Number of entries:", len(json_data))


# - The structure of this file is as follows, where we have the given response in the test dataset (`'output'`) and responses by two different models (`'model 1 response'` and `'model 2 response'`):

# In[4]:


json_data[0]


# - Below is a small utility function that formats the input for visualization purposes later:

# In[5]:


def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. Write a response that "
        f"appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    instruction_text + input_text

    return instruction_text + input_text


# - Now, let's try the ollama API to compare the model responses (we only evaluate the first 5 responses for a visual comparison):

# In[6]:


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
    print(">>", query_model(prompt))
    print("\n-------------------------")


# - Note that the responses are very verbose; to quantify which model is better, we only want to return the scores:

# In[7]:


from tqdm import tqdm


def generate_model_scores(json_data, json_key):
    scores = []
    for entry in tqdm(json_data, desc="Scoring entries"):
        prompt = (
            f"Given the input `{format_input(entry)}` "
            f"and correct output `{entry['output']}`, "
            f"score the model response `{entry[json_key]}`"
            f" on a scale from 0 to 100, where 100 is the best score. "
            f"Respond with the integer number only."
        )
        score = query_model(prompt)
        try:
            scores.append(int(score))
        except ValueError:
            continue

    return scores


# - Let's now apply this evaluation to the whole dataset and compute the average score of each model (this takes about 1 minute per model on an M3 MacBook Air laptop)
# - Note that ollama is not fully deterministic across operating systems (as of this writing) so the numbers you are getting might slightly differ from the ones shown below

# In[8]:


from pathlib import Path

for model in ("model 1 response", "model 2 response"):

    scores = generate_model_scores(json_data, model)
    print(f"\n{model}")
    print(f"Number of scores: {len(scores)} of {len(json_data)}")
    print(f"Average score: {sum(scores)/len(scores):.2f}\n")

    # Optionally save the scores
    save_path = Path("scores") / f"llama3-8b-{model.replace(' ', '-')}.json"
    with open(save_path, "w") as file:
        json.dump(scores, file)


# - Based on the evaluation above, we can say that the 1st model is better than the 2nd model
