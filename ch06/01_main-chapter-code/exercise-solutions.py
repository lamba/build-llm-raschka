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

# # Chapter 6 Exercise solutions

# ## Exercise 6.1: Increasing the context length

# We can pad the inputs to the maximum number of tokens the model supports by setting the max length to 1024:
# 
# ```python
# max_length = 1024
# 
# train_dataset = SpamDataset(base_path / "train.csv", max_length=max_length, tokenizer=tokenizer)
# val_dataset = SpamDataset(base_path / "validation.csv", max_length=max_length, tokenizer=tokenizer)
# test_dataset = SpamDataset(base_path / "test.csv", max_length=max_length, tokenizer=tokenizer)
# ```
# 
# or, equivalently, we can define the `max_length` via:
# 
# ```python
# max_length = model.pos_emb.weight.shape[0]
# ```
# 
# or
# 
# ```python
# max_length = BASE_CONFIG["context_length"]
# ```

# For convenience, you can run this experiment via
# 
# ```bash
# python additional-experiments.py --context_length "model_context_length"
# ```
# 
# using the code in the [../02_bonus_additional-experiments](../02_bonus_additional-experiments) folder, which results in a substantially worse test accuracy of 78.33% (versus the 95.67% in the main chapter).

# ## Exercise 6.2: Finetuning the whole model

# Instead of finetuning just the final transformer block, we can finetune the entire model by removing the following lines from the code:
# 
# ```python
# for param in model.parameters():
#     param.requires_grad = False
# ```
# 
# For convenience, you can run this experiment via
# 
# ```bash
# python additional-experiments.py --trainable_layers all
# ```
# 
# using the code in the [../02_bonus_additional-experiments](../02_bonus_additional-experiments) folder, which results in a 1% improved test accuracy of 96.67% (versus the 95.67% in the main chapter).

# ## Exercise 6.3: Finetuning the first versus last token 

# Rather than finetuning the last output token, we can finetune the first output token by changing 
# 
# ```python
# model(input_batch)[:, -1, :]
# ```
# 
# to
# 
# ```python
# model(input_batch)[:, 0, :]
# ```
# 
# everywhere in the code.
# 
# For convenience, you can run this experiment via
# 
# ```
# python additional-experiments.py --trainable_token first
# ```
# 
# using the code in the [../02_bonus_additional-experiments](../02_bonus_additional-experiments) folder, which results in a substantially worse test accuracy of 75.00% (versus the 95.67% in the main chapter).
