import os, sys
import json
import torch

torch.backends.cuda.matmul.allow_tf32 = True

import logging
import copy
from tqdm import tqdm
from datetime import timedelta

import torch.nn as nn
from typing import List, Optional, Union, Tuple
import warnings
import time
import re
import transformers
from PIL import Image
import requests
import torch
import io
from typing import Dict
from transformers import AutoTokenizer, AutoProcessor

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle


import os

from PIL import Image
import pandas as pd
import requests


def preprocess_llama3(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    max_len=2048,
    system_message: str = "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.",
) -> Dict:
    # roles = {"human": "<|start_header_id|>user<|end_header_id|>", "gpt": "<|start_header_id|>assistant<|end_header_id|>"}
    roles = {"human": "user", "gpt": "assistant"}

    # Add image tokens to tokenizer as a special tokens
    # Use a deepcopy of tokenizer so that we don't modify on the tokenizer
    tokenizer = copy.deepcopy(tokenizer)
    # When there is actually an image, we add the image tokens as a special token
    if has_image:
        tokenizer.add_tokens(["<image>"], special_tokens=True)
    image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    bos_token_id = tokenizer.convert_tokens_to_ids("<|begin_of_text|>")
    start_header_id = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
    end_header_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

    unmask_tokens = ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "\n\n"]
    unmask_tokens_idx = [tokenizer.convert_tokens_to_ids(tok) for tok in unmask_tokens]

    # After update, calling tokenizer of llama3 will
    # auto add bos id for the tokens. ヽ(｀⌒´)ﾉ
    def safe_tokenizer_llama3(text):
        input_ids = tokenizer(text).input_ids
        if input_ids[0] == bos_token_id:
            input_ids = input_ids[1:]
        return input_ids

    nl_tokens = tokenizer.convert_tokens_to_ids("\n\n")
    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        # New version, use apply chat template
        # Build system message for each sentence
        input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            # Make sure llava data can load
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role =  roles.get(role, role)
            
            if role == 'assistant':
                input_id += tokenizer.encode("<|start_header_id|>assistant<|end_header_id|>\n\n")
            else:
                conv = [{"role" : role, "content" : content}]
                # First is bos token we don't need here
                encode_id = tokenizer.apply_chat_template(conv)[1:]
                input_id += encode_id
                    
        for idx, encode_id in enumerate(input_id):
            if encode_id == image_token_index:
                input_id[idx] = IMAGE_TOKEN_INDEX
        input_ids.append(input_id)
    input_ids = torch.tensor(input_ids, dtype=torch.long)

    return input_ids

class Llava(nn.Module):
    """
    Llava Model
    """

    def __init__(
        self,
        pretrained: str = "liuhaotian/llava-v1.5-7b",
        truncation: Optional[bool] = True,
        device: Optional[str] = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        trust_remote_code: Optional[bool] = False,
        revision=None,
        use_flash_attention_2=True,
        device_map="auto",
        conv_template="vicuna_v1",
        use_cache=True,
        truncate_context=False,  # whether to truncate the context in generation, set it False for LLaVA-1.6
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(pretrained, None, get_model_name_from_path(pretrained), device_map=device_map, use_flash_attention_2=use_flash_attention_2)
        self._config = self._model.config
        self._model.eval()

        self.device = device



reason_model = Llava(
    pretrained="/path/to/reason_model", 
    device='cuda',
)

summary_model = Llava(
    pretrained="/path/to/summary_model", 
    device='cuda',
)

conv_mode = "llava_llama_3"

image = Image.open('example.png')

origin_question = "<image>\nWhich method is best among the benchmarks?\nA.Baseline.\nB.Chain-of-Thought.\nC.Insight-V.\nAnswer the question with the options provided."

reason_question = origin_question + "\n\nPerform step-by-step reasoning of the problem. Only provide the reasoning process."

# conv = conv_templates[conv_mode].copy()
# conv.append_message(conv.roles[0], question)
# conv.append_message(conv.roles[1], None)
# prompt = conv.get_prompt()

input_ids = preprocess_llama3([[{'from': 'human', 'value': reason_question},{'from': 'gpt','value': None}]], reason_model._tokenizer, has_image=True).cuda()
pad_token_ids = reason_model._tokenizer.pad_token_id if reason_model._tokenizer.pad_token_id is not None else reason_model._tokenizer.eos_token_id
attention_masks = input_ids.ne(pad_token_ids).to('cuda')
image_tensor = process_images([image], reason_model._image_processor, reason_model._config)
if type(image_tensor) is list:
    image_tensor = [_image.to(dtype=torch.bfloat16, device='cuda') for _image in image_tensor]
else:
    image_tensor = image_tensor.to(dtype=torch.bfloat16, device='cuda')

gen_kwargs = {}
gen_kwargs["temperature"] = 0.2
gen_kwargs["max_new_tokens"] = 1024
gen_kwargs["top_p"] = 0.95
if "num_beams" not in gen_kwargs:
    gen_kwargs["num_beams"] = 1
gen_kwargs["image_sizes"] = [image.size]

cont = reason_model._model.generate(
    input_ids,
    attention_mask=attention_masks,
    pad_token_id=pad_token_ids,
    images=image_tensor,
    image_sizes=gen_kwargs["image_sizes"],
    do_sample=True if gen_kwargs["temperature"] > 0 else False,
    temperature=gen_kwargs["temperature"],
    top_p=gen_kwargs["top_p"],
    num_beams=gen_kwargs["num_beams"],
    max_new_tokens=gen_kwargs["max_new_tokens"],
    use_cache=True,
)
text_outputs = reason_model._tokenizer.batch_decode(cont, skip_special_tokens=False)

reason_chain = text_outputs[0].split("<thoughts>")[-1].split("</thoughts>")[0]
reason_chain = "<thoughts>\n" + reason_chain + "</thoughts>\n"

summary_question = "I will give you a reasoning process of the question. You should determine whether the reasoning process is correct about the question. If it is correct, please summarize the answer based on the reasoning process. If it is incorrect, answer the question and ignore the reasoning process. You shold directly give the summarization or answer as you are directly answer the QUESTION  without saying your judgement about the reasoning process.\n\nQUESTION: " + origin_question + f"\n\nREASON PROCESS: {reason_chain}"

input_ids = preprocess_llama3([[{'from': 'human', 'value': summary_question},{'from': 'gpt','value': None}]], summary_model._tokenizer, has_image=True).cuda()
pad_token_ids = summary_model._tokenizer.pad_token_id if summary_model._tokenizer.pad_token_id is not None else summary_model._tokenizer.eos_token_id
attention_masks = input_ids.ne(pad_token_ids).to('cuda')
image_tensor = process_images([image], summary_model._image_processor, summary_model._config)
if type(image_tensor) is list:
    image_tensor = [_image.to(dtype=torch.bfloat16, device='cuda') for _image in image_tensor]
else:
    image_tensor = image_tensor.to(dtype=torch.bfloat16, device='cuda')

gen_kwargs = {}
gen_kwargs["temperature"] = 0.2
gen_kwargs["max_new_tokens"] = 4096
gen_kwargs["top_p"] = 0.95
if "num_beams" not in gen_kwargs:
    gen_kwargs["num_beams"] = 1
gen_kwargs["image_sizes"] = [image.size]

cont = summary_model._model.generate(
    input_ids,
    attention_mask=attention_masks,
    pad_token_id=pad_token_ids,
    images=image_tensor,
    image_sizes=gen_kwargs["image_sizes"],
    do_sample=True if gen_kwargs["temperature"] > 0 else False,
    temperature=gen_kwargs["temperature"],
    top_p=gen_kwargs["top_p"],
    num_beams=gen_kwargs["num_beams"],
    max_new_tokens=gen_kwargs["max_new_tokens"],
    use_cache=True,
)

text_outputs = summary_model._tokenizer.batch_decode(cont, skip_special_tokens=True)

print(text_outputs)