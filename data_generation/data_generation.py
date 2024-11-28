import time
import json
import glob
from PIL import Image
import io
import os
import base64
import tqdm
from dataclasses import dataclass
from transformers import AutoProcessor
from transformers.hf_argparser import HfArgumentParser
from vllm import LLM, SamplingParams
from vision_process import process_vision_info

MODEL_PATH = "/path/to/qwen2-vl"

llm = LLM(
    model=MODEL_PATH,
    limit_mm_per_prompt={"image": 10, "video": 10},
)

processor = AutoProcessor.from_pretrained(MODEL_PATH)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.05,
    max_tokens=1024,
    stop_token_ids=[],
)

def make_api_call(messages, max_tokens, is_final_answer=False):
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)

    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs

    llm_inputs = {
        "prompt": prompt,
        "multi_modal_data": mm_data,
    }


    outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
    generated_text = outputs[0].outputs[0].text

    print(generated_text)
    return generated_text


def generate_response(prompt, img_query=None, rounds=5):
    messages = [
            {
                "role": "system", 
                "content": 
                """You are an expert AI assistant that explains your reasoning step by step. For each step, provide a title that describes what you're doing in that step, along with the content. Decide if you need another step or if you're ready to give the final answer. Respond in JSON format with 'title', 'content', and 'next_action' (either 'continue' or 'final_answer') keys. USE AT LEAST 2 STEPS OF REASONING. BE AWARE OF YOUR LIMITATIONS AS AN LLM AND WHAT YOU CAN AND CANNOT DO. IN YOUR REASONING, INCLUDE EXPLORATION OF ALTERNATIVE ANSWERS. CONSIDER YOU MAY BE WRONG, AND IF YOU ARE WRONG IN YOUR REASONING, WHERE IT WOULD BE. FULLY TEST ALL OTHER POSSIBILITIES. YOU CAN BE WRONG. WHEN YOU SAY YOU ARE RE-EXAMINING, ACTUALLY RE-EXAMINE, AND USE ANOTHER APPROACH TO DO SO. DO NOT JUST SAY YOU ARE RE-EXAMINING. USE AT LEAST 3 METHODS TO DERIVE THE ANSWER. USE BEST PRACTICES.
                Example of a valid JSON response:
                {
                "title": "Identifying Key Information",
                "content": "To begin solving this problem, we need to carefully examine the given information and identify the crucial elements that will guide our solution process. This involves...",
                "next_action": "continue"
                }
                STRICTLY FOLLOW THE JSON RESPONSE FORMAT. THE RESPONDE SHOULD START WITH "{". DO NOT START WITH "```json" OR ANYTHING ELSE.
                """
            },
        {"role": "user", "content": [{"type": "image", "image": img_query, "min_pixels": 224 * 224, "max_pixels": 1280 * 28 * 28},{"type": "text", "text":prompt}]},
        {"role": "assistant", "content": "Thank you! I will now think step by step following my instructions, starting at the beginning after decomposing the problem."}
    ]

    steps = []
    step_count = 1
    total_thinking_time = 0
    
    while True:
        for retry in range(3):
            try:
                start_time = time.time()
                step_data = make_api_call(messages, 300)
                end_time = time.time()
                thinking_time = end_time - start_time
                total_thinking_time += thinking_time

                step_data = json.loads(step_data)

                steps.append({
                    "count" : step_count,
                    "title": step_data['title'],
                    "content": step_data['content'],
                    "next_action": step_data['next_action']
                })
                break
            except Exception as e:
                messages.append(
                    {"role": "user", "content": """
                    Please response in a valid JSON response.
                    Response Format:
                    {
                        "title": "xxx", (Summarize this step response)
                        "content": "xxx", (Please provide an one-more different step analysis here that you believe is likely to answer the question correctly)
                        "next_action" "xxx",  ("continue" or "final_answer")
                    }
                    """}
                )
                if retry == 2:
                    step_data = {"title": "Error", "content": f"Failed to generate step after 3 attempts. Error: {str(e)}",
                            "next_action": "final_answer"}
                    break
        
        messages.append({"role": "assistant", "content": json.dumps(step_data)})
        
        if step_data['next_action'] == 'final_answer' or step_count > 15:
            break
        else:
            messages.append(
                {"role": "user", "content": """
                Please continue provide detailed, step-by-step explanations of your thought process. And response in a valid JSON response.
                Use JSON with keys: 'title', 'content', 'next_action' (values: 'continue' if you think you need to continue explain to answer the question correctly, or 'final_answer' if you think you can correctly answer the question based on the above step-by-step explaintion)
                Noting: the value of 'next_action' is continue, the content of this step must be very different from previous content. Otherwise, please set 'next_action' to final_answer.
                Example of a valid JSON response:
                {
                "title": "xxx",(Summarize this step response)
                "content": "xxx", (Please provide an one-more different step analysis here that you believe is likely to answer the question correctly)
                "next_action" "xxx",  ("continue" or "final_answer")
                }
                If "next_action" in previous round is "continue", then you have to strictly follow the JSON format and DO NOT DIRECTLY PROVIDE THE ANSWER IN THIS ROUND. The JSON response must have "title", "content" and "next_action".
                YOU SHOULD ONLY PERFORM ONE STEP REASONING AT A TIME. DO NOT PROVIDE MULTIPLE STEPS IN ONE RESPONSE.
                """}
            )

        step_count += 1

        # Yield after each step for Streamlit to update
        yield steps, None  # We're not yielding the total time until the end
                    

    # Generate final answer
    messages.append({"role": "user", 
        "content": """Please provide the final answer based solely on your reasoning above. USE JSON FORMAT. Retain any formatting as instructed by the original prompt, such as exact formatting for free response or multiple choice.
        Example of a valid FINAL ANSWER response:
        {
        "title": "FINAL ANSWER",
        "content": "The answer to the question"
        }
        MAKE SURE TO FOLLOW the formatting as instructed by the original prompt in the answer.
        """
        })
    
    start_time = time.time()
    final_data = make_api_call(messages, 1200, is_final_answer=True)
    end_time = time.time()
    thinking_time = end_time - start_time
    total_thinking_time += thinking_time
    
    steps.append(("Final Answer", final_data, thinking_time))

    yield steps, total_thinking_time

def inference(user_query, img_query):
    steps = []
    total_thinking_time = 0
    for steps_part, thinking_time in generate_response(user_query, img_query):
        steps.extend(steps_part)
        total_thinking_time = thinking_time
        
    return total_thinking_time, steps