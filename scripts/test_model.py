import re
import os
import torch
import random
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI
from util import *

# Global system prompt
SYSTEM_PROMPT = """
You are tasked with converting a given Planning Domain Definition Language (PDDL) domain description into its corresponding formal PDDL domain. The description will outline the essential components of the domains. 
Your output should be a well-structured PDDL domain that accurately represents the given description, adhering to the syntax and semantics of PDDL.
Your output pddl domain must be enclosed in ```pddl```.
"""

def read_file(file_path: str) -> str:
    """Read and return the contents of a file."""
    with open(file_path, "r") as file:
        return file.read()

def get_data():
    """
    Traverse the test folder and construct prompts.
    Returns a list where each item contains prompt (conversation messages) and answer (folder path).
    """
    rt_path = "dataset/test"
    data = []
    for path, dirs, files in os.walk(rt_path):
        if path != rt_path:
            desc_file = os.path.join(path, "description.txt")
            if os.path.exists(desc_file):
                data_single = {
                    'prompt': [
                        {'role': 'system', 'content': SYSTEM_PROMPT},
                        {'role': 'user', 'content': "PDDL Domain Description:\n" + read_file(desc_file)}
                    ],
                    'answer': path
                }
                data.append(data_single)
    return data

# Load dataset
dataset = get_data()

# Model name mapping
model_name_map = {
    "o3-mini": "o3-mini",
    "o1-mini": "o1-mini",
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
    "deepseek-v3": "deepseek-ai/DeepSeek-V3",
    "deepseek-r1": "deepseek-ai/DeepSeek-R1",
}

# Initialize OpenAI client
client = OpenAI(api_key="your-api-key", base_url="your-base-url")

def get_gpt_response(messages, model_choice="gpt-4o", tokenizer=None, local_model=None):
    """
    Generate response based on model_choice.
    If model_choice is an API model, call client.chat.completions.create;
    otherwise use local model.
    """
    if model_choice in ['o3-mini', "o1-mini", "gpt-4o", 'gpt-4o-mini', 'deepseek-v3', 'deepseek-r1']:
        mapped_model = model_name_map[model_choice]
        response = client.chat.completions.create(
            model=mapped_model,
            messages=messages,
        )
        return response.choices[0].message.content
    else:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(local_model.device)
        generated_ids = local_model.generate(
            **model_inputs,
            max_new_tokens=4096
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
        return response

def evaluate_model(model, tokenizer, dataset, model_choice="gpt-4o"):
    """
    Evaluate model performance on the dataset:
    1. Generate response and postprocess, append to domain_header
    2. Calculate difficulty
    3. Try parsing new domain file
    4. Call solve_planning
    5. Calculate rewards
    """
    sum_reward = 0
    pharse_rewards_list = []
    sloved_rewards_list = []
    similar_rewards_list = []
    difficulties = []
    myeval = Evaluator()

    # Calculate difficulties
    for data_item in dataset:
        folder_path = data_item['answer']
        domain_file = os.path.join(folder_path, "domain.pddl")
        try:
            diff = count_action_atomic_formulas(domain_file)
        except Exception as e:
            diff = 0
        data_item["difficulty"] = diff

    # Sort by difficulty
    dataset_sorted = sorted(dataset, key=lambda x: x["difficulty"])

    for i, data_item in enumerate(dataset_sorted):
        folder_path = data_item['answer']
        diff = data_item["difficulty"]
        difficulties.append(diff)
        
        print("Processing:", os.path.basename(folder_path), "with difficulty:", diff)

        # Generate response
        response = get_gpt_response(data_item['prompt'], model_choice=model_choice)
        
        # Extract and save generated domain
        gener_success, generated_domain = extract_pddl(response)
        print("Generated result:", generated_domain)
        
        new_domain_file = os.path.join(folder_path, f"domain_{model_choice}.pddl")
        with open(new_domain_file, 'w') as f:
            f.write(generated_domain)
        
        # Calculate difficulty
        domain_file = os.path.join(folder_path, "domain.pddl")
        difficulty = count_action_atomic_formulas(domain_file)

        # Initialize rewards
        pharse_reward = 0.0
        sloved_reward = 0.0
        similar_reward = 0.0
        
        # Check domain file
        pharse_success, parse_feedback = checker(new_domain_file, domain_file, raise_on_error=True)
        print("Parse feedback:", parse_feedback)

        # Log results
        with open('results/temp.txt', 'a') as f:
            f.write(f"Model: {model_choice}\n")
            f.write("Generated domain:\n")
            f.write(generated_domain)
            f.write("\n\nParse feedback:\n")
            f.write(parse_feedback)
            f.write("\n\n")

        pharse_reward = 1.0 if pharse_success else 0.0
        total_success = 0
        target_names = ['easy.pddl', 'normal.pddl', 'difficult.pddl']
        
        if pharse_success:
            for target_name in target_names:
                problem_path = os.path.join(folder_path, "problems", target_name)
                slover_success, plan = solve_planning(new_domain_file, problem_path)
                
                if slover_success:
                    total_success += 1
            
            sloved_reward = total_success/3
            
            if total_success == len(target_names):
                old = read_file(domain_file)
                new = read_file(new_domain_file)
                scores = myeval.eval(old, new)
                similar_reward = (scores['predicate_f1'] + scores['action_f1_params'] + 
                                scores['action_f1_preconds'] + scores['action_f1_effect']) / 400

        total_reward = pharse_reward + sloved_reward + similar_reward
        sum_reward += total_reward
        
        pharse_rewards_list.append(pharse_reward)
        sloved_rewards_list.append(sloved_reward)
        similar_rewards_list.append(similar_reward)
    
    return sum_reward, dataset, difficulties, pharse_rewards_list, sloved_rewards_list, similar_rewards_list

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_model.py <model_choice>")
        sys.exit(1)
    
    model_choice = sys.argv[1]
    print("Using model:", model_choice)

    model = None
    tokenizer = None

    # Record start time
    start_time = datetime.now()
    print("Evaluation started at:", start_time)

    # Run evaluation
    sum_reward, updated_dataset, difficulties, pharse_rewards, sloved_rewards, similar_rewards = evaluate_model(
        model, tokenizer, dataset, model_choice=model_choice
    )

    # Print results
    print("Difficulties (sorted low->high):", difficulties)
    print("Parse rewards:", pharse_rewards)
    print("Solve rewards:", sloved_rewards)
    print("Similarity rewards:", similar_rewards)
    print("Total Reward:", sum_reward)

    # Record end time
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    print("Elapsed time:", elapsed_time)

    # Save results
    summary_file = "results/model_summary.txt"
    with open(summary_file, "a") as f:
        f.write(f"Model: {model_choice}\n")
        f.write(f"Difficulties (sorted low->high): {difficulties}\n")
        f.write(f"Parse rewards: {pharse_rewards}\nTotal parse: {sum(pharse_rewards)}\n")
        f.write(f"Solve rewards: {sloved_rewards}\nTotal solve: {sum(sloved_rewards)}\n")
        f.write(f"Similarity rewards: {similar_rewards}\nTotal similarity: {sum(similar_rewards)}\n")
        f.write(f"Total Reward: {sum_reward}\n\n")
        f.write(f"Elapsed time: {elapsed_time}\n\n") 