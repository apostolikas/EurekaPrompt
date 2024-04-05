from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from torch.nn.functional import cosine_similarity
from itertools import combinations
import math
from my_utils import *
import re

def calculate_entropy(clusters):
    total_responses = sum(len(cluster) for cluster in clusters)
    probabilities = [len(cluster) / total_responses for cluster in clusters]
    entropy_value = -sum(p * math.log2(p) if p != 0 else 0 for p in probabilities)
    return entropy_value

def extract_clusters(input_string):
    pattern = r'Cluster \d+: \[(.*?)\]'
    matches = re.findall(pattern, input_string)
    clusters = [match.strip().split(', ') for match in matches]
    return clusters

python_string = '''Cluster 1: ['Response 1', 'Response 2', 'Response 4', 'Response 5', 'Response 8', 'Response 9']
Cluster 2: ['Response 3', 'Response 10']
Cluster 3: ['Response 6', 'Response 7']'''

clusters = extract_clusters(python_string)
print(clusters)


def extract_responses(filename, model_name):

    with open(filename, 'r') as file:
        file_content = file.read()

    lines = file_content.split('\n')
    my_dictionary = {}
    if model_name == 'starling' or model_name == 'openchat':
    
        for line in lines:
            if line == '':
                continue
            sections = line.split('|')
            for i in range(0, len(sections), 4):

                if len(sections) > 4:
                    prompt = sections[0].split(':')[1].strip()
                    decode_strategy = sections[1].split(':')[1].strip()
                    sample = sections[2].split('Sample:')[1].strip()
                    output = ''.join(map(str, sections[2 + 1:]))
                    output = output.split('GPT4 Correct Assistant:')[1].replace('"]', '')
                else:
                    prompt = sections[i].split(':')[1].strip()
                    decode_strategy = sections[i + 1].split(':')[1].strip()
                    sample = sections[i + 2].split('Sample:')[1].strip()
                    output = sections[i + 3].split('GPT4 Correct Assistant:')[1].replace('"]', '')

                if prompt not in my_dictionary:
                    my_dictionary[prompt] = {}
                if decode_strategy not in my_dictionary[prompt]:
                    my_dictionary[prompt][decode_strategy] = {}
                
                my_dictionary[prompt][decode_strategy][sample] = output
        return my_dictionary
    else:
        raise NotImplementedError("Model not supported")

def construct_prompt(responses):
    prompt = "<s> [INST] Determine the number of semantic sets in the given responses. The number of semantic sets shows in how many different clusters we could assign these responses to. You need to check whether the responses within the same cluster exactly describe the same thing such as the same entity, digit, or arithmetical results. Different wording should not affect the clustering as long as the semantics do not change. Give more focus to the entity, the digits, the reasoning steps and the final answer.\n"
    for i, response in enumerate(responses):
        prompt += f"Response {i + 1}: {response}\n"
    prompt += "Output format: Cluster 1: ['Response 1', 'Response 2', 'Response 3']\n Cluster 2: ['Response 4', 'Response 5', 'Response 6']\n ... \n Cluster N: ['Response M', 'Response N'] [/INST]"
    return prompt

prompts = [
    "Let's think step by step",
    "Let's first understand the problem and devise a plan to solve the problem. Then, let's carry out the plan and solve the problem step by step",
    "Let's first understand the problem, extract relevant variables and their corresponding numerals, and devise a plan. Then, let's carry out the plan, calculate the intermediate results (pay attention to calculation and common sense), solve the problem step by step, and show the answer",
    "Let's work this out in a step by step way to be sure we have the right answer",
    "Take a deep breath and work on this problem step-by-step",
    "Break this down",
    "A little bit of arithmetic and a logical approach will help us quickly arrive at the solution to this problem",
    # "Embark on a journey to derive the solution to this problem",
    # "Compute the solution with a calculated, stepwise approach",
    # "Let's be very precise and accurate in our calculations",
    # "Our approach will be to methodically work through the problem, ensuring accuracy at each step to derive the correct answer",
    # "Slow down, let's break this down into manageable steps",
    "Focus on strategic thinking to swiftly find the accurate solution"
] 

decode_strategies = ["greedy", "contrastive_search", "multinomial_sampling", "beam_search", "beam_search_with_multinomial_sampling", "top_k_sampling", "top_p_sampling", "sampling0.25", "sampling0.5", "sampling0.75"]

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1", quantization_config=bnb_config, device_map="auto")
filename = './decode_results.txt'
answer_dict = extract_responses(filename, 'starling')
task = 'GSM8K'
entropy_dict = {}
file_name = f"./uncertainties/entropy_{task}.txt"


with open(file_name, 'w') as file:
    for prompt in prompts:
        entropy_list = []
        for sample in range(1):
            responses = []
            for decode_strategy in decode_strategies:
                responses.append(answer_dict[prompt][decode_strategy][f'{sample}']) 
            clustering_prompt = construct_prompt(responses)
            inputs = tokenizer(clustering_prompt, return_tensors="pt").to('cuda')
            outputs = model.generate(**inputs, max_new_tokens=300, pad_token_id = tokenizer.eos_token_id)
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            output_text = output_text.split("[/INST]")[1]
            clusters = extract_clusters(output_text)
            entropy = calculate_entropy(clusters)
            entropy_list.append((sample, entropy))
            print(f"Prompt: {prompt}, Sample: {sample}, Entropy: {entropy}")
            file.write(f"Prompt: {prompt}, Sample: {sample}, Entropy: {entropy}\n")
        entropy_dict[prompt] = entropy_list
        average_entropy = sum(entropy for _, entropy in entropy_list) / len(entropy_list)
        file.write(f"Average entropy for prompt {prompt}: {average_entropy}\n")
        print(f"Average entropy for prompt {prompt}: {average_entropy}")


