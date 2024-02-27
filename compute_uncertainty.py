from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.nn.functional import cosine_similarity
from itertools import combinations
import math

def calculate_entropy(clusters):
    total_responses = sum(len(cluster) for cluster in clusters)
    probabilities = [len(cluster) / total_responses for cluster in clusters]
    entropy_value = -sum(p * math.log2(p) if p != 0 else 0 for p in probabilities)
    return entropy_value


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
    prompt = "Determine the number of semantic sets in the given responses. The number of semantic sets shows in how many different clusters we could assign these responses to. You need to check whether the responses within the same cluster exactly describe the same thing such as the same entity, digit, or arithmetical results.\n"
    for i, response in enumerate(responses):
        prompt += f"Response {i + 1}: {response}\n"
    prompt += "Output format: Cluster 1: ['Response 1', 'Response 2', 'Response 3']\n Cluster 2: ['Response 4', 'Response 5', 'Response 6']\n ... \n Cluster N: ['Response M', 'Response N']"
    return prompt


def get_response(model, tokenizer, input_prompt):
    input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids.to('cuda')
    outputs = model.generate(
        input_ids,
        max_new_tokens=200,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    response_ids = outputs[0]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
    return response_text


prompts = [
    "Let's think step by step",
    "Let's first understand the problem and devise a plan to solve the problem. Then, let's carry out the plan and solve the problem step by step",
    "Let's first understand the problem, extract relevant variables and their corresponding numerals, and devise a plan. Then, let's carry out the plan, calculate the intermediate results (pay attention to calculation and common sense), solve the problem step by step, and show the answer",
    "Let's work this out in a step by step way to be sure we have the right answer",
    "Take a deep breath and work on this problem step-by-step",
    "Break this down",
    "A little bit of arithmetic and a logical approach will help us quickly arrive at the solution to this problem",
    "Embark on a journey to derive the solution to this problem",
    "Compute the solution with a calculated, stepwise approach",
    "Let's be very precise and accurate in our calculations",
    "Our approach will be to methodically work through the problem, ensuring accuracy at each step to derive the correct answer",
    "Slow down, let's break this down into manageable steps"
] 

decode_strategies = ["greedy", "contrastive_search", "multinomial_sampling", "beam_search", "beam_search_with_multinomial_sampling", "top_k_sampling", "top_p_sampling", "sampling0.25", "sampling0.5", "sampling0.75"]

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] 
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def extract_embeddings(model, tokenizer, responses):
    encoded_input = tokenizer(responses, padding=True, truncation=True, return_tensors='pt').to('cuda')
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    combinations = list(combinations(range(len(responses)), 2))
    for i, j in combinations:
        similarity = cosine_similarity(sentence_embeddings[i].unsqueeze(0), sentence_embeddings[j].unsqueeze(0))
        print(f"Similarity between response {i} and response {j} is {similarity.item()}")


tokenizer = AutoTokenizer.from_pretrained("berkeley-nest/Starling-LM-7B-alpha")
model = AutoModelForCausalLM.from_pretrained("berkeley-nest/Starling-LM-7B-alpha", torch_dtype = torch.float16)
model = model.to('cuda')
filename = './decode_results.txt'
answer_dict = extract_responses(filename, 'starling')


for prompt in prompts:
    for sample in range(len(testset)):
        responses = []
        for decode_strategy in decode_strategies:
            responses.append(answer_dict[prompt][decode_strategy][f'{sample}']) 
            prompt = construct_prompt(responses)
            model_input = f'''GPT4 Correct User: {prompt}<|end_of_turn|>GPT4 Correct Assistant:'''
            response = get_response(model, tokenizer, model_input)
            response_text = response.split("GPT4 Correct Assistant:")[1]


    



