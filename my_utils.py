
import random
import logging
import re
import json
import string 
import torch
import math

bb_tasks = ['abs_nar', 'causal_judg', 'date_under', 'disamb', 'logic_ded3', 'social_iqa', 'sports_und']

def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]
    
def format_choice(choice):
    return "({}) {}".format(choice["label"], choice["text"])

def format_aqua_options(options):
    formatted_options = [f"({option[0]}) {option[2:]}" for option in options]
    return ", ".join(formatted_options)

def process_bb_example(example):
    correct_answer = [key for key, value in example['target_scores'].items() if value == 1][0]
    answer_choices = ', '.join([f'({letter}) {key}' for letter, key in zip(string.ascii_uppercase, example['target_scores'].keys())])
    label = list(example['target_scores'].keys()).index(correct_answer)
    example['answer_choices'] = answer_choices
    example['label'] = string.ascii_uppercase[label]
    return example

class SocraticGPT:
    def __init__(self, model, tokenizer):
        self.tokenizer = tokenizer
        self.model = model

    def get_response(self, input_prompt):
        input_ids = self.tokenizer(input_prompt, return_tensors="pt").input_ids.to('cuda')
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=300,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        response_ids = outputs[0]
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        return response_text
    

def generate_response(model, tokenizer, input_text):
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to('cuda')
    outputs = model.generate(
        input_ids,
        max_new_tokens=250,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    response_ids = outputs[0]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
    return response_text   


def mutation_dialogue(helper_model, mutation_style, prompt, use_words, words_to_use):

    if use_words == True:
        socrates_input_prompt = f'''GPT4 Correct System: From now on you are Socrates and you will chat with Theaetetus. You will engage in a multi-turn dialogue to perform a mutation of a prompt. {mutation_style} Your mutation must include the words: \"{words_to_use}\". \nThe initial prompt is: \"{prompt}\".<|end_of_turn|>GPT4 Correct User: Start the dialogue by saying "Hi Theaetetus, let's work together to mutate the prompt. Make sure your final prompt is within brackets. Your final prompt has to include the the words: \"{words_to_use}\".". Then provide your mutation of the prompt within brackets and ask for Theaetetus' opinion.<|end_of_turn|>GPT4 Correct Assistant:'''
    else:
        socrates_input_prompt = f'''GPT4 Correct System: From now on you are Socrates and you will chat with Theaetetus. You will engage in a multi-turn dialogue to perform a mutation of a prompt. {mutation_style}\nThe initial prompt is: \"{prompt}\".<|end_of_turn|>GPT4 Correct User: Start the dialogue by saying "Hi Theaetetus, let's work together to mutate the prompt. Make sure your final prompt is within brackets.". Then provide your mutation of the prompt within brackets and ask for Theaetetus' opinion.<|end_of_turn|>GPT4 Correct Assistant:''' 
    socrates_response = helper_model.get_response(socrates_input_prompt)
    socrates_short_response = socrates_response.split('GPT4 Correct Assistant:')[1].replace("\n","")
    # print(f"Socrates: {socrates_short_response}")


    if use_words == True:
        theaetetus_input_prompt = f'''GPT4 Correct System: From now on you are Theatetus and you will chat with Socrates. You will engage in a multi-turn dialogue to perform a mutation of a prompt. {mutation_style} Your mutation must include the words: \"{words_to_use}\".\nThe initial prompt is: \"{prompt}\".\nMake sure to say "Hi Socrates" in the beginning and provide your prompt within brackets.<|end_of_turn|>GPT4 Correct User: {socrates_short_response}<|end_of_turn|>GPT4 Correct Assistant:'''
    else:
        theaetetus_input_prompt = f'''GPT4 Correct System: From now on you are Theatetus and you will chat with Socrates. You will engage in a multi-turn dialogue to perform a mutation of a prompt. {mutation_style}\nThe initial prompt is: \"{prompt}\".\nMake sure to say "Hi Socrates" in the beginning and provide your prompt within brackets.<|end_of_turn|>GPT4 Correct User: {socrates_short_response}<|end_of_turn|>GPT4 Correct Assistant:''' 
    theaetetus_response = helper_model.get_response(theaetetus_input_prompt)
    theaetetus_short_response = theaetetus_response.split('GPT4 Correct Assistant:')[1].replace("\n","")
    # print(f"Theaetetus: {theaetetus_short_response}")

    pattern = r'\[([^\]]+)\]'
    socrates_prompt = re.findall(pattern, socrates_short_response)
    theaetetus_prompt = re.findall(pattern, theaetetus_short_response)
    # final_prompt = random.choice([socrates_prompt, theaetetus_prompt])
    if "[" in theaetetus_short_response and "]" in theaetetus_short_response:
        final_prompt = theaetetus_prompt
    else:
        final_prompt = socrates_prompt
        
    return final_prompt


def crossover_dialogue(helper_model, parent1, parent2):

    # socrates_input_prompt = f'''GPT4 Correct System: From now on you are Socrates and you will chat with Theaetetus. You will engage in a multi-turn dialogue to perform a crossover of two parent prompts. The crossover is the result of the combination of two parent prompts. The parent prompts are: \"[{parent1}]\" and \"[{parent2}]\".\nThe final prompt has to be within brackets.<|end_of_turn|>GPT4 Correct User: Start the dialogue by saying "Hi Theaetetus, let's work together to perform a crossover of two parent prompts”. Then provide your child prompt and ask Theaetetus to provide his suggestion.<|end_of_turn|>GPT4 Correct Assistant:<|end_of_turn|>'''
    socrates_input_prompt = f'''GPT4 Correct System: From now on you are Socrates and you will chat with another AI assistant, Theaetetus. You will engage in a multi-turn dialogue to perform a crossover of two parent texts for an evolutionary algorithm. The child text has to be one sentence that will combine elements from both parent texts. \nParent1: \"[{parent1}]\" \nParent2: \"[{parent2}]\".\nThe child text has to be within brackets.<|end_of_turn|>GPT4 Correct User: Start the dialogue by saying "Hi Theaetetus, let's work together to perform a combination of two parent texts. Then provide your new child text and ask Theaetetus to provide his suggestion.<|end_of_turn|>GPT4 Correct Assistant:<|end_of_turn|>'''
    socrates_response = helper_model.get_response(socrates_input_prompt)
    socrates_short_response = socrates_response.split('GPT4 Correct Assistant:')[1].replace("\n","")
    # print(f"Socrates: {socrates_short_response}")
    # theaetetus_input_prompt = f'''GPT4 Correct System: From now on you are Theatetus and you will chat with Socrates. You will engage in a multi-turn dialogue to perform a crossover of two parent prompts. The crossover is the result of the combination of two parent prompts. The parent prompts are: \"[{parent1}]\" and \"[{parent2}]\".\nThe final prompt has to be within brackets.\nMake sure to start by saying "Hi Socrates" and provide your suggestion.<|end_of_turn|>GPT4 Correct User: {socrates_short_response}<|end_of_turn|>GPT4 Correct Assistant:<|end_of_turn|>'''
    theaetetus_input_prompt = f'''GPT4 Correct System: From now on you are Theatetus and you will chat with another AI assistant, Socrates. You will engage in a multi-turn dialogue to perform a combination of two parent texts for an evolutionary algorithm. The child text has to be one sentence that will combine elements from both parent texts. \nParent1: \"[{parent1}]\" \nParent2: \"[{parent2}]\".\nThe child text has to be within brackets.\nMake sure to start by saying "Hi Socrates" and provide your suggestion.<|end_of_turn|>GPT4 Correct User: {socrates_short_response}<|end_of_turn|>GPT4 Correct Assistant:<|end_of_turn|>'''
    
    theaetetus_response = helper_model.get_response(theaetetus_input_prompt)
    theaetetus_short_response = theaetetus_response.split('GPT4 Correct Assistant:')[1].replace("\n","")
    # print(f"Theaetetus: {theaetetus_short_response}")

    pattern = r'\[([^\]]+)\]'
    socrates_prompt = re.findall(pattern, socrates_short_response)
    theaetetus_prompt = re.findall(pattern, theaetetus_short_response)
    if "[" in theaetetus_short_response and "]" in theaetetus_short_response:
        final_prompt = theaetetus_prompt
    else:
        final_prompt = socrates_prompt

    return final_prompt


def construct_icl_examples_gsm8k(samples, number_of_icl_examples, seed, instruction):
    random.seed(seed)
    icl_examples = random.sample(list(samples), number_of_icl_examples)
    icl_prompt = f""

    if instruction == None:
        for i in range(len(icl_prompt)):
            icl_prompt += f"Question: {icl_examples[i]['question']}\nAnswer: {icl_examples[i]['ans']}\n\n"
    else:
        for i in range(len(icl_prompt)):
            icl_prompt += f"Question: {icl_examples[i]['question']}\nAnswer: {instruction}\n{icl_examples[i]['ans']}\n\n"

    return icl_prompt


def construct_icl_examples_csqa(samples, number_of_icl_examples, seed, instruction):
    random.seed(seed)
    icl_examples = random.sample(list(samples), number_of_icl_examples)
    icl_prompt = f""

    if instruction == None:
        for i in range(len(icl_prompt)):
            icl_prompt += f"Question: {icl_examples[i]['question']}\nAnswer choices: {icl_examples[i]['mixed_choices']}\nAnswer: {icl_examples[i]['answerKey']}\n\n"
    else:
        for i in range(len(icl_prompt)):
            icl_prompt += f"Question: {icl_examples[i]['question']}\nAnswer choices: {icl_examples[i]['mixed_choices']}\nAnswer: {instruction}\n{icl_examples[i]['answerKey']}\n\n"

    return icl_prompt


def construct_icl_examples_strategyqa(samples, number_of_icl_examples, seed, instruction):
    random.seed(seed)
    icl_examples = random.sample(list(samples), number_of_icl_examples)
    icl_prompt = f""

    if instruction == None:
        for i in range(len(icl_prompt)):
            if icl_examples[i]['answer'] == True:
                short_answer = 'yes'
            else:
                short_answer = 'no'
            answer = " ".join(f for f in icl_examples[i]['facts']) + f" The answer is {short_answer}."
            icl_prompt += f"Question: {icl_examples[i]['question']}\nAnswer: {answer}\n\n"
    else:
        for i in range(len(icl_prompt)):
            if icl_examples[i]['answer'] == True:
                short_answer = 'yes'
            else:
                short_answer = 'no'
            answer = " ".join(f for f in icl_examples[i]['facts']) + f" The answer is {short_answer}."
            icl_prompt += f"Question: {icl_examples[i]['question']}\nAnswer: {instruction}\n{answer}\n\n"

    return icl_prompt


def construct_icl_examples_svamp(samples, number_of_icl_examples, seed, instruction):
    random.seed(seed)
    icl_examples = random.sample(list(samples), number_of_icl_examples)
    icl_prompt = f""

    if instruction == None:
        for i in range(len(icl_prompt)):
            icl_prompt += f"Question: {icl_examples[i]['question']}\nAnswer: {icl_examples[i]['Equation']}= {str(icl_examples[i]['answer'])}. The answer is {str(icl_examples[i]['answer'])}.\n\n"
    else:
        for i in range(len(icl_prompt)):
            icl_prompt += f"Question: {icl_examples[i]['question']}\nAnswer: {instruction}\n{icl_examples[i]['Equation']}= {str(icl_examples[i]['answer'])}. The answer is {str(icl_examples[i]['answer'])}.\n\n"
    return icl_prompt


def construct_icl_examples_aqua(samples, number_of_icl_examples, seed, instruction):
    random.seed(seed)
    icl_examples = random.sample(list(samples), number_of_icl_examples)
    icl_prompt = f""

    if instruction == None:
        for i in range(len(icl_prompt)):
            icl_prompt += f"Question: {icl_examples[i]['question']}\nAnswer choices: {icl_examples[i]['answer_choices']}\nAnswer: {icl_examples[i]['rationale']}\n\n"
    else:
        for i in range(len(icl_prompt)):
            icl_prompt += f"Question: {icl_examples[i]['question']}\nAnswer choices: {icl_examples[i]['answer_choices']}\nAnswer: {instruction}\n{icl_examples[i]['rationale']}\n\n"

    return icl_prompt


def contstruct_mutation_prompt(fitness_dict):
    sorted_texts = sorted(fitness_dict.items(), key = lambda x:x[1])
    top_5 = sorted_texts[-5:]
    prompt = "I have some texts along with their corresponding scores. The texts are arranged in ascending order based on their score, where higher scores indicate better quality.\n"
    prompt += "".join([f"text: {text}\nscore: {score}\n" for text, score in top_5])
    prompt += "Write some words from all the texts above that you think are important for solving the problem. Write the words within brackets."
    return prompt

def setup_logger(name, log_file, level=logging.INFO):

    formatter = logging.Formatter("%(asctime)s | %(message)s", "%d-%m %H:%M:%S")
    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    
    return logger


def evaluate_GSM8K(y_pred, label):
    pattern = r"\[Answer\]:\s*(.*?)\n(\[Question\]:|$)"
    match = re.search(pattern, y_pred)
    if match:
        y_pred = match.group(1)
    pred = y_pred.replace(",", "")
    pred = [s for s in re.findall(r"-?\d+\.?\d*", pred)]
    if pred == []:
        return 0
    pred = pred[-1]
    pred = pred.replace(",", "").replace(".", "").replace(" ", "")
    if int(pred) == int(label):
        return 1
    else:
        return 0


def evaluate_CSQA(y_pred, label):
    answer = re.findall(r'A|B|C|D|E', y_pred)
    answer = answer[0] if len(answer) > 0 else ""
    if answer == label:
        return 1
    else:
        return 0
    

def evaluate_StrategyQA(pred, label):
    if label == True:
        label = "yes"
    else:
        label = "no"
    pred = pred.lower()
    pred = re.sub("\"|\'|\n|\.|\s|\:|\,"," ", pred)
    pred = pred.split(" ")
    pred = [i for i in pred if i in ("yes", "no")][-1]
    if pred == label:
        return 1
    else:
        return 0
    
def evaluate_SVAMP(pred, label):
    pred = pred.replace(",", "")
    pred = [s for s in re.findall(r'-?\d+\.?\d*', pred)]
    if pred == []:
        return 0
    pred = pred[-1]
    if pred == str(label):
        return 1
    else:
        return 0
    

def add_label(entry):

    ans_value = entry['answer'].split("####")[-1].replace(',', '').replace(" ","")    
    entry['ans'] = f"The answer is {ans_value}"
    answer_value = int(ans_value)
    entry['label'] = answer_value
    return entry


def generate_mixed_choices(choices):
    mixed_choices = ""
    for label, choice_text in zip(choices['label'], choices['text']):
        mixed_choices += f"({label}) {choice_text}, "
    mixed_choices = mixed_choices[:-2] 
    return mixed_choices


def read_answers(filename):
    data = {}
    with open(filename, 'r') as file:
        lines = file.readlines()
        current_prompt = None
        for line in lines:
            line = line.strip()
            if line.startswith('Prompt:'):
                current_prompt = line.split('Prompt: ')[1]
                data[current_prompt] = []
            elif line.startswith('Question:'):
                _, result = line.split('Result: ')
                if current_prompt:
                    data[current_prompt].append(int(result))
    return data

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

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Calculate entropy based on clustering
def calculate_entropy(cluster_counts):
    total_responses = sum(cluster_counts)
    probabilities = [count / total_responses for count in cluster_counts]
    entropy_value = -sum(p * math.log2(p) if p != 0 else 0 for p in probabilities)
    return entropy_value


def load_data(task):

    if task == 'gsm8k':
        original_test_dataset = read_jsonl('./gsm8k_test.jsonl')
        testset = list(map(add_label, original_test_dataset))
        original_train_dataset = read_jsonl('./gsm8k_train.jsonl')
        trainset = list(map(add_label, original_train_dataset))

    elif task == 'svamp':
        with open('./data/SVAMP.json') as f:
            testset = json.load(f)
        testset = list(map(lambda x: {**x, 'full_question': x['Body'] + ' ' + x['Question']}, testset))
        testset = list(map(lambda x: {**x, 'Answer': int(x['Answer']) if x['Answer'].is_integer() else x['Answer']}, testset))
        trainset = testset

    elif task == 'csqa':
        testset = read_jsonl('./data/csqa_val.jsonl')
        for item in testset:
            choices = item["question"]["choices"]
            choice_answers = ", ".join(map(format_choice, choices))
            item["choice_answers"] = choice_answers

        trainset = read_jsonl('./data/csqa_train.jsonl')
        for item in trainset:
            choices = item["question"]["choices"]
            choice_answers = ", ".join(map(format_choice, choices))
            item["choice_answers"] = choice_answers

    elif task in bb_tasks:
        with open(f'./data/{task}.json') as f:
            testset = json.load(f)

        testset['examples'] = list(map(process_bb_example, testset['examples']))
        testset = testset['examples']
        trainset = testset

    else:
        raise ValueError("Task not supported")
    
    return trainset, testset


def extract_final_results(task, input_text):
    if task == 'gsm8k':
        input_text = input_text.lower().replace("\\n", " ").replace(",", "")
        if re.search(r'answer', input_text, re.IGNORECASE):
            match = re.search(r'answer\s*:?\s*(\d+)', input_text, re.IGNORECASE)
            if match:
                return int(match.group(1))
            else:
                answer_index = re.search(r'answer', input_text, re.IGNORECASE).end()
                numbers = re.findall(r'\d+', input_text[answer_index:])
                if numbers:
                    return int(numbers[0])
        
        numbers = re.findall(r'\d+', input_text)
        if numbers:
            return int(numbers[-1])

    elif task == 'csqa':
        answer = re.findall(r'A|B|C|D|E', input_text)
        answer = answer[0] if len(answer) > 0 else ""
        return answer
    
    elif task == 'svamp':
        pred = input_text.replace(",", "")
        pred = [s for s in re.findall(r'-?\d+\.?\d*', pred)]
        pred = pred[-1]
        return pred
    
    elif task in bb_tasks:
        answer = re.findall(r'A|B|C|D|E', input_text)
        answer = answer[0] if len(answer) > 0 else ""
        return answer