
import random
import logging
import re
import json
import string 

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
