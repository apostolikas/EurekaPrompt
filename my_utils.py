
import random
import logging
import re
import json

def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]
    
def format_choice(choice):
    return "({}) {}".format(choice["label"], choice["text"])

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

def contstruct_mutation_prompt(fitness_dict):
    sorted_texts = sorted(fitness_dict.items(), key = lambda x:x[1])
    top_5 = sorted_texts[-5:]
    prompt = "I have some texts along with their corresponding scores. The texts are arranged in ascending order based on their score, where higher scores indicate better quality.\n"
    prompt += "".join([f"text: {text}\nscore: {score}\n" for text, score in top_5])
    prompt += "Write some parts from all the texts above that you think are important for solving the problem. Write the parts within brackets."
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

def evaluate_CSQA(y_pred, choices, label):
    label_text = None
    for choice in choices.split(','):
        choice = choice.strip()
        if choice.startswith("(" + label + ")"):
            label_text = choice.split(')')[1].strip()
            break

    if label_text is None:
        return "Invalid label provided"

    label_text_words = set(label_text.split())
    y_pred_words = set(y_pred.split())

    if label_text_words.intersection(y_pred_words) or label in y_pred:
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


gsm8k_initial_prompts = [
    "Let's think step by step",
    "Let's first understand the problem and devise a plan to solve the problem. Then, let's carry out the plan and solve the problem step by step",
    "Let's first understand the problem, extract relevant variables and their corresponding numerals, and devise a plan. Then, let's carry out the plan, calculate the intermediate results (pay attention to calculation and common sense), solve the problem step by step, and show the answer",
    "Let's work this out in a step by step way to be sure we have the right answer",
    "Take a deep breath and work on this problem step-by-step",
    "Break this down",
    "A little bit of arithmetic and a logical approach will help us quickly arrive at the solution to this problem",
    "Let's combine our numerical command and clear thinking to quickly and accurately decipher the answer",
    "Let's be very precise and accurate in our calculations",
    "Let's create a simplified version of the problem to gain insights and test potential solutions",
    "Embark on a journey to derive the solution to this problem",
    "Compute the solution with a calculated, stepwise approach",
    "Let's be very precise and accurate in our calculations",
    "Our approach will be to methodically work through the problem, ensuring accuracy at each step to derive the correct answer",
    "Slow down, let's break this down into manageable steps",
    "Inhale deeply, exhale slowly, and embark on this problem-solving journey with a step-by-step mindset",

]

csqa_initial_prompts = [
    "Let's think step by step",
    "Let's devise a plan and solve the problem step by step"
    "Let's first understand the problem, extract relevant variables and their corresponding numerals, and devise a plan. Then, let's carry out the plan, calculate the intermediate results (pay attention to calculation and common sense), solve the problem step by step, and show the answer",
]

# csqa_mutation_styles = [
#     "The mutation is a variant of the input prompt that introduces a structured thought process.",
#     "The mutation is a variant of the input prompt that highlights strategic thought processes and logical reasoning.",
#     "The mutation is a variant of the input prompt that introduces logic and makes it easier to understand.",
#     "The mutation is a variant of the input prompt that focuses on logic and reasoning.",
#     "The mutation is a variant of the input prompt that adds more details.",
#     "The mutation is a variant of the input prompt that makes it more well-considered and logical.",
#     ]

# gsm8k_mutation_styles = [
#     "The mutation is a variant of the input prompt that introduces a structured thought process.",
#     "The mutation is a variant of the input prompt that highlights strategic thought processes and mathematical reasoning.",
#     "The mutation is a variant of the input prompt that introduces mathematical reasoning and makes it easier to understand.",
#     "The mutation is a variant of the input prompt that focuses on mathematical reasoning and logical thinking.",
#     "The mutation is a variant of the input prompt that adds more details.",
#     "The mutation is a variant of the input prompt that makes it more well-considered and logical.",
#     ]


csqa_mutation_styles = [
    "The mutation is a variant of the input prompt that introduces a structured thought process.",
    "The mutation is a variant of the input prompt using unconventional thinking.",
    "The mutation is a variant of the input prompt that provides an alternative viewpoint.",
    "The mutation presents a tweaked version of the task, emphasizing logical steps in problem-solving.",
    "This variant of the prompt, through mutation, offers a fresh perspective on the problem, focusing on strategic thinking.",
    "Through mutation, the prompt is altered to showcase problem-solving strategies and logical reasoning.",
    "The mutation introduces a revised version of the prompt, aiming to illuminate the process of logical reasoning and problem-solving.",
    ]

gsm8k_mutation_styles = [
    "The mutation is a variant of the input prompt that introduces a structured thought process.",
    # "The mutation is a variant of the input prompt that highlights strategic thought processes and mathematical reasoning.",
    # "The mutation is a variant of the input prompt that introduces mathematical reasoning and makes it easier to understand.",
    "The mutation is a variant of the input prompt using unconventional thinking.",
    "The mutation is a variant of the input prompt that provides an alternative viewpoint.",
    # "The mutation is a variant of the input prompt that makes it more well-considered and logical.",
    "The mutation presents a tweaked version of the task, emphasizing logical steps in problem-solving.",
    "This variant of the prompt, through mutation, offers a fresh perspective on the problem, focusing on strategic thinking.",
    "Through mutation, the prompt is altered to showcase problem-solving strategies and logical reasoning.",
    "The mutation introduces a revised version of the prompt, aiming to illuminate the process of logical reasoning and problem-solving.",
    ]