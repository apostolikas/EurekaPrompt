from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import random
import argparse
import logging
import torch
import re
import json
import time 

def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]


def construct_icl_examples_gsm8k(samples, number_of_icl_examples, seed, instruction):
    random.seed(seed)
    icl_examples = random.sample(list(samples), number_of_icl_examples)
    icl_prompt = f""

    if instruction == None:
        for i in range(len(icl_prompt)):
            icl_prompt += f"Q: {icl_examples[i]['question']}\A: {icl_examples[i]['ans']}\n\n"
    else:
        for i in range(len(icl_prompt)):
            icl_prompt += f"Q: {icl_examples[i]['question']}\A: {instruction}\n{icl_examples[i]['ans']}\n\n"

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



def setup_logger(name, log_file, level=logging.INFO):

    formatter = logging.Formatter("%(asctime)s | %(message)s", "-%d-%m %H:%M:%S")
    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    
    return logger


def evaluate_GSM8K(y_pred, label):
    pattern = r"\[A\]:\s*(.*?)\n(\[Q\]:|$)"
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


# gsm8k_initial_prompts = [
#     "Let's think step by step",
#     "Let's first understand the problem and devise a plan to solve the problem. Then, let's carry out the plan and solve the problem step by step",
#     "Let's first understand the problem, extract relevant variables and their corresponding numerals, and devise a plan. Then, let's carry out the plan, calculate the intermediate results (pay attention to calculation and common sense), solve the problem step by step, and show the answer",
#     "Let's work this out in a step by step way to be sure we have the right answer",
#     "Take a deep breath and work on this problem step-by-step",
#     "Break this down",
#     "A little bit of arithmetic and a logical approach will help us quickly arrive at the solution to this problem",
#     "Let's combine our numerical command and clear thinking to quickly and accurately decipher the answer",
#     "Divide the topic into smaller, digestible sections and examine it thoroughly, while gradually reducing the speed of discussion",
#     "Let's submerge ourselves in the conundrum, identify vital variables and their numerical values, and establish a plan. As we carry out the plan, let's scrutinize intermediate findings (ensure correct numerical calculations and logical reasoning), tackle the problem progressively, and unveil the answer",
#     "Let's understand the problem, devise a step-by-step plan, and ensure precise and accurate calculations to find the right answer"

# ]

gsm8k_initial_prompts = [
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
    "Let's submerge ourselves in the conundrum, identify vital variables and their numerical values, and establish a plan. As we carry out the plan, let's scrutinize intermediate findings (ensure correct numerical calculations and logical reasoning), tackle the problem progressively, and unveil the answer",
    "Focus on strategic thinking to swiftly find the accurate solution"
    "Divide the topic into smaller, digestible sections and examine it thoroughly, while gradually reducing the speed of discussion",

]

csqa_initial_prompts = [
    "Let's think step by step",
    "Let's devise a plan and solve the problem step by step"
    "Let's first understand the problem, extract relevant variables and their corresponding numerals, and devise a plan. Then, let's carry out the plan, calculate the intermediate results (pay attention to calculation and common sense), solve the problem step by step, and show the answer",
]



class InferenceEvalauator:
    def __init__(self, args, testset, model, tokenizer, trainset, population):
        self.args = args
        self.testset = testset
        self.trainset = trainset
        self.model = model
        self.tokenizer = tokenizer
        self.initial_population = population

    def evaluate_prompt(self, prompt):
        accuracy = 0
        num_of_samples = len(self.testset)

        if self.args.task == 'gsm8k':

            random.seed(self.args.seed)
            samples = random.sample(self.testset, num_of_samples)

            for sample in tqdm(samples):
                question = sample['question']
                label = sample['label']

                if self.args.use_icl_examples:
                    # include contrastive cot examples
                    icl_prompt = construct_icl_examples_gsm8k(self.trainset, self.initial_population, self.args.num_icl_examples, prompt)
                    model_input = f'''{icl_prompt}Q: {question}\A: {prompt}'''
                else:
                    model_input = f'''Q: {question}\nA: {prompt}'''
                
                if self.args.model == 'starling' or self.args.model == 'openchat':
                    input_prompt = f'''GPT4 Correct User: {model_input}<|end_of_turn|>GPT4 Correct Assistant:'''
                    input_ids = self.tokenizer(input_prompt, return_tensors="pt").input_ids.to('cuda')
                    outputs = self.model.generate(input_ids, max_new_tokens=250, pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id)
                    response_ids = outputs[0]
                    text_output = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                    text_output = text_output.split("GPT4 Correct Assistant:")[1]

                elif self.args.model == 'openhermes':
                    # Use role-play prompting 
                    messages = [
                        {"role": "system", "content": "From now on, you are an excellent math teacher and always teach your students math problems correctly. And I am one of your students."},
                        {"role": "user", "content": model_input},
                    ]
                    gen_input = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to('cuda')
                    generated_ids = self.model.generate(gen_input, max_new_tokens = 250)
                    decoded = self.tokenizer.batch_decode(generated_ids)
                    text_output = decoded[0]

                    text_output = text_output.split("A:<|im_end|>")[1]

                accuracy += evaluate_GSM8K(text_output, label)

        elif self.args.task == 'csqa':
            
            random.seed(self.args.seed)
            samples = random.sample(self.testset, num_of_samples)

            for sample in tqdm(samples):
                question = sample['question']
                choices = sample['mixed_choices']
                label = sample['label']

                if self.args.use_icl_examples:
                    # include contrastive cot examples
                    icl_prompt = construct_icl_examples_csqa(self.trainset, self.initial_population, self.args.num_icl_examples, prompt)
                    model_input = f'''{icl_prompt}Question: {question}\nAnswer: {prompt}'''
                else:
                    model_input = f'''Question: {question}\nAnswer Choices: {choices}\nAnswer: {prompt}'''
                input_prompt = f'''GPT4 Correct User: {model_input}<|end_of_turn|>GPT4 Correct Assistant:'''
                text_output = self.model.get_response(input_prompt)
                accuracy += evaluate_CSQA(text_output, choices, label)

        accuracy = accuracy/num_of_samples

        return accuracy


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Settings for the Inference')
    parser.add_argument('--task', default='gsm8k', type=str, help='Task to be solved. Choose one of: [gsm8k, csqa]')
    parser.add_argument('--use_icl_examples', default=False, type=bool, help='whether to use in-context learning examples or not')
    parser.add_argument('--num_icl_examples', default=1, type=int, help='number of in-context learning examples used for evaluation')
    parser.add_argument('--model', default='openchat', type=str, help='which model to use')
    parser.add_argument('--seed', default=0, type=int, help='type of mutation')
    args = parser.parse_args()
    
    logger_name = f"Inference_Eval_{args.task}_output.log"
    logger = setup_logger('progress_logger', logger_name)

    if args.model == 'starling':

        tokenizer = AutoTokenizer.from_pretrained("berkeley-nest/Starling-LM-7B-alpha")
        model = AutoModelForCausalLM.from_pretrained("berkeley-nest/Starling-LM-7B-alpha", torch_dtype = torch.float16)

    elif args.model == 'openhermes':

        tokenizer = AutoTokenizer.from_pretrained("teknium/OpenHermes-2.5-Mistral-7B")
        model = AutoModelForCausalLM.from_pretrained("teknium/OpenHermes-2.5-Mistral-7B", torch_dtype = torch.float16)

    elif args.model == 'openchat':

        tokenizer = AutoTokenizer.from_pretrained("openchat/openchat_3.5")
        model = AutoModelForCausalLM.from_pretrained("openchat/openchat_3.5", torch_dtype = torch.float16)

    model = model.to('cuda')
    
    if args.task == 'gsm8k':
        original_test_dataset = read_jsonl('./data/gsm8k_test.jsonl')
        testset = list(map(add_label, original_test_dataset))
        original_train_dataset = read_jsonl('./data/gsm8k_train.jsonl')
        trainset = list(map(add_label, original_train_dataset))
    
    elif args.task == 'csqa':
        original_test_dataset = load_dataset("commonsense_qa", split='validation')
        testset = list(map(lambda instance: {**instance, 'mixed_choices': generate_mixed_choices(instance['choices'])}, original_test_dataset))
        original_train_dataset = load_dataset("commonsense_qa", split='train')
        trainset = list(map(lambda instance: {**instance, 'mixed_choices': generate_mixed_choices(instance['choices'])}, original_train_dataset))
    else:
        raise ValueError("Task not supported")

    if args.task == 'gsm8k':
        initial_population = gsm8k_initial_prompts
    elif args.task == 'csqa':
        initial_population = csqa_initial_prompts

    inference_engine = InferenceEvalauator(args, testset, model, tokenizer, trainset, initial_population)

    logger.info(f"Arguments: {args}")
    logger.info(f"Evaluation of the prompt population")

    fitness_dict = {}

    for prompt in initial_population:
        start = time.time()
        fitness_dict[prompt] = inference_engine.evaluate_prompt(prompt)
        end = time.time()
        print(f"Evaluated prompt: {prompt} | Accuracy: {fitness_dict[prompt]} | Time: {round((end-start),4)} seconds")
        logger.info(f"Prompt: {prompt} with accuracy {fitness_dict[prompt]}")