from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import argparse
import torch
import re
import json
import time 
import logging 

def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]

def setup_logger(name, log_file, level=logging.INFO):

    formatter = logging.Formatter("%(asctime)s | %(message)s", "%d-%m %H:%M:%S")
    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    
    return logger

def add_label(entry):

    ans_value = entry['answer'].split("####")[-1].replace(',', '').replace(" ","")    
    entry['ans'] = f"The answer is {ans_value}"
    answer_value = int(ans_value)
    entry['label'] = answer_value
    return entry


class InferenceEvalauator:
    def __init__(self, args, logger, testset, model, tokenizer, trainset, population):
        self.args = args
        self.logger = logger
        self.testset = testset
        self.trainset = trainset
        self.model = model
        self.tokenizer = tokenizer
        self.initial_population = population

    def evaluate_prompt(self, prompt):

        if self.args.task == 'gsm8k':

            random.seed(self.args.seed)
            samples = random.sample(self.testset, self.args.num_of_samples)

            for i,sample in enumerate(samples):
                question = sample['question']
                label = sample['label']
                model_input = f'''Q: {question}\nA: {prompt}'''
                start = time.time()

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

                end = time.time()
                self.logger.info(f"Sample: {i} | Prompt: {prompt} | Question: {question} | Label: {label} | Model Output: {text_output} | Time: {round((end-start),4)}")
            
        else:
            raise ValueError("Task not supported yet")



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Settings for the Inference')
    parser.add_argument('--task', default='gsm8k', type=str, help='Task to be solved. Choose one of: [gsm8k, csqa]')
    parser.add_argument('--model', default='starling', type=str, help='which model to use')
    parser.add_argument('--seed', default=0, type=int, help='type of mutation')
    parser.add_argument('--num_of_samples', default = 1, type = int, help = 'how many samples to take from testset')
    args = parser.parse_args()
    
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

    logger_name = f"Compare_outputs_{args.task}.log"
    logger = setup_logger('progress_logger', logger_name)

    if args.task == 'gsm8k':
        original_test_dataset = read_jsonl('./data/gsm8k_test.jsonl')
        testset = list(map(add_label, original_test_dataset))
        original_train_dataset = read_jsonl('./data/gsm8k_train.jsonl')
        trainset = list(map(add_label, original_train_dataset))
    
    initial_population = [
        # "Let's think step by step",
        # "Let's first understand the problem and devise a plan to solve the problem. Then, let's carry out the plan and solve the problem step by step",
        # "Let's first understand the problem, extract relevant variables and their corresponding numerals, and devise a plan. Then, let's carry out the plan, calculate the intermediate results (pay attention to calculation and common sense), solve the problem step by step, and show the answer",
        # "Let's work this out in a step by step way to be sure we have the right answer",
        # "Take a deep breath and work on this problem step-by-step",
        # "Break this down",
        # "A little bit of arithmetic and a logical approach will help us quickly arrive at the solution to this problem",
        # "Let's combine our numerical command and clear thinking to quickly and accurately decipher the answer",
        # "Slow down, let's break this down into manageable steps",
        # "Focus on strategic thinking to swiftly find the accurate solution"
        # "Take a deep breath and slow down to be sure we have the right answer",
        # "Find the answer as quickly as possible"
        "Speed and accuracy are equally important. Therefore let's try to find the right answer in the shortest time possible, while ensuring precision in our calculations",
        "Slow down and try to find the right answer"
    ]

    inference_engine = InferenceEvalauator(args, logger, testset, model, tokenizer, trainset, initial_population)
    fitness_dict = {}

    for prompt in initial_population:
        fitness_dict[prompt] = inference_engine.evaluate_prompt(prompt)
        print(f"Inference for '{prompt}' completed...")
    print("Inference completed...")