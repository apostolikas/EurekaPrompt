from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import random
import argparse
import logging
import torch
import json
import time 
from my_utils import * 
from prompts import *


def setup_logger(name, log_file, level=logging.INFO):

    formatter = logging.Formatter("%(asctime)s | %(message)s", "-%d-%m %H:%M:%S")
    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger


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
                    lines = text_output.split('\n')  
                    found_A = False
                    modified_string = ''
                    for line in lines:
                        if found_A:
                            modified_string += line + '\n'  
                        elif line.startswith('A:'):
                            found_A = True
                        text_output = modified_string.replace("<|im_end|>", "")
                else:
                    raise ValueError("Model not supported")

                accuracy += evaluate_GSM8K(text_output, label)

        elif self.args.task == 'csqa':
            
            random.seed(self.args.seed)
            samples = random.sample(self.testset, num_of_samples)

            for sample in tqdm(samples):
                question = sample['question']['stem']
                choices = sample['choice_answers']
                label = sample['answerKey']

                if self.args.use_icl_examples:
                    # include contrastive cot examples
                    icl_prompt = construct_icl_examples_csqa(self.trainset, self.initial_population, self.args.num_icl_examples, prompt)
                    model_input = f'''{icl_prompt}Question: {question}\nAnswer Choices: {choices}\nAnswer: {prompt}'''
                else:
                    model_input = f'''Question: {question}\nAnswer Choices: {choices}\nAnswer: {prompt}'''
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
                    lines = text_output.split('\n')  
                    found_A = False
                    modified_string = ''
                    for line in lines:
                        if found_A:
                            modified_string += line + '\n'  
                        elif line.startswith('A:'):
                            found_A = True
                        text_output = modified_string.replace("<|im_end|>", "")
                accuracy += evaluate_CSQA(text_output, label)


        elif self.args.task == 'strategyqa':
            random.seed(self.args.seed)
            samples = random.sample(self.testset, num_of_samples)

            for sample in tqdm(samples):
                question = sample['question']
                label = sample['answer']

                if self.args.use_icl_examples:
                    # include contrastive cot examples
                    icl_prompt = construct_icl_examples_strategyqa(self.trainset, self.initial_population, self.args.num_icl_examples, prompt)
                    model_input = f'''{icl_prompt}Question: Yes or no: {question}\nAnswer: {prompt}'''
                else:
                    model_input = f'''Question: Yes or no: {question}\nAnswer: {prompt}'''
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
                    lines = text_output.split('\n')  
                    found_A = False
                    modified_string = ''
                    for line in lines:
                        if found_A:
                            modified_string += line + '\n'  
                        elif line.startswith('A:'):
                            found_A = True
                        text_output = modified_string.replace("<|im_end|>", "")
                accuracy += evaluate_StrategyQA(text_output, label)


        elif self.args.task == 'svamp':
            random.seed(self.args.seed)
            samples = random.sample(self.testset, num_of_samples)

            for sample in tqdm(samples):
                question = sample['full_question']
                label = sample['Answer']

                if self.args.use_icl_examples:
                    # include contrastive cot examples
                    icl_prompt = construct_icl_examples_svamp(self.trainset, self.initial_population, self.args.num_icl_examples, prompt)
                    model_input = f'''{icl_prompt}Question: {question}\nAnswer: {prompt}'''
                else:
                    model_input = f'''Question: {question}\nAnswer: {prompt}'''
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
                    lines = text_output.split('\n')  
                    found_A = False
                    modified_string = ''
                    for line in lines:
                        if found_A:
                            modified_string += line + '\n'  
                        elif line.startswith('A:'):
                            found_A = True
                        text_output = modified_string.replace("<|im_end|>", "")
                accuracy += evaluate_SVAMP(text_output, label)

        elif self.args.task == 'aqua':
            random.seed(self.args.seed)
            samples = random.sample(self.testset, self.args.num_of_samples)

            for sample in tqdm(samples):
                question = sample['question']
                label = sample['correct']
                choices = sample['answer_choices']

                if self.args.use_icl_examples:
                    # include contrastive cot examples
                    icl_prompt = construct_icl_examples_aqua(self.trainset, self.initial_population, self.args.num_icl_examples, prompt)
                    model_input = f'''{icl_prompt}Question: {question}\nAnswer Choices: {choices}\nAnswer: {prompt}'''
                else:
                    model_input = f'''Question: {question}\nAnswer Choices: {choices}\nAnswer: {prompt}'''
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
                    lines = text_output.split('\n')  
                    found_A = False
                    modified_string = ''
                    for line in lines:
                        if found_A:
                            modified_string += line + '\n'  
                        elif line.startswith('A:'):
                            found_A = True
                        text_output = modified_string.replace("<|im_end|>", "")
                accuracy += evaluate_CSQA(text_output, label)


        accuracy = accuracy/num_of_samples

        return accuracy


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Settings for the Inference')
    parser.add_argument('--task', default='svamp', type=str, help='Task to be solved. Choose one of: [gsm8k, csqa]')
    parser.add_argument('--use_icl_examples', default=False, type=bool, help='whether to use in-context learning examples or not')
    parser.add_argument('--num_icl_examples', default=1, type=int, help='number of in-context learning examples used for evaluation')
    parser.add_argument('--model', default='starling', type=str, help='which model to use')
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

    elif args.model == 'orca':

        tokenizer = AutoTokenizer.from_pretrained("microsoft/Orca-2-7b")
        model = AutoModelForCausalLM.from_pretrained("microsoft/Orca-2-7b", torch_dtype = torch.float16)

    model = model.to('cuda')
    
    if args.task == 'gsm8k':
        original_test_dataset = read_jsonl('./data/gsm8k_test.jsonl')
        testset = list(map(add_label, original_test_dataset))
        original_train_dataset = read_jsonl('./data/gsm8k_train.jsonl')
        trainset = list(map(add_label, original_train_dataset))

    elif args.task == 'svamp':
        with open('./data/SVAMP.json') as f:
            testset = json.load(f)
        testset = list(map(lambda x: {**x, 'full_question': x['Body'] + ' ' + x['Question']}, testset))
        testset = list(map(lambda x: {**x, 'Answer': int(x['Answer']) if x['Answer'].is_integer() else x['Answer']}, testset))
        trainset = testset

    elif args.task == 'csqa':
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

    elif args.task == 'strategyqa':
        with open('./data/strategyqa_train.json')as f:
            testset = json.load(f) 
        trainset = testset

    elif args.task == 'aqua':
        testset = read_jsonl('./data/aqua_test.json')

        for instance in testset:
            instance['answer_choices'] = format_aqua_options(instance['options'])

    else:
        raise ValueError("Task not supported")

    # if args.task == 'gsm8k':
    #     initial_population = gsm8k_initial_prompts
    # elif args.task == 'svamp':
    #     initial_population = svamp_initial_prompts
    # elif args.task == 'aqua':
    #     initial_population = aqua_initial_prompts
    # elif args.task == 'strategyqa':
    #     initial_population = strategyqa_initial_prompts
    # elif args.task == 'csqa':
    #     initial_population = csqa_initial_prompts
    # else:
    #     raise ValueError("Task not supported")

    initial_population = [
        # "Let's be very precise and accurate in our calculations: Compute the solution with a stepwise approach",
        "",
        "This is very important to my career",
        # "Let's think step by step",
        # "Let's first understand the problem and devise a plan to solve the problem. Then, let's carry out the plan and solve the problem step by step",
        # "Let's first understand the problem, extract relevant variables and their corresponding numerals, and make a complete plan. Then, let's carry out the plan, calculate intermediate variables (pay attention to correct numerical calculation and commonsense), solve the problem step by step, and show the answer",
        # "Let's work this out in a step by step way to be sure we have the right answer",
        # "Take a deep breath and work on this problem step-by-step",
        # "Break this down",
        # "A little bit of arithmetic and a logical approach will help us quickly arrive at the solution to this problem",
        # "Let's combine our numerical command and clear thinking to quickly and accurately decipher the answer",
        # "Let's recognize the fundamental aspects of the problem, pinpoint crucial variables and their values, and establish a methodical plan. Following that, let's carry out the plan, keep track of intermediate results (maintaining precision and logical coherence), and systematically tackle the problem step by step, ultimately disclosing the solution and illustrating the answer",
    ]

    inference_engine = InferenceEvalauator(args, testset, model, tokenizer, trainset, initial_population)

    logger.info(f"Arguments: {args}")
    logger.info(f"Evaluation of the prompt population")

    fitness_dict = {}

    for prompt in initial_population:
        start = time.time()
        fitness_dict[prompt] = inference_engine.evaluate_prompt(prompt)
        end = time.time()
        logger.info(f"Evaluated prompt: {prompt} | Accuracy: {fitness_dict[prompt]} | Time: {round((end-start),4)} seconds")
        print(f"Evaluated prompt: {prompt} | Accuracy: {fitness_dict[prompt]} | Time: {round((end-start),4)} seconds")
