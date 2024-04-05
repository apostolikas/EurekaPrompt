from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
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
        num_of_samples = 2 #len(self.testset)
        file_name = f"./inference_logs/{self.args.task}_answers.txt"

        with open(file_name, 'w') as f:

            f.write(f"Prompt: {prompt}\n")
            
            if self.args.task == 'gsm8k':

                random.seed(self.args.seed)
                samples = random.sample(self.testset, num_of_samples)

                for i, sample in enumerate(tqdm(samples)):
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

                    elif self.args.model == 'mixtral':
                        input_prompt = f"<s> [INST] {model_input} [/INST]"
                        inputs = self.tokenizer(input_prompt, return_tensors="pt").to('cuda')
                        outputs = model.generate(**inputs, max_new_tokens=300, pad_token_id = tokenizer.eos_token_id)
                        text_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
                        text_output = text_output.split("[/INST]")[1]

                    else:
                        raise ValueError("Model not supported")
                    
                    result = evaluate_GSM8K(text_output, label)
                    f.write(f"Question: {i} | Result: {result}\n")
                    accuracy += result


            elif self.args.task == 'csqa':
                
                file_name = f"./inference_logs/{self.args.task}_answers.txt"
                random.seed(self.args.seed)
                samples = random.sample(self.testset, num_of_samples)

                for i, sample in enumerate(tqdm(samples)):
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

                    result = evaluate_CSQA(text_output, label)
                    f.write(f"Question: {i} | Result: {result}\n")
                    accuracy += result

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

                file_name = f"./inference_logs/{self.args.task}_answers.txt"
                random.seed(self.args.seed)
                samples = random.sample(self.testset, num_of_samples)

                for i, sample in enumerate(tqdm(samples)):
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

                    result = evaluate_SVAMP(text_output, label)
                    f.write(f"Question: {i} | Result: {result}\n")
                    accuracy += result

            elif self.args.task == 'aqua':
                random.seed(self.args.seed)
                samples = random.sample(self.testset, num_of_samples)

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
                        outputs = self.model.generate(input_ids, max_new_tokens=300, pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id)
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

            elif self.args.task == 'abs_nar':

                file_name = f"./inference_logs/{self.args.task}_answers.txt"
                num_of_samples = len(self.testset['examples'])
                random.seed(self.args.seed)
                samples = random.sample(self.testset['examples'], num_of_samples)

                for i, sample in enumerate(tqdm(samples)):
                    narrative = sample['input']
                    label = sample['label']
                    answer_choices = sample['answer_choices']
                    model_input = f'''Question: Can you choose the most related proverb from the list of 5 proverbs given a narrative?\nNarrative: {narrative}\nAnswer choices: {answer_choices}\nAnswer: {prompt}'''
                    input_prompt = f'''GPT4 Correct User: {model_input}<|end_of_turn|>GPT4 Correct Assistant:'''
                    input_ids = self.tokenizer(input_prompt, return_tensors="pt").input_ids.to('cuda')
                    outputs = self.model.generate(input_ids, max_new_tokens=250, pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id)
                    response_ids = outputs[0]
                    text_output = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                    text_output = text_output.split("GPT4 Correct Assistant:")[1]
                    result = evaluate_CSQA(text_output, label)
                    f.write(f"Question: {i} | Result: {result}\n")
                    accuracy += result
                    
            elif self.args.task == 'disamb':
                file_name = f"./inference_logs/{self.args.task}_answers.txt"
                num_of_samples = len(self.testset['examples'])
                random.seed(self.args.seed)
                samples = random.sample(self.testset['examples'], num_of_samples)
                question = 'Can you clarify the meaning of the sentence with ambiguous pronouns?'

                for i, sample in enumerate(tqdm(samples)):
                    context = sample['input']
                    label = sample['label']
                    answer_choices = sample['answer_choices']
                    model_input = f'''Question: {question}\nSentence: {context}\nAnswer choices: {answer_choices}\nAnswer: {prompt}'''
                    input_prompt = f'''GPT4 Correct User: {model_input}<|end_of_turn|>GPT4 Correct Assistant:'''
                    input_ids = self.tokenizer(input_prompt, return_tensors="pt").input_ids.to('cuda')
                    outputs = self.model.generate(input_ids, max_new_tokens=250, pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id)
                    response_ids = outputs[0]
                    text_output = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                    text_output = text_output.split("GPT4 Correct Assistant:")[1]
                    result = evaluate_CSQA(text_output, label)
                    f.write(f"Question: {i} | Result: {result}\n")
                    accuracy += result

            elif self.args.task == 'logic_ded3':
                file_name = f"./inference_logs/{self.args.task}_answers.txt"
                num_of_samples = len(self.testset['examples'])
                random.seed(self.args.seed)
                samples = random.sample(self.testset['examples'], num_of_samples)
                question = 'What is the correct answer based on the context?'

                for i, sample in enumerate(tqdm(samples)):
                    context = sample['input']
                    label = sample['label']
                    answer_choices = sample['answer_choices']
                    model_input = f'''Question: {question}\nContext: {context}\nAnswer choices: {answer_choices}\nAnswer: {prompt}'''
                    input_prompt = f'''GPT4 Correct User: {model_input}<|end_of_turn|>GPT4 Correct Assistant:'''
                    input_ids = self.tokenizer(input_prompt, return_tensors="pt").input_ids.to('cuda')
                    outputs = self.model.generate(input_ids, max_new_tokens=250, pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id)
                    response_ids = outputs[0]
                    text_output = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                    text_output = text_output.split("GPT4 Correct Assistant:")[1]
                    result = evaluate_CSQA(text_output, label)
                    f.write(f"Question: {i} | Result: {result}\n")
                    accuracy += result

            elif self.args.task == 'social_iqa' or self.args.task == 'sports_und' or self.args.task == 'date_under' or self.args.task == 'causal_judg':
                file_name = f"./inference_logs/{self.args.task}_answers.txt"
                num_of_samples = len(self.testset['examples'])
                random.seed(self.args.seed)
                samples = random.sample(self.testset['examples'], num_of_samples)

                for i, sample in enumerate(tqdm(samples)):
                    question = sample['input']
                    label = sample['label']
                    answer_choices = sample['answer_choices']
                    model_input = f'''Question: {question}\nAnswer choices: {answer_choices}\nAnswer: {prompt}'''
                    input_prompt = f'''GPT4 Correct User: {model_input}<|end_of_turn|>GPT4 Correct Assistant:'''
                    input_ids = self.tokenizer(input_prompt, return_tensors="pt").input_ids.to('cuda')
                    outputs = self.model.generate(input_ids, max_new_tokens=250, pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id)
                    response_ids = outputs[0]
                    text_output = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                    text_output = text_output.split("GPT4 Correct Assistant:")[1]
                    result = evaluate_CSQA(text_output, label)
                    f.write(f"Question: {i} | Result: {result}\n")
                    accuracy += result

        accuracy = accuracy/num_of_samples

        return accuracy


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Settings for the Inference')
    parser.add_argument('--task', default='gsm8k', type=str, help='Task to be solved. Choose one of: [gsm8k, csqa]')
    parser.add_argument('--use_icl_examples', default=False, type=bool, help='whether to use in-context learning examples or not')
    parser.add_argument('--num_icl_examples', default=1, type=int, help='number of in-context learning examples used for evaluation')
    parser.add_argument('--model', default='starling', type=str, help='which model to use')
    parser.add_argument('--seed', default=0, type=int, help='type of mutation')
    args = parser.parse_args()
    
    logger_name = f"./inference_logs/Inference_Eval_{args.task}_output.log"
    logger = setup_logger('progress_logger', logger_name)

    bb_tasks = ['abs_nar', 'causal_judg', 'date_under', 'disamb', 'logic_ded3', 'social_iqa', 'sports_und']

    if args.model == 'starling':
        tokenizer = AutoTokenizer.from_pretrained("berkeley-nest/Starling-LM-7B-alpha")
        model = AutoModelForCausalLM.from_pretrained("berkeley-nest/Starling-LM-7B-alpha", torch_dtype = torch.float16, device_map="auto")

    elif args.model == 'openhermes':
        tokenizer = AutoTokenizer.from_pretrained("teknium/OpenHermes-2.5-Mistral-7B")
        model = AutoModelForCausalLM.from_pretrained("teknium/OpenHermes-2.5-Mistral-7B", torch_dtype = torch.float16, device_map="auto")

    elif args.model == 'openchat':
        tokenizer = AutoTokenizer.from_pretrained("openchat/openchat_3.5")
        model = AutoModelForCausalLM.from_pretrained("openchat/openchat_3.5", torch_dtype = torch.float16, device_map="auto")

    elif args.model == 'orca':
        tokenizer = AutoTokenizer.from_pretrained("microsoft/Orca-2-7b")
        model = AutoModelForCausalLM.from_pretrained("microsoft/Orca-2-7b", torch_dtype = torch.float16, device_map="auto")

    elif args.model == 'mixtral':
          
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
        model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1", quantization_config=bnb_config, device_map="auto")
    
    # model = model.to('cuda')
    
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
        trainset = testset

    elif args.task in bb_tasks:
        with open(f'./data/{args.task}.json') as f:
            testset = json.load(f)

        testset['examples'] = list(map(process_bb_example, testset['examples']))
        trainset = testset

    else:
        raise ValueError("Task not supported")

    initial_population = [
        "Let's think step by step",
        "Let's first understand the problem and devise a plan to solve the problem. Then, let's carry out the plan and solve the problem step by step",
        "Let's first understand the problem, extract relevant variables and their corresponding numerals, and make a complete plan. Then, let's carry out the plan, calculate intermediate variables (pay attention to correct numerical calculation and commonsense), solve the problem step by step, and show the answer",
        "Let's work this out in a step by step way to be sure we have the right answer",
        "Take a deep breath and work on this problem step-by-step",
        "Focus on strategic thinking to swiftly find the accurate solution",
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
