from datasets import load_dataset
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import random
import torch
import time
import transformers
from prompts.math_prompts import *

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

def add_label(entry):

    ans_value = f'''{entry['answer'].split("####")[0]}The answer is {int(entry['answer'].split('####')[1].replace(',', '',))}'''
    entry['ans'] = ans_value
    answer_value = int(entry['answer'].split('####')[1].replace(',', ''))
    entry['label'] = answer_value
    
    return entry

def construct_icl_examples(samples, number_of_examples, seed):
    random.seed(seed)
    icl_examples = random.sample(list(samples), number_of_examples)
    icl_prompt = f''''''

    for i in range(len(icl_examples)):
        icl_prompt += f"Question: {icl_examples[i]['question']}\nThe answer is {icl_examples[i]['label']}\n\n"
    return icl_prompt


def construct_ins_examples(samples, number_of_examples,seed,instruct):
    random.seed(seed)
    ins_examples = random.sample(list(samples), number_of_examples)
    ins_prompt = f''''''

    for i in range(len(ins_prompt)):
        ins_prompt += f"Question: {ins_examples[i]['question']}\n{instruct}\n{ins_examples[i]['ans']}\n\n"
    return ins_prompt


def construct_cot_examples(samples, number_of_examples,seed):
    random.seed(seed)
    cot_examples = random.sample(list(samples), number_of_examples)
    cot_prompt = f''''''

    for i in range(len(cot_examples)):
        cot_prompt += f"Question: {cot_examples[i]['question']}\nLet's think step by step.\n{cot_examples[i]['ans']}\n\n"
    return cot_prompt

def construct_ps_examples(samples, number_of_examples,seed):
    random.seed(seed)
    cot_examples = random.sample(list(samples), number_of_examples)
    cot_prompt = f''''''

    for i in range(len(cot_examples)):
        cot_prompt += f"Question: {cot_examples[i]['question']}\nLet's first understand the problem and devise a plan to solve the problem. Then, let's carry out the plan and solve the problem step by step.\n{cot_examples[i]['ans']}\n\n"
    return cot_prompt

def inference(args, model, tokenizer, math_prompt):

    original_test_dataset = load_dataset("gsm8k", 'main', split='test')
    original_train_dataset = load_dataset("gsm8k", 'main', split='train')

    trainset = original_train_dataset.map(add_label)
    testset = original_test_dataset.map(add_label)

    testset = testset.shuffle(seed=args.seed).select(range(100))

    acc = 0

    # system_message = "You are Orca, an AI language model created by Microsoft. You are a cautious assistant and a great math teacher that follows instructions carefully."

    system_message = "You are an AI language model. You are a cautious assistant and a great math teacher that follows instructions carefully."


    filename = f"./mistral_experiment_{args.mode}_{args.ins}_seed_{args.seed}.txt"

    with open(filename, 'w') as f:

        f.write(f"MODE:{args.mode}\tINSTRUCTION:{args.ins}\n")
        
        for i,sample in enumerate(testset):

            question = sample['question']
            label = sample['label']
                
            if args.ins == 'cot':

                if args.mode == '0shot':

                    model_input = f'''Question: {question}\nLet's think step by step.'''

                elif args.mode == '1shot':

                    cot_examples = construct_cot_examples(trainset, 1, args.seed)
                    model_input = f'''{cot_examples}Question: {question}\nLet's think step by step.'''

                elif args.mode == '3shot':

                    cot_examples = construct_cot_examples(trainset, 3, args.seed)
                    model_input = f'''{cot_examples}Question: {question}\nLet's think step by step.'''

                elif args.mode == '5shot':
                        
                    icl_examples = construct_cot_examples(trainset, 5, args.seed)
                    model_input = f'''{icl_examples}Question: {question}\nLet's think step by step.'''

                else:

                    raise ValueError("Choose one of: [0shot, 1shot, 3shot, 5shot]")

            elif args.ins == "icl":

                if args.mode == '0shot':

                    model_input = f'''Question: {question}'''

                elif args.mode == '1shot':

                    icl_examples = construct_icl_examples(trainset, 1, args.seed)
                    model_input = f'''{icl_examples}Question: {question}'''

                elif args.mode == '3shot':

                    icl_examples = construct_icl_examples(trainset, 3, args.seed)
                    model_input = f'''{icl_examples}Question: {question}'''

                elif args.mode == '5shot':
                        
                    icl_examples = construct_icl_examples(trainset, 5, args.seed)
                    model_input = f'''{icl_examples}Question: {question}'''

                else:

                    raise ValueError("Choose one of: [0shot, 1shot, 3shot, 5shot]")

            elif args.ins == "ps":

                if args.mode == '0shot':

                    model_input = f'''Question: {question}\nLet's first understand the problem and devise a plan to solve the problem. Then, let's carry out the plan and solve the problem step by step.'''

                elif args.mode == '1shot':

                    cot_examples = construct_ps_examples(trainset, 1, args.seed)
                    model_input = f'''{cot_examples}Question: {question}\nLet's first understand the problem and devise a plan to solve the problem. Then, let's carry out the plan and solve the problem step by step.'''

                elif args.mode == '3shot':

                    cot_examples = construct_ps_examples(trainset, 3, args.seed)
                    model_input = f'''{cot_examples}Question: {question}\nLet's first understand the problem and devise a plan to solve the problem. Then, let's carry out the plan and solve the problem step by step.'''

                elif args.mode == '5shot':
                        
                    icl_examples = construct_ps_examples(trainset, 5, args.seed)
                    model_input = f'''{icl_examples}Question: {question}\nLet's first understand the problem and devise a plan to solve the problem. Then, let's carry out the plan and solve the problem step by step.'''

                else:

                    raise ValueError("Choose one of: [0shot, 1shot, 3shot, 5shot]")
                
            elif args.ins == "instruct":

                if args.mode == '0shot':

                    model_input = f'''Question: {question}\n{math_prompt}'''

                elif args.mode == '1shot':

                    ins_examples = construct_ins_examples(trainset, 1, args.seed, math_prompt)
                    model_input = f'''{ins_examples}Question: {question}\n{math_prompt}'''

                elif args.mode == '3shot':

                    ins_examples = construct_ins_examples(trainset, 3, args.seed, math_prompt)
                    model_input = f'''{ins_examples}Question: {question}\n{math_prompt}'''

                elif args.mode == '5shot':
                        
                    ins_examples = construct_ins_examples(trainset, 5, args.seed, math_prompt)
                    model_input = f'''{ins_examples}Question: {question}\n{math_prompt}'''

                else:

                    raise ValueError("Choose one of: [0shot, 1shot, 3shot, 5shot]")

            else:

                raise ValueError("Choose one of: [icl, cot, ps, instruct]")

            ### ORCA-2 ###

            # prompt = f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{model_input}<|im_end|>\n<|im_start|>assistant"

            ### Mistral Instruct ###


            role = "From now on, you are an excellent math teacher and always teach your students math problems correctly. And I am one of your students."

            role_feedback = "As your math teacher, I'll do my best to explain mathematical concepts correctly so that you can understand them easily. Let's solve the math problems together."

            prompt = [
            {"role": "user", "content": role},
            {"role": "assistant", "content": role_feedback},
            {"role": "user", "content": model_input}
            ]

            # prompt = [
            # {"role": "user", "content": "You are an AI language model. You are a cautious assistant and a great math teacher that follows instructions carefully."},
            # {"role": "assistant", "content": "I will follow the instructions in order to solve the math problems you give me."},
            # {"role": "user", "content": model_input}
            # ]

            start = time.time()

            ### Mistral Instruct ###

            encodeds = tokenizer.apply_chat_template(prompt, return_tensors="pt")
            model_inputs = encodeds.to('cuda')
            generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
            decoded = tokenizer.batch_decode(generated_ids)
            text_output = decoded[0]

            ### ORCA-2 ###

            # input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to("cuda")
            # output_ids = model.generate(input_ids)
            # text_output = tokenizer.batch_decode(output_ids)[0]


            ### Mistral 8x7 ###

            # messages = [{"role": "user", "content": model_input}]
            # prompt = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            # outputs = model(prompt, max_new_tokens=1000, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
            # text_output = outputs[0]["generated_text"]

         
            end = time.time()
            
            result = evaluate_GSM8K(text_output, label)
            acc += result

            print(f"Example {i} : {result} with elapsed time {end-start} ")
            
            summary = f"EXAMPLE:{i}\nMODEL INPUT:{model_input}\nCORRECT LABEL:{label}\nRESULT:{result}\nLLM:{text_output}\n"
            
            f.write(summary)

        acc = acc/len(testset)

        f.write(f"\nTest set Accuracy: {acc}")

    return acc

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Settings for Baseline')
    parser.add_argument('--mode', default='5shot', type=str, help='Use ICL or not. Choose one of: [0shot, 1shot, 3shot, 5shot]')
    parser.add_argument('--ins', default='cot', type=str, help='Use instruction or not. Choose one of: [icl, cot, ps, instruct]')
    parser.add_argument('--seed', default=0, type=int, help='Random seed to use for sampling')

    args = parser.parse_args()

    # model = AutoModelForCausalLM.from_pretrained("microsoft/Orca-2-7b", device_map = 'auto', torch_dtype = torch.float16)
    # tokenizer = AutoTokenizer.from_pretrained("microsoft/Orca-2-7b", use_fast = False)

    # model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = transformers.pipeline(
    #     "text-generation",
    #     model=model_name,
    #     model_kwargs={"torch_dtype": torch.float16, "load_in_4bit": True},
    # )


    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", device_map = 'auto', torch_dtype = torch.float16)
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

    math_prompts = shorter_prompts
    modes = ['0shot']
    instructions = ['instruct']

    for math_prompt in math_prompts:
        for mod in modes:
            for ins in instructions:
                args.mode = mod
                args.ins = ins
                acc = inference(args, model, tokenizer, math_prompt)
                print(f"Test set Accuracy for 100 samples: {acc}")
