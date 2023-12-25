from utils import *
from datasets import load_dataset
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import random
import torch
import time
import transformers
from prompts.nli_prompts import *
from data_handler import *

def evaluate_ar(y_pred:str, label:str):
    pass


def construct_icl_examples(samples, number_of_examples, seed):
    random.seed(seed)
    icl_samples = random.sample(samples, number_of_examples)
    icl_prompt = f''''''
    
    for icl_sample in icl_samples:
        question = icl_samples['instruction']
        text = icl_samples['input']
        label = icl_sample['output']
        icl_prompt += f"Question: {question}\n\n{text}\n\nAnswer: {label}\n\n"
        
    return icl_prompt


def construct_ins_examples(samples, number_of_examples, seed, instruct):
    random.seed(seed)
    icl_samples = random.sample(samples, number_of_examples)
    icl_prompt = f''''''

    for icl_sample in icl_samples:
        question = icl_samples['instruction']
        text = icl_samples['input']
        label = icl_sample['output']
        icl_prompt += f"Question: {question}\n\n{text}\n\n{instruct}\n\nAnswer: {label}\n\n"
        
    return icl_prompt


def inference(args, model, tokenizer, ar_prompt):

    random.seed(args.seed)
    testset = ReaxysAR().data
    
    acc = 0

    # system_message = "You are Orca, an AI language model created by Microsoft. You are a cautious assistant and a great math teacher that follows instructions carefully."

    filename = f"./Starling_OQA_{args.mode}_{args.ins}_seed_{args.seed}.txt"

    with open(filename, 'w') as f:

        f.write(f"MODE:{args.mode}\tINSTRUCTION:{args.ins}\n")
        f.write(f"CQA instruction: {ar_prompt}")
        
        for i,sample in enumerate(testset):

            question = sample['instruction']
            text = sample['input']
            label = sample['output']
            
            if args.ins == "icl":

                if args.mode == '0shot':
                    model_input = f"Question: {question}\n\n{text}\n\nAnswer: {label}\n\n"

                elif args.mode == '1shot':
                    icl_examples = construct_icl_examples(testset, 1, args.seed)
                    model_input = f"{icl_examples}Question: {question}\n\n{text}\n\nAnswer: {label}\n\n"

                elif args.mode == '3shot':
                    icl_examples = construct_icl_examples(testset, 3, args.seed)
                    model_input = f"{icl_examples}Question: {question}\n\n{text}\n\nAnswer: {label}\n\n"

                elif args.mode == '5shot':
                    icl_examples = construct_icl_examples(testset, 5, args.seed)
                    model_input = f"{icl_examples}Question: {question}\n\n{text}\n\nAnswer: {label}\n\n"

                else:
                    raise ValueError("Choose one of: [0shot, 1shot, 3shot, 5shot]")
                
            elif args.ins == "instruct":

                if args.mode == '0shot':
                    model_input = f"Question: {question}\n\n{text}\n\n{ar_prompt}\n\nAnswer: {label}\n\n"

                elif args.mode == '1shot':
                    ins_examples = construct_ins_examples(testset, 1, args.seed, ar_prompt)
                    model_input = f"{ins_examples}Question: {question}\n\n{text}\n\n{ar_prompt}\n\nAnswer: {label}\n\n"

                elif args.mode == '3shot':
                    ins_examples = construct_ins_examples(testset, 3, args.seed, ar_prompt)
                    model_input = f"{ins_examples}Question: {question}\n\n{text}\n\n{ar_prompt}\n\nAnswer: {label}\n\n"

                elif args.mode == '5shot':
                    ins_examples = construct_ins_examples(testset, 5, args.seed, ar_prompt)
                    model_input = f"{ins_examples}Question: {question}\n\n{text}\n\n{ar_prompt}\n\nAnswer: {label}\n\n"

                else:
                    raise ValueError("Choose one of: [0shot, 1shot, 3shot, 5shot]")

            else:
                raise ValueError("Choose one of: [icl, instruct]")

            ### ORCA-2 ###

            # prompt = f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{model_input}<|im_end|>\n<|im_start|>assistant"


            start = time.time()

            ### ORCA-2 ###

            # input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to("cuda")
            # output_ids = model.generate(input_ids)
            # text_output = tokenizer.batch_decode(output_ids)[0]

            ### Mistral 8x7 ###

            # messages = [{"role": "user", "content": model_input}]
            # prompt = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            # outputs = model(prompt, max_new_tokens=1000, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
            # text_output = outputs[0]["generated_text"]

            ### Starling ###

            input_ids = tokenizer(model_input, return_tensors = 'pt').input_ids.to('cuda')
            output_ids = model.generate(input_ids)
            text_output = tokenizer.batch_decode(output_ids)[0]
    
         
            end = time.time()
            
            result = evaluate_ar(text_output, label)
            acc += result

            print(f"Example {i} : {result} with elapsed time {end-start} ")
            
            summary = f"\nEXAMPLE:{i}\nMODEL INPUT:{model_input}\nCORRECT LABEL:{label}\nRESULT:{result}\nLLM:{text_output}\n"
            
            f.write(summary)

        acc = acc/len(testset)

        f.write(f"\nTest set Accuracy: {acc}")

    return acc

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Settings for Baseline')
    parser.add_argument('--mode', default='0shot', type=str, help='Use ICL or not. Choose one of: [0shot, 1shot, 3shot, 5shot]')
    parser.add_argument('--ins', default='instruct', type=str, help='Use instruction or not. Choose one of: [icl, instruct]')
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

    model = AutoModelForCausalLM("berkeley-nest/Starling-LM-7B-alpha", device_map="auto", torch_dtype = torch.float16)
    tokenizer = AutoTokenizer("berkeley-nest/Starling-LM-7B-alpha")

    modes = ['0shot']
    instructions = ['instruct']

    for mod in modes:
        for ins in instructions:
            args.mode = mod
            args.ins = ins
            acc = inference(args, model, tokenizer, cqa_promp=None)
            print(f"Test set Accuracy : {acc}")
