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

def evaluate_ner(y_pred:str, ground_truth:dict):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    labels = ground_truth.values()
    
    predicted_entities = re.findall(r'\[(.*?)\]', y_pred)
    predicted_entities = [entity.split(',') for entity in predicted_entities]

    # Count true positives, false positives, and false negatives
    for entity in predicted_entities:
        if entity in labels:
            true_positives += 1
        else:
            false_positives += 1

    for entity in labels:
        if entity not in predicted_entities:
            false_negatives += 1

    return true_positives, false_positives, false_negatives


def construct_icl_examples(samples, number_of_examples, seed):
    random.seed(seed)
    icl_samples = random.sample(samples, number_of_examples)
    icl_prompt = f''''''
    
    for icl_sample in icl_samples:
        text = icl_sample['input']
        output = icl_sample['output']
        icl_prompt += f"Q: What entities can be found in the text?\n\nText: {text}\n\nA: {output}"

    return icl_prompt


def construct_ins_examples(samples, number_of_examples, seed, instruct):
    random.seed(seed)
    icl_samples = random.sample(samples, number_of_examples)
    icl_prompt = f''''''

    for icl_sample in icl_samples:
        text = icl_sample['input']
        output = icl_sample['output']
        icl_prompt += f"Q: What entities can be found in the text?\n\nText: {text}\n\n{instruct}\n\nA: {output}"

    return icl_prompt


def inference(args, model, tokenizer, ner_prompt):

    random.seed(args.seed)
    testset = EmbaseNLI().data
    
    true_positives, false_positives, false_negatives = 0, 0, 0

    # system_message = "You are Orca, an AI language model created by Microsoft. You are a cautious assistant and a great math teacher that follows instructions carefully."

    filename = f"./Starling_NER_{args.mode}_{args.ins}_seed_{args.seed}.txt"

    with open(filename, 'w') as f:

        f.write(f"MODE:{args.mode}\tINSTRUCTION:{args.ins}\n")
        f.write(f"NER instruction: {ner_prompt}")
        
        for i,sample in enumerate(testset):

            text = sample['input']
            label = sample['output']
            
            if args.ins == "icl":

                if args.mode == '0shot':
                    model_input = f"Q: What entities can be found in the text?\n\nText: {text}\n\nA:"

                elif args.mode == '1shot':
                    icl_examples = construct_icl_examples(testset, 1, args.seed)
                    model_input = f"{icl_examples}Q: What entities can be found in the text?\n\nText: {text}\n\nA:"

                elif args.mode == '3shot':
                    icl_examples = construct_icl_examples(testset, 3, args.seed)
                    model_input = f"{icl_examples}Q: What entities can be found in the text?\n\nText: {text}\n\nA:"

                elif args.mode == '5shot':
                    icl_examples = construct_icl_examples(testset, 5, args.seed)
                    model_input = f"{icl_examples}Q: What entities can be found in the text?\n\nText: {text}\n\nA:"

                else:
                    raise ValueError("Choose one of: [0shot, 1shot, 3shot, 5shot]")
                
            elif args.ins == "instruct":

                if args.mode == '0shot':
                    model_input = f"Q: What entities can be found in the text?\n\nText: {text}\n\n{ner_prompt}\n\nA:"

                elif args.mode == '1shot':
                    ins_examples = construct_ins_examples(testset, 1, args.seed, ner_prompt)
                    model_input = f"{ins_examples}Q: What entities can be found in the text?\n\nText: {text}\n\n{ner_prompt}\n\nA:"

                elif args.mode == '3shot':
                    ins_examples = construct_ins_examples(testset, 3, args.seed, ner_prompt)
                    model_input = f"{ins_examples}Q: What entities can be found in the text?\n\nText: {text}\n\n{ner_prompt}\n\nA:"

                elif args.mode == '5shot':
                    ins_examples = construct_ins_examples(testset, 5, args.seed, ner_prompt)
                    model_input = f"{ins_examples}Q: What entities can be found in the text?\n\nText: {text}\n\n{ner_prompt}\n\nA:"

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
            
            sample_tp, sample_fp, sample_fn = evaluate_ner(text_output, label)
            true_positives += sample_tp
            false_positives += sample_fp
            false_negatives += sample_fn

            print(f"Example {i} time elapsed {end-start} ")
            
            summary = f"EXAMPLE:{i}\nMODEL INPUT:{model_input}\nCORRECT LABEL:{label}\n\nLLM:{text_output}\n"
            
            f.write(summary)


        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0        

        f.write(f"\nTest set Precision: {precision}")
        f.write(f"\nTest set Recall: {recall}")
        f.write(f"\nTest set F1 score: {f1_score}")

    return precision, recall, f1_score

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

    model = AutoModelForCausalLM("berkeley-nest/Starling-LM-7B-alpha", torch_dtype = torch.float16)
    tokenizer = AutoTokenizer("berkeley-nest/Starling-LM-7B-alpha")
    model = model.to('cuda')

    modes = ['0shot']
    instructions = ['icl']

    for mod in modes:
        for ins in instructions:
            args.mode = mod
            args.ins = ins
            precision, recall, f1_score = inference(args, model, tokenizer, ner_prompt=None)
            print(f"Test set Precision: {precision}")
            print(f"Test set Recall: {recall}")
            print(f"Test set F1 score: {f1_score}")

