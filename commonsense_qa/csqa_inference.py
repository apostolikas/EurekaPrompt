from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re 


# def evaluate_csqa(text, label): # without cot
#     # Find the text after GPT 4 Correct Assistant: until \n
#     answer_text = text.split('GPT 4 Correct Assistant:')[1].split('\n')[0]
#     if label + ')' in answer_text or f'({label}' in answer_text:
#         return 1
#     else:
#         return 0

def evaluate_csqa(text, label): # with cot
    answer_text = text.split('GPT 4 Correct Assistant:')[1]

    answer_text = re.findall(r'\((.*?)\)', answer_text)

    answer_text = answer_text[-1]

    if answer_text == label:
        return 1
    else:
        return 0
    

def generate_response(model, tokenizer, model_input):
    input_ids = tokenizer.encode(model_input, return_tensors='pt').to('cuda')
    sample_output = model.generate(
        input_ids,
        max_new_tokens=350, 
        pad_token_id = tokenizer.pad_token_id,
        eos_token_id = tokenizer.eos_token_id,
        do_sample=True,   
    )
    return tokenizer.decode(sample_output[0], skip_special_tokens=True)


class CSQA(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example = self.dataset[index]
        question = example['question']
        mapped_choices = example['choices']['mapped']
        formatted_mapped_choices = ' '.join(mapped_choices)
        answer_key = example['answerKey']

        return question, formatted_mapped_choices, answer_key

def process_choices(example):
    choices_dict = example['choices']
    mapped_choices = [f'({label}) {text}' for label, text in zip(choices_dict['label'], choices_dict['text'])]
    mapped_choices = [f'({label}) {text},' for label, text in zip(choices_dict['label'], choices_dict['text'])]
    mapped_choices[-1] = mapped_choices[-1][:-1]
    choices_dict['mapped'] = mapped_choices
    return example


csqa_dataset = load_dataset('tau/commonsense_qa', split='validation').select(range(35))
csqa = CSQA(csqa_dataset.map(process_choices))

tokenizer = AutoTokenizer.from_pretrained("berkeley-nest/Starling-LM-7B-alpha")
model = AutoModelForCausalLM.from_pretrained("berkeley-nest/Starling-LM-7B-alpha", torch_dtype = torch.float16)
model = model.to('cuda')

population = [
    # "Let's think step by step",
    # "Let's first understand the problem and devise a plan to solve the problem. Then, let's carry out the plan and solve the problem step by step" ,
    # "Take a deep breath and work on this problem step-by-step",
    # "Let's work this out in a step by step way to be sure we have the right answer",
    # "Let's embrace a structured thought process, navigating through the question systematically",
    "Let's focus on logic and reasoning to arrive at a well-considered solution" ,
    "Take a step-by-step approach to this problem" ,
    "Let's embrace a structured thought process, navigating through the problem systematically",
]

for prompt in population:
    acc = 0

    for i, sample in enumerate(csqa):
        question, choices, answer_key = sample
        model_input = f"Question: {question}\n\nChoices: {choices}\n\nAnswer: {prompt}"
        single_turn_input = f"GPT4 Correct User: {model_input} <|end_of_turn|>GPT 4 Correct Assistant:"
        text_output = generate_response(model, tokenizer, single_turn_input)
        result = evaluate_csqa(text_output, answer_key)
        acc += result
        print(f"Example: {i}\nModel Response: {text_output.split('GPT 4 Correct Assistant:')[1]}\nCorrect Answer: {answer_key}\nResult: {result}\n\n")

    print(f"Prompt: {prompt} | Accuracy: {acc/len(csqa)}")
