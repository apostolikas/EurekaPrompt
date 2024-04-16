import evaluate
from prompts import *
from my_utils import *

if __name__ == '__main__':

    model_name = "berkeley-nest/Starling-LM-7B-alpha"

    perplexity = evaluate.load("perplexity", module_type="metric")

    prompts = gsm8k_inference_prompts

    task = 'gsm8k'
    num_of_samples = 50
    prompts = load_inference_prompts(task)
    _, testset = load_data(task)
    random.seed(0)
    samples = random.sample(testset, num_of_samples)

    input_dict = {}

    for prompt in prompts:
        inputs = []
        for i, sample in enumerate(samples):
            if task == 'gsm8k':
                question = sample['question']
                model_input = f'''Question: {question}\nAnswer: {prompt}'''
                input_texts = f'''GPT4 Correct User: {model_input}<|end_of_turn|>GPT4 Correct Assistant:'''

            elif task == 'svamp':
                question = sample['full_question']
                model_input = f'''Question: {question}\nAnswer: {prompt}'''
                input_texts = f'''GPT4 Correct User: {model_input}<|end_of_turn|>GPT4 Correct Assistant:'''

            elif task == 'csqa':    
                question = sample['question']['stem']
                choices = sample['choice_answers']
                model_input = f'''Question: {question}\nAnswer Choices: {choices}\nAnswer: {prompt}'''
                input_texts = f'''GPT4 Correct User: {model_input}<|end_of_turn|>GPT4 Correct Assistant:'''

            elif task == 'abs_nar':
                narrative = sample['input']
                answer_choices = sample['answer_choices']
                model_input = f'''Question: Can you choose the most related proverb from the list of 5 proverbs given a narrative?\nNarrative: {narrative}\nAnswer choices: {answer_choices}\nAnswer: {prompt}'''
                input_texts = f'''GPT4 Correct User: {model_input}<|end_of_turn|>GPT4 Correct Assistant:'''

            elif task in ['causal_judg', 'social_iqa', 'date_under', 'sports_und']:
                question = sample['input']
                answer_choices = sample['answer_choices']
                model_input = f'''Question: {question}\nAnswer choices: {answer_choices}\nAnswer: {prompt}'''
                input_texts = f"GPT4 Correct User: {model_input}<|end_of_turn|>GPT4 Correct Assistant:"

            inputs.append(input_texts)
        input_dict[prompt] = inputs


    # Write the results to a file
    with open(f'./perplexities/perplexity_{task}.txt', 'w') as file:
        for prompt, input_texts in input_dict.items():

            file.write(f"Prompt: {prompt}\n")

            results = perplexity.compute(model_id=model_name,
                                        add_start_token=False,
                                        predictions=input_texts)
            
            for i, perplexity_value in enumerate(results['perplexities'], start=1):
                file.write(f"Question: {i} | Perplexity: {perplexity_value}\n")

            print(f"Avg Perplexity for prompt {prompt}: {results['mean_perplexity']}\n")
            file.write(f"Avg Perplexity: {results['mean_perplexity']}\n")
            



