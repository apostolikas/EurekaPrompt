from my_utils import *
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from prompts import *
import re
from itertools import combinations

def evaluate_prompt(model, task, testset, prompt):

    accuracy = 0 
    num_of_samples = 50
    seed = 0 

    if task == 'gsm8k':
        random.seed(seed)
        samples = random.sample(testset, num_of_samples)

        for sample in samples:
            question = sample['question']
            label = sample['label']
            model_input = f'''Question: {question}\nAnswer: {prompt}'''
            input_prompt = f'''GPT4 Correct User: {model_input}<|end_of_turn|>GPT4 Correct Assistant:'''
            text_output = model.get_response(input_prompt)
            accuracy += evaluate_GSM8K(text_output, label)

        accuracy = accuracy/num_of_samples

    elif task == 'csqa':
            
        random.seed(seed)
        samples = random.sample(testset, num_of_samples)

        for sample in samples:
            question = sample['question']['stem']
            choices = sample['choice_answers']
            label = sample['answerKey']
            model_input = f'''Question: {question}\nAnswer Choices: {choices}\nAnswer: {prompt}'''
            input_prompt = f'''GPT4 Correct User: {model_input}<|end_of_turn|>GPT4 Correct Assistant:'''
            text_output = model.get_response(input_prompt)
            text_output = text_output.split('GPT4 Correct Assistant:')[1]
            accuracy += evaluate_CSQA(text_output, label)

        accuracy = accuracy/num_of_samples

    elif task == 'svamp':

        random.seed(seed)
        samples = random.sample(testset, num_of_samples)

        for sample in samples:
            question = sample['full_question']
            label = sample['Answer']
            model_input = f'''Question: {question}\nAnswer: {prompt}'''
            input_prompt = f'''GPT4 Correct User: {model_input}<|end_of_turn|>GPT4 Correct Assistant:'''
            text_output = model.get_response(input_prompt)
            text_output = text_output.split('GPT4 Correct Assistant:')[1]
            accuracy += evaluate_SVAMP(text_output, label)

        accuracy = accuracy/num_of_samples

    elif task == 'abs_nar':
            
        random.seed(seed)
        samples = random.sample(testset, num_of_samples)

        for sample in samples:
            narrative = sample['input']
            label = sample['label']
            answer_choices = sample['answer_choices']
            model_input = f'''Question: Can you choose the most related proverb from the list of 5 proverbs given a narrative?\nNarrative: {narrative}\nAnswer choices: {answer_choices}\nAnswer: {prompt}'''
            input_prompt = f'''GPT4 Correct User: {model_input}<|end_of_turn|>GPT4 Correct Assistant:'''
            text_output = model.get_response(input_prompt)
            text_output = text_output.split('GPT4 Correct Assistant:')[1]
            accuracy += evaluate_CSQA(text_output, label)

        accuracy = accuracy/num_of_samples

    elif task == 'disamb':

        random.seed(seed)
        samples = random.sample(testset, num_of_samples)
        question = 'Can you claritfy the meaning of the sentence with ambiguous pronouns?'

        for sample in samples:
            context = sample['input']
            label = sample['label']
            answer_choices = sample['answer_choices']
            model_input = f'''Question: {question}\nSentence: {context}\nAnswer choices: {answer_choices}\nAnswer: {prompt}'''
            input_prompt = f'''GPT4 Correct User: {model_input}<|end_of_turn|>GPT4 Correct Assistant:'''
            text_output = model.get_response(input_prompt)
            text_output = text_output.split('GPT4 Correct Assistant:')[1]
            accuracy += evaluate_CSQA(text_output, label)

        accuracy = accuracy/num_of_samples

    elif task == 'social_iqa' or task == 'sports_und' or task == 'date_under' or task == 'causal_judg':

        random.seed(seed)
        samples = random.sample(testset, num_of_samples)

        for sample in samples:
            question = sample['input']
            label = sample['label']
            answer_choices = sample['answer_choices']
            model_input = f'''Question: {question}\nAnswer choices: {answer_choices}\nAnswer: {prompt}'''
            input_prompt = f'''GPT4 Correct User: {model_input}<|end_of_turn|>GPT4 Correct Assistant:'''
            text_output = model.get_response(input_prompt)
            text_output = text_output.split('GPT4 Correct Assistant:')[1]
            accuracy += evaluate_CSQA(text_output, label)

        accuracy = accuracy/num_of_samples

    return accuracy


if __name__ == '__main__':

    tokenizer = AutoTokenizer.from_pretrained("berkeley-nest/Starling-LM-7B-alpha")
    model = AutoModelForCausalLM.from_pretrained("berkeley-nest/Starling-LM-7B-alpha", torch_dtype = torch.float16, device_map = 'auto')

    model = SocraticGPT(model, tokenizer)
    crossover = True
    mutation = False
    tasks = ['gsm8k', 'svamp', 'csqa', 'abs_nar', 'disamb', 'sports_und', 'date_under']

    for task in tasks:

        print(f"Task: {task}")

        transet, testset = load_data(task)

        initial_population = load_inference_prompts(task)
        mutation_styles = load_mutation_prompts(task)

        avg_child_socratic = []
        avg_child_normal = []
        avg_mut_socratic = []
        avg_mut_normal = []
        
        crossover = True
        mutation = False

        if crossover:

            for parent0, parent1 in combinations(initial_population, 2):

                print(f"Parent 1: {parent0}")
                print(f"Parent 2: {parent1}")

                final_prompt = crossover_dialogue(model, parent0, parent1)
                print(f"LLM Conversation output: {final_prompt}")

                socratic_child = final_prompt[-1]
                print(f'The Socratic child prompt is: "{socratic_child}"')
                acc0 = evaluate_prompt(model, task, testset, socratic_child)
                print(f'Accuracy with Socratic child: {acc0}')

                avg_child_socratic.append(acc0)

                input_text = f'''GPT4 Correct System: You will help me perform a crossover of two parent texts for an evolutionary algorithm. The child text has to be one sentence that will combine elements from both parent texts. \nParent1: \"[{parent0}]\" \nParent2: \"[{parent1}]\".\nThe child text has to be within brackets.<|end_of_turn|>GPT4 Correct User: Provide your new child text within brackets.<|end_of_turn|>GPT4 Correct Assistant:<|end_of_turn|>'''
                response_text = model.get_response(input_text)
                pattern = r'\[([^\]]+)\]'
                final_prompt = re.findall(pattern, response_text)
                child = final_prompt[-1]
                print(f"The normal child prompt is: {child}")
                acc1 = evaluate_prompt(model, task, testset, child)
                print(f'Accuracy with normal child: {acc1}')
                print()

                avg_child_normal.append(acc1)

            print(f"Average accuracy for Socratic child: {sum(avg_child_socratic)/len(avg_child_socratic)}")
            print(f"Average accuracy for normal child: {sum(avg_child_normal)/len(avg_child_normal)}")


        if mutation:

            for i, prompt in enumerate(initial_population):

                for mutation_style in mutation_styles:

                    # mutation_style = random.choice(mutation_styles)
                    print(f"Mutation style: {mutation_style}")

                    input_text = f'''GPT4 Correct System: You will help me to perform a mutation of a prompt. {mutation_style}\nThe initial prompt is: \"{prompt}\".<|end_of_turn|>GPT4 Correct User: Provide your mutation of the prompt within brackets.<|end_of_turn|>GPT4 Correct Assistant:''' 
                    response_text = model.get_response(input_text)
                    pattern = r'\[([^\]]+)\]'
                    try:
                        final_prompt = re.findall(pattern, response_text)
                        mutated_prompt = final_prompt[-1]
                        print(f"Normal: {type(final_prompt)} The mutated child prompt is: {mutated_prompt}")
                        acc3 = evaluate_prompt(model, task, testset, mutated_prompt)
                        print(f'Accuracy with normal mutation: {acc3}')
                        avg_mut_normal.append(acc3)


                        llm_conversation_mutated = mutation_dialogue(model, mutation_style, prompt, False, None)
                        if isinstance(llm_conversation_mutated, list):
                            theaetetus_mutated_child = llm_conversation_mutated[-1]
                        print(f"The Socratic mutated child prompt is: {theaetetus_mutated_child}")
                        acc2 = evaluate_prompt(model, task, testset, theaetetus_mutated_child)
                        print(f'Accuracy with Socratic mutation: {acc2}')

                        avg_mut_socratic.append(acc2)

                    except:
                        print("Couldn't provide mutated prompts within brackets")
                        continue
                    
            print(f"Average accuracy for Socratic mutation: {sum(avg_mut_socratic)/len(avg_mut_socratic)}")
            print(f"Average accuracy for normal mutation: {sum(avg_mut_normal)/len(avg_mut_normal)}")

        