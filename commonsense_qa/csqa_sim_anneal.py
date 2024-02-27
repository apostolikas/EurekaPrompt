import random
from utils import *
import argparse
import logging
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from conversation import SocraticGPT
import torch
from torch.utils.data import Dataset, DataLoader
import math

def evaluate_csqa(text, label):
    answer_text = text.split('GPT 4 Correct Assistant:')[1]
    answer_text = re.findall(r'\(([^)]*)\)', answer_text)

    if answer_text:
        answer_text = answer_text[-1]

        if answer_text == label:
            return 1
        else:
            return 0
    else:
        # Handle the case where no content is found within parentheses
        return 0

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


def setup_logger(name, log_file, level=logging.INFO):
    '''
    This function sets up the logger.
    '''

    formatter = logging.Formatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    
    return logger


class SimAnneal:
    def __init__(self, args, testset):
        self.args = args
        self.testset = testset
        self.tokenizer = AutoTokenizer.from_pretrained("berkeley-nest/Starling-LM-7B-alpha")
        self.model = AutoModelForCausalLM.from_pretrained("berkeley-nest/Starling-LM-7B-alpha", torch_dtype = torch.float16)
        self.model = self.model.to('cuda')
        self.logger = setup_logger('progress_logger', 'CSQA_SA_output.log')


    def evaluate(self, prompt):
        '''
        This function evaluates the fitness of a prompt by calculating the accuracy of the model on the dataset.
        The current version only supports GSM8K and checks whether the final number in the output is the same as the label.
        '''
        
        score = 0

        testloader = DataLoader(self.testset, batch_size=1, shuffle=True, num_workers=0)

        for i,sample in enumerate(tqdm(testloader)):

            question, choices, answer_key = sample

            model_input = f"Question: {question[0]}\n\nChoices: {choices[0]}\n\nAnswer: {prompt}"
            single_turn_input = f"GPT4 Correct User: {model_input} <|end_of_turn|>GPT 4 Correct Assistant:"
            text_output = generate_response(self.model, self.tokenizer, single_turn_input)
            result = evaluate_csqa(text_output, answer_key[0])
            score += result

        score = score/self.args.num_of_samples
        return score
    

    def mutate_with_dialogue(self, prompt, mutation_style):

        socrates = SocraticGPT(role="Socrates", model = self.model, tokenizer = self.tokenizer, mutation_style=mutation_style)
        theaetetus = SocraticGPT(role="Theaetetus",  model = self.model, tokenizer = self.tokenizer, mutation_style=mutation_style)

        initial_prompt = prompt

        socrates.set_problem(initial_prompt)
        theaetetus.set_problem(initial_prompt)

        for _ in range(socrates.n_round):

            socrates_response = socrates.get_response()
            # print(f"{socrates.role}: {socrates_response}")

            theaetetus_response = theaetetus.get_response()
            # print(f"{theaetetus.role}: {theaetetus_response}")

            if "final" in socrates_response.lower():
                break

        print(f"Generated solution with : {mutation_style}")
        final_prompt = socrates_response  
        mutated_prompt = re.findall(r'"([^"]*)"', final_prompt)
        final_prompt = mutated_prompt[1:]
        final_prompt = [prompt.replace('"',"'").replace("[","").replace("]","").replace("!",".") for prompt in final_prompt]

        final_prompt = random.sample(final_prompt, 1)
        print(f"Generated Prompt: {final_prompt[0]}")

        return final_prompt[0]


    def run_simulated_annealing(self, prompt_pool):
            '''
            This function runs the simulated annealing algorithm to optimize the prompt.
            '''

            initial_prompt = random.choice(prompt_pool)
            current_prompt = initial_prompt
            best_prompt = initial_prompt
            temperature = self.args.initial_temperature
            mut_styles = [
            "The mutation is a variant of the input prompt using unconventional thinking.",
            "The mutation is a variant of the input prompt that highlights strategic thought processes and logical reasoning.",
            "The mutation is a variant of the input prompt that introduces logic and makes it easier to understand.",
            "The mutation is a variant of the input prompt that focuses on logic and reasoning.",
            "The mutation is a variant of the input prompt that adds more details.",
            "The mutation is a variant of the input prompt that makes it more well-considered and logical.",
            ]
            stop_step = 0 
            end_algorithm = 0 

            while temperature > 10: #1.0:

                for _ in range(self.args.iterations_per_temperature):

                    if stop_step >= 2:
                        # print("Stop iterating...")
                        # break
                        print("Changing direction by selecting a new prompt from the prompt pool...")
                        current_prompt = random.choice(prompt_pool)
                        self.logger(f"Selecting another prompt: {current_prompt}")
                        end_algorithm += 1

                    if end_algorithm >= 2:
                        print("Ending algorithm...")
                        break

                    mutation_style = random.choice(mut_styles)
                    neighbor_prompt = self.mutate_with_dialogue(current_prompt, mutation_style)

                    current_score = self.evaluate(current_prompt)
                    neighbor_score = self.evaluate(neighbor_prompt)

                    print(f"Current Prompt: {current_prompt} with score {current_score}")
                    print(f"Generated Prompt: {neighbor_prompt} with score {neighbor_score}")

                    # Metropolis criterion
                    if neighbor_score > current_score or random.uniform(0, 1) < math.exp((neighbor_score - current_score) / temperature):
                        current_prompt = neighbor_prompt
                        print(f"Replaced current prompt with neighbor prompt since {neighbor_score} > {current_score}")
                        max_score = self.evaluate(best_prompt)
                        print(f"Best score so far is {max_score} with prompt {best_prompt}")
                        # Update the best prompt if needed
                        if neighbor_score > max_score:
                            best_prompt = neighbor_prompt
                            best_score = neighbor_score
                            print(f"Replaced best prompt with neighbor prompt since {neighbor_score} > {max_score}")

                    else:
                        stop_step += 1
                        print("Stop step increased: ", stop_step)


                self.logger.info("After {} iterations at temperature {}, the best prompt is: {} with score {}".format(self.args.iterations_per_temperature, temperature, best_prompt, best_score))

                temperature *= self.args.cooling_rate

                self.logger.info("Temperature is now {}".format(temperature))
                

            self.logger.info("The best prompt is {} with score {}".format(best_prompt, best_score))

            return best_prompt, best_score


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--initial_temperature", default=100, type=float, help="Initial temperature for simulated annealing")
    parser.add_argument("--cooling_rate", default=0.6, type=float, help="Cooling rate for simulated annealing")
    parser.add_argument("--iterations_per_temperature", default=5, type=int, help="Number of iterations per temperature for simulated annealing")
    parser.add_argument("--num_of_samples", default=20, type=int, help="Number of samples to evaluate the fitness on")
    args = parser.parse_args()

    csqa_dataset = load_dataset('tau/commonsense_qa', split='validation').select(range(args.num_of_samples))
    csqa = CSQA(csqa_dataset.map(process_choices))

    prompt_engine = SimAnneal(args, csqa)

    scores = {}
    prompt_pool = [
        "Let's think step by step",
        "Let's first understand the problem and devise a plan to solve the problem. Then, let's carry out the plan and solve the problem step by step" ,
        "Take a deep breath and work on this problem step-by-step",
        "Let's work this out in a step by step way to be sure we have the right answer",
        "Let's embrace a structured thought process, navigating through the question systematically",
        "Let's focus on logic and reasoning to arrive at a well-considered solution" ,
        "Take a step-by-step approach to this problem" ,
        "Let's embrace a structured thought process, navigating through the problem systematically",
    ]

    best_prompt, best_score = prompt_engine.run_simulated_annealing(prompt_pool)
    scores[best_prompt] = best_score

    # # Evaluate the prompts on the full dataset
    # for prompt in scores.keys():
    #     score = prompt_engine.evaluate(prompt, len(testset))
    #     scores[prompt] = score

    print(f"The optimal prompt is {max(scores, key = scores.get)} with a score of {scores[max(scores, key = scores.get)]}")