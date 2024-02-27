from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import random
import argparse
import logging
from utils import *
import torch
from conversation import SocraticGPT
from torch.utils.data import Dataset, DataLoader

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


class SocraticGPT_for_crossover:
    def __init__(self, role, model, tokenizer, n_round=1):
        self.role = role
        self.tokenizer = tokenizer
        self.model = model
        self.n_round = n_round
        self.other_role = "Theaetetus" if role == "Socrates" else "Socrates"
        self.history = []

    def set_problem(self, parent_prompt1, parent_prompt2):
        self.history.append({
            "role": "system",
            "content": f"{self.role} and {self.other_role} are two AI assistants for Tony to perform a crossover of two prompts. The crossover is the result of the combination of two parent prompts. The parent prompts are: \"[{parent_prompt1}]\" and \"[{parent_prompt2}]\".\n\n{self.role} and {self.other_role} will engage in multi-round dialogue to perform the crossover for an evolutionary algorithm. The final child prompt has to be within brackets."
        })
        self.history.append({
            "role": "assistant",
            "content": f"Hi {self.other_role}, let's work together to perform a crossover of two prompts. The crossover is the result of the combination of two parent prompts. Both of us can suggest improvements. The child prompt has to be short and concise. The child prompt has to be within brackets."
        })

    def get_response(self):
        input_prompt = "".join([f"{msg['role']}: {msg['content']}" for msg in self.history])
        input_ids = self.tokenizer(input_prompt, return_tensors="pt").input_ids.to('cuda')
        outputs = self.model.generate(
            input_ids,
            max_length=1000,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        response_ids = outputs[0]
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        self.history.append({
            "role": "assistant",
            "content": response_text
        })
        return response_text

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


class GenPrompt:
    def __init__(self, args, testset, model, tokenizer):
        self.args = args
        self.testset = testset
        self.model = model
        self.tokenizer = tokenizer
        self.initial_population = self.initialise_population()

        
    def initialise_population(self):
        '''
        This function initialises the population of prompts.
        The current version suports only prompts that will be created with external tools.
        '''

        initial_prompts = [
            "Let's think step by step",
            "Let's first understand the problem and devise a plan to solve the problem. Then, let's carry out the plan and solve the problem step by step",
            "Take a deep breath and work on this problem step-by-step",
            "Let's work this out in a step by step way to be sure we have the right answer",
            "Let's focus on logic and reasoning to arrive at a well-considered solution",
            "Our approach will be to methodically work through the problem, ensuring accuracy at each step to derive the correct answer",
        ]

        return initial_prompts


    def evaluate_population(self, population):
        '''
        This function evaluates the fitness of the populuation by calling 'evaluate_fitness' 
        for each prompt of the population.
        '''

        fitness_dict = {}

        for prompt in population:
            fitness_dict[prompt] = self.evaluate_fitness(prompt)
            print(f"Evaluated prompt: {prompt} with fitness {fitness_dict[prompt]}")
        return fitness_dict


    def evaluate_fitness(self, prompt):
        '''
        This function evaluates the fitness of a prompt by calculating the accuracy of the model on the dataset.
        The current version only supports GSM8K and checks whether the final number in the output is the same as the label.
        '''

        # Calculate the accuracy
        fitness = 0

        testloader = DataLoader(self.testset, batch_size=1, shuffle=True, num_workers=0)

        for i,sample in enumerate(tqdm(testloader)):

            question, choices, answer_key = sample

            model_input = f"Question: {question[0]}\n\nChoices: {choices[0]}\n\nAnswer: {prompt}"
            single_turn_input = f"GPT4 Correct User: {model_input} <|end_of_turn|>GPT 4 Correct Assistant:"
            text_output = generate_response(model, tokenizer, single_turn_input)
            result = evaluate_csqa(text_output, answer_key[0])
            fitness += result

        fitness = fitness/self.args.num_of_samples

        return fitness


    def select_parents(self, fitness_dict):
        '''
        This function selects the parents of the next generation. 
        The current version supports only the roulette wheel selection method based on the fitness of the prompts.
        '''

        parents = random.choices(list(fitness_dict.keys()), weights = list(fitness_dict.values()), k = 2)

        return parents


    def crossover(self, parents):
        '''
        This function performs crossover between the parents to create a child.
        For now it will only support only the LLM crossover method.
        '''

        parent_prompt1 = parents[0]
        parent_prompt2 = parents[1]

        # crossover_prompt = f'''
        # I have two parent prompts for an evolutionary algorithm.
        # text:
        # {parent_prompt1}
        # text:
        # {parent_prompt2}
        # Write your new text that is the child of the crossover of the old ones. Keep it short and concise and write only the new text in square brackets.
        # '''

        socrates = SocraticGPT_for_crossover(role="Socrates", model = self.model, tokenizer = self.tokenizer)
        theaetetus = SocraticGPT_for_crossover(role="Theaetetus",  model = self.model, tokenizer = self.tokenizer)

        socrates.set_problem(parent_prompt1, parent_prompt2)
        theaetetus.set_problem(parent_prompt1, parent_prompt2)

        for _ in range(socrates.n_round):
            socrates_response = socrates.get_response()
            if "final" in socrates_response.lower():
                break
            theaetetus_response = theaetetus.get_response()

        final_prompt = socrates_response  
        final_prompt = re.findall(r'\[(.*?)\]', final_prompt)[-1]    
        print(f"Final Child prompt: {final_prompt}")

        return final_prompt


    def mutate(self, child, population):
        '''
        This function mutates the child with probability 0.5. The population will definitely be mutated.
        The current version supports only the LLM mutation method.
        '''

        # mutation_styles = ["Modify the prompt to make it more detailed",
        #                     "Improve the prompt by adding helpful advice",
        #                     "Change the wording of the prompt in an unexpected way",
        #                     "Modify the prompt to help an LLM follow the instructions",
        #                     "Generate a mutated version of the prompt by adding more details",
        #                     "Mutate the prompt to provide an alternative viewpoint",]

        mutation_styles = [
            "The mutation is a variant of the input prompt that introduces a structured thought process.",
            "The mutation is a variant of the input prompt that highlights strategic thought processes and logical reasoning.",
            "The mutation is a variant of the input prompt that introduces logic and makes it easier to understand.",
            "The mutation is a variant of the input prompt that focuses on logic and reasoning.",
            "The mutation is a variant of the input prompt that adds more details.",
            "The mutation is a variant of the input prompt that makes it more well-considered and logical.",
            ]


        new_prompts = []

        if random.random() > 0.5:
            print(f"Mutating the child {child}")
            mutated_child = self.mutate_with_dialogue(child, mutation_style = random.choice(mutation_styles))
            print(f"The mutated child prompt is {mutated_child}")
            new_prompts.append(mutated_child)

        if self.args.mutate_population:
            print("Mutating random prompts from the population")
            random_prompts = random.choice(population)
            print(f"The prompt that will be mutated is {random_prompts}")
            mutated_child = self.mutate_with_dialogue(random_prompts, mutation_style = random.choice(mutation_styles))
            new_prompts.append(mutated_child)
            print(f"The mutated prompt is {mutated_child}")

            # mutated_prompts = [self.mutate_with_LLM(prompt) for prompt in random_prompts]
            # mutated_prompts = [self.mutate_with_dialogue(prompt, mutation_style = mutation_style) for prompt in random_prompts]
            # new_prompts.extend(mutated_prompts)
        
        return new_prompts
        
    def mutate_with_dialogue(self, prompt, mutation_style):

        socrates = SocraticGPT(role="Socrates", model = self.model, tokenizer = self.tokenizer, mutation_style=mutation_style)
        theaetetus = SocraticGPT(role="Theaetetus",  model = self.model, tokenizer = self.tokenizer, mutation_style=mutation_style)

        initial_prompt = prompt

        socrates.set_problem(initial_prompt)
        theaetetus.set_problem(initial_prompt)

        for _ in range(socrates.n_round):

            socrates_response = socrates.get_response()
            # print(f"{socrates.role}: {socrates_response}")

            if "final" in socrates_response.lower():
                break

            theaetetus_response = theaetetus.get_response()
            # print(f"{theaetetus.role}: {theaetetus_response}")

        print(f"Generated solution with : {mutation_style}")
        final_prompt = socrates_response  
        final_prompt = re.findall(r'\[(.*?)\]', final_prompt)[-1]    
        print(f"Final Mutated prompt: {final_prompt}")
        return final_prompt


if __name__ == "__main__":
    
    logger = setup_logger('progress_logger', 'CSQA_EvolAlg_output.log')

    parser = argparse.ArgumentParser(description='Settings for the Evolutionary Algorithms')
    parser.add_argument('--use_icl_examples', default=False, type=bool, help='whether to use in-context learning examples or not')
    parser.add_argument('--num_icl_examples', default=1, type=int, help='number of in-context learning examples used for evaluation')
    parser.add_argument('--num_of_samples', default=30, type=int, help='number of samples used for evaluation')
    parser.add_argument('--iterations', default=20, type=int, help='number of iterations for the EA')
    parser.add_argument('--number_of_mutations', default=1, type=int, help='number of mutations to perform')
    parser.add_argument('--mutate_population', default=False, type=bool, help='whether to mutate the population or not')
    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained("berkeley-nest/Starling-LM-7B-alpha")
    model = AutoModelForCausalLM.from_pretrained("berkeley-nest/Starling-LM-7B-alpha", torch_dtype = torch.float16)
    model = model.to('cuda')

    csqa_dataset = load_dataset('tau/commonsense_qa', split='validation').select(range(args.num_of_samples))
    csqa = CSQA(csqa_dataset.map(process_choices))

    prompt_engine = GenPrompt(args, csqa, model, tokenizer)

    best_fitness = 0
    stagnation_count = 0
    patience = 10

    for iter in range(args.iterations):

        if iter == 0:
            logger.info(f"Evaluation of the initial population")
            population = [
                "Let's think step by step",
                "Let's first understand the problem and devise a plan to solve the problem. Then, let's carry out the plan and solve the problem step by step" ,
                "Take a deep breath and work on this problem step-by-step",
                "Let's work this out in a step by step way to be sure we have the right answer",
                # "Let's embrace a structured thought process, navigating through the question systematically",
                # "Let's focus on logic and reasoning to arrive at a well-considered solution" ,
                # "Take a step-by-step approach to this problem" ,
            ]
            print(f"The population is {population}")

            # fitness_dict = {
            #     "Let's think step by step" : 0.8571,
            #     "Let's first understand the problem and devise a plan to solve the problem. Then, let's carry out the plan and solve the problem step by step" : 0.7714,
            #     "Take a deep breath and work on this problem step-by-step" : 0.7142,
            #     "Let's work this out in a step by step way to be sure we have the right answer": 0.7428,
            #     "Let's embrace a structured thought process, navigating through the question systematically": 0.7142,
            #     # "Let's focus on logic and reasoning to arrive at a well-considered solution": 0.9142,
            #     "Take a step-by-step approach to this problem" : 0.7142,
            #     "Let's embrace a structured thought process, navigating through the problem systematically": 0.7714,
            # }

            fitness_dict = prompt_engine.evaluate_population(population)

            for prompt in population:
                logger.info(f"Generation {iter}: {prompt} with fitness {fitness_dict[prompt]}")
            logger.info(f"Genetic Algorithms starts")

        else:
            
            parents = prompt_engine.select_parents(fitness_dict)
            parents = random.sample(list(fitness_dict.keys()), 2)
            children = prompt_engine.crossover(parents)
            fitness_dict.update(prompt_engine.evaluate_population([children]))
            population.extend(children)
            logger.info(f"Generation {iter} Crossover: '{children}' with fitness {fitness_dict[children]}")
            print("Crossover done.")
            new_prompts = prompt_engine.mutate(children, population)
            fitness_dict.update(prompt_engine.evaluate_population(new_prompts))
            for prompt in new_prompts:
                logger.info(f"Generation {iter} Mutation: [{prompt}] with fitness {fitness_dict[prompt]}")

            population.extend(new_prompts)
            best_prompt = max(fitness_dict, key = fitness_dict.get)
            current_best_fitness = fitness_dict[best_prompt]

            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_prompt = best_prompt
                stagnation_count = 0
            else:
                stagnation_count += 1

            logger.info(f"Generation {iter}: Best prompt: {best_prompt} with fitness {best_fitness}")

            if stagnation_count >= patience:
                logger.info(f"Converged at generation {iter} with best prompt: {best_prompt} with fitness {best_fitness}")
                break







