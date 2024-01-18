from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import random
import argparse
import logging
from utils import *
from prompts.math_prompts import *
from prompts.nli_prompts import *
from prompts.open_qa_prompts import *
import numpy as np
import torch
from conversation import SocraticGPT


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
    def __init__(self, args, trainset, testset, model, tokenizer, dataset):
        self.args = args
        self.trainset = trainset
        self.testset = testset
        self.dataset = dataset
        self.model = model
        self.tokenizer = tokenizer
        self.initial_population = self.initialise_population(self.args)

        
    def initialise_population(self, args):
        '''
        This function initialises the population of prompts.
        The current version suports only prompts that will be created with external tools.
        '''

        # if args.task == 'gsm8k':
        #     if args.type_of_prompts == 'short':
        #         initial_prompts = shorter_prompts
        #     elif args.type_of_prompts == 'normal':
        #         initial_prompts = mixed_prompts
        #     elif args.type_of_prompts == 'long':
        #         initial_prompts = long_prompts
        #     elif args.type_of_prompts == 'abstract':
        #         initial_prompts = abstract_prompts
        #     elif args.type_of_prompts == 'passive':
        #         initial_prompts = passive_voice_prompts
        #     else:
        #         initial_prompts = []
        #         for task in task_description:
        #             for style in thinking_styles:
        #                 prompt = f"{task}\n{style}."
        #                 initial_prompts.append(prompt)     
        
        # elif args.task == 'nli' or args.task == 'open_qa':
        #     initial_prompts = []
        #     for task in task_description:
        #         for style in thinking_styles:
        #             prompt = f"{task}.\n{style}."
        #             initial_prompts.append(prompt)

        initial_prompts = [
            "Let's think step by step",
            "Let's first understand the problem and devise a plan to solve the problem. Then, let's carry out the plan and solve the problem step by step",
            "Take a deep breath and work on this problem step-by-step",
            "Let's submerge ourselves in the conundrum, identify vital variables and their numerical values, and establish a plan. As we carry out the plan, let's scrutinize intermediate findings (ensure correct numerical calculations and logical reasoning), tackle the problem progressively, and unveil the answer",
            "Let's work this out in a step by step way to be sure we have the right answer",
            "Let's be very precise and accurate in our calculations"
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

        samples = self.dataset.shuffle(seed=self.args.seed).select(range(self.args.num_of_samples))

        for i,sample in enumerate(tqdm(samples)):

            question = sample['question']
            label = sample['label']

            # Construct the prompt
            if self.args.use_icl_examples:

                if self.args.use_contrastive_cot:
                    contrastive_prompt = construct_contrastive_icl_example(contrastive_samples, self.args.num_icl_examples)
                    model_input = f'''{contrastive_prompt}\n\nQuestion: {question}\n\n{prompt}'''
                else:
                    icl_prompt = construct_icl_examples(self.trainset, self.initial_population, self.args.num_icl_examples)
                    model_input = f'''{icl_prompt}\n\nQuestion: {question}\n\n{prompt}'''
          
            else:
                model_input = f'''Question: {question}\n\nAnswer: {prompt}\n\n'''

            input_ids = self.tokenizer(model_input, return_tensors="pt").input_ids.to('cuda')
            outputs = self.model.generate(
                input_ids,
                max_length=500,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            response_ids = outputs[0]
            text_output = self.tokenizer.decode(response_ids, skip_special_tokens=True)

            fitness += evaluate_GSM8K(text_output, label)

        fitness = fitness/self.args.num_of_samples

        return fitness


    def select_parents(self, fitness_dict):
        '''
        This function selects the parents of the next generation. 
        The current version supports only the roulette wheel selection method based on the fitness of the prompts.
        '''

        # parents_pool = []
        # fitness_weights = []

        # Pick two prompts from fitness dict based on their fitness 

        parents = random.choices(list(fitness_dict.keys()), weights = list(fitness_dict.values()), k = 2)

        # for prompt in fitness_dict.keys():
        #     parents_pool.append(prompt)
        #     fitness_weights.append(fitness_dict[prompt])

        # parents = random.choice(parents_pool, weights = fitness_weights, k = 2) # k = self.args.number_of_parents)

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

        # input_ids = self.tokenizer(crossover_prompt, return_tensors="pt").input_ids.to('cuda')
        # outputs = self.model.generate(
        #     input_ids,
        #     max_length=200,
        #     pad_token_id=self.tokenizer.pad_token_id,
        #     eos_token_id=self.tokenizer.eos_token_id,
        # )
        # response_ids = outputs[0]
        # child_prompt = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        # print(child_prompt)
        # # Find the text in the square brackets
        # pattern = r'\[([^\]]+)\]'
        # final_child_prompt = re.findall(pattern, child_prompt)
        # final_child_prompt = final_child_prompt[0]
        # print(f"Final child prompt: {final_child_prompt}")

        return final_prompt


    def mutate(self, child, population):
        '''
        This function mutates the child with probability 0.5. The population will definitely be mutated.
        The current version supports only the LLM mutation method.
        '''

        mutation_styles = ["Modify the prompt to make it more detailed",
                            "Improve the prompt by adding helpful advice",
                            "Change the wording of the prompt in an unexpected way",
                            "Modify the prompt to help an LLM follow the instructions",
                            "Generate a mutated version of the prompt by adding more details",
                            "Mutate the prompt to provide an alternative viewpoint",]

        new_prompts = []

        if random.random() > 0.5:
            print(f"Mutating the child {child}")
            # mutated_child = self.mutate_with_LLM(child) 
            mutated_child = self.mutate_with_dialogue(child, mutation_style = random.choice(mutation_styles))
            print(f"The mutated child prompt is {mutated_child}")
            new_prompts.append(mutated_child)

        if self.args.mutate_population:
            print("Mutating random prompts from the population")
            # random_prompts = random.sample(population, self.args.number_of_mutations)
            random_prompts = random.choice(population)
            print(f"The prompt that will be mutated is {random_prompts}")
            mutated_child = self.mutate_with_dialogue(random_prompts, mutation_style = random.choice(mutation_styles))
            new_prompts.append(mutated_child)
            print(f"The mutated prompt is {mutated_child}")

            # mutated_prompts = [self.mutate_with_LLM(prompt) for prompt in random_prompts]
            # mutated_prompts = [self.mutate_with_dialogue(prompt, mutation_style = mutation_style) for prompt in random_prompts]
            # new_prompts.extend(mutated_prompts)
        
        return new_prompts
    

    def mutate_with_LLM(self, prompt):
        '''
        This function mutates a prompt with the LLM method.
        '''
    
        if self.args.mutation_type == 'separate':

            thinking_styles = random.sample(thinking_styles, 5)
            task_descriptions = random.sample(task_description, 5)

            mutation_prompt = f'''
            I have an example text which consist of an INSTRUCTION and a TASK DESCRIPTION and how it is changed.
            text:
            INSTRUCTION: {thinking_styles[0]} TASK DESCRIPTION: {task_descriptions[0]}
            changed text:
            INSTRUCTION: {thinking_styles[1]} TASK DESCRIPTION: {task_descriptions[1]}
            text:
            INSTRUCTION: {thinking_styles[2]} TASK DESCRIPTION: {task_descriptions[2]}
            changed text:
            INSTRUCTION: {thinking_styles[3]} TASK DESCRIPTION: {task_descriptions[3]}

            {random.choice(mutation_prompts)}. Your answer should only be the new  INSTRUCTION and the new TASK DESCRIPTION:
            INSTRUCTION: {thinking_styles[4]} TASK DESCRIPTION: {task_descriptions[4]}
            '''

        elif self.args.mutation_type == 'mixed':

            mutation_prompt = f'''
            I have some texts with their mutated versions. A mutation is a change in the original text.
            text:
            {random.choice(shorter_prompts)}
            mutated text:
            {random.choice(abstract_prompts)}
            text:
            {random.choice(long_prompts)}
            changed text:
            {random.choice(shortest_prompts)}

            {random.choice(mutation_prompts)}. Your answer should only be the new mutated text:
            {prompt}
            '''   
            

        input_ids = self.helper_tokenizer(mutation_prompt, return_tensors="pt").input_ids.to('cuda')
        outputs = self.helper_model.generate(
            input_ids,
            pad_token_id=self.helper_tokenizer.pad_token_id,
            eos_token_id=self.helper_tokenizer.eos_token_id,
        )
        mutated_prompt = self.helper_tokenizer.decode(outputs[0], skip_special_tokens=True)

        return mutated_prompt
    

    def mutate_with_dialogue(self, prompt, mutation_style):

        mutation_styles = ["Modify the prompt to make it more detailed",
                            "Improve the prompt by adding helpful advice",
                            "Change the wording of the prompt in an unexpected way",
                            "Modify the prompt to help an LLM follow the instructions",
                            "Generate a mutated version of the prompt by adding more details",
                            "Mutate the prompt to provide an alternative viewpoint",]

        mutation_style = random.choice(mutation_styles)

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
    
    logger = setup_logger('progress_logger', 'EA_output.log')

    parser = argparse.ArgumentParser(description='Settings for the Evolutionary Algorithms')
    parser.add_argument('--task', default='gsm8k', type=str, help='Task to be solved. Choose one of: [gsm8k, nli, open_qa]')
    parser.add_argument('--type_of_prompts', default='short', type=str, help='Type of prompts for the initial population')
    parser.add_argument('--use_contrastive_cot', default=False, type=bool, help='whether to use contrastive cot in-context learning examples or not')
    parser.add_argument('--use_icl_examples', default=False, type=bool, help='whether to use in-context learning examples or not')
    parser.add_argument('--num_icl_examples', default=1, type=int, help='number of in-context learning examples used for evaluation')
    parser.add_argument('--num_of_samples', default=25, type=int, help='number of samples used for evaluation')
    parser.add_argument('--iterations', default=20, type=int, help='number of iterations for the EA')
    # parser.add_argument('--number_of_parents', default=2, type=int, help='number of parents to select')
    parser.add_argument('--number_of_mutations', default=1, type=int, help='number of mutations to perform')
    parser.add_argument('--seed', default=0, type=int, help='type of mutation')
    parser.add_argument('--mutate_population', default=False, type=bool, help='whether to mutate the population or not')
    args = parser.parse_args()
    
    # model = AutoModelForCausalLM.from_pretrained("microsoft/Orca-2-7b", device_map = 'auto', load_in_8bit = True)
    # tokenizer = AutoTokenizer.from_pretrained("microsoft/Orca-2-7b", use_fast = False)

    # Initializing model and dataset and engine

    tokenizer = AutoTokenizer.from_pretrained("berkeley-nest/Starling-LM-7B-alpha")
    model = AutoModelForCausalLM.from_pretrained("berkeley-nest/Starling-LM-7B-alpha", torch_dtype = torch.float16)
    model = model.to('cuda')

    original_test_dataset = load_dataset("gsm8k", 'main', split='test')
    original_train_dataset = load_dataset("gsm8k", 'main', split='train')

    testset = original_test_dataset.map(add_label)
    trainset = original_train_dataset.map(add_label)

    prompt_engine = GenPrompt(args, trainset, testset, model, tokenizer, testset)

    best_fitness = 0
    stagnation_count = 0
    patience = 5

    for iter in range(args.iterations):

        if iter == 0:
            logger.info(f"Evaluation of the initial population")
            # population = prompt_engine.initialise_population(args)
            population = [
            "Let's think step by step",
            "Let's first understand the problem and devise a plan to solve the problem. Then, let's carry out the plan and solve the problem step by step",
            "Take a deep breath and work on this problem step-by-step",
            "Let's submerge ourselves in the conundrum, identify vital variables and their numerical values, and establish a plan. As we carry out the plan, let's scrutinize intermediate findings (ensure correct numerical calculations and logical reasoning), tackle the problem progressively, and unveil the answer",
            "Let's work this out in a step by step way to be sure we have the right answer",
            "Let's be very precise and accurate in our calculations"
            ]
            print(f"The population is {population}")
            # fitness_dict = prompt_engine.evaluate_population(population)

            fitness_dict = {
                "Let's think step by step" : 0.2333,
                "Let's first understand the problem and devise a plan to solve the problem. Then, let's carry out the plan and solve the problem step by step" : 0.2,
                "Take a deep breath and work on this problem step-by-step" : 0.2333,
                "Let's submerge ourselves in the conundrum, identify vital variables and their numerical values, and establish a plan. As we carry out the plan, let's scrutinize intermediate findings (ensure correct numerical calculations and logical reasoning), tackle the problem progressively, and unveil the answer" : 0.4666,
                "Let's work this out in a step by step way to be sure we have the right answer": 0.3333,
                "Let's be very precise and accurate in our calculations": 0.3666,
                "Take a step-by-step approach to this problem" : 0.4666,
            }

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







