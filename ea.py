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
import faiss


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
    def __init__(self, args, trainset, testset, model, tokenizer):
        self.args = args
        self.trainset = trainset
        self.testset = testset
        self.model = model
        self.tokenizer = tokenizer
        self.initial_population = self.initialise_population(self.args)
        self.helper_model = AutoModelForCausalLM.from_pretrained("TheBloke/OpenHermes-2.5-Mistral-7B-GGUF", model_file="./openhermes-2.5-mistral-7b.Q4_K_M.gguf", model_type="mistral")

    def initialise_population(self, args):
        '''
        This function initialises the population of prompts.
        The current version suports only prompts that will be created with external tools.
        '''

        if args.task == 'gsm8k':
            if args.type_of_prompts == 'short':
                initial_prompts = shorter_prompts
            elif args.type_of_prompts == 'normal':
                initial_prompts = mixed_prompts
            elif args.type_of_prompts == 'long':
                initial_prompts = long_prompts
            elif args.type_of_prompts == 'abstract':
                initial_prompts = abstract_prompts
            elif args.type_of_prompts == 'passive':
                initial_prompts = passive_voice_prompts
            else:
                initial_prompts = []
                for task in task_description:
                    for style in thinking_styles:
                        prompt = f"{task}\n{style}."
                        initial_prompts.append(prompt)     
        
        elif args.task == 'nli' or args.task == 'open_qa':
            initial_prompts = []
            for task in task_description:
                for style in thinking_styles:
                    prompt = f"{task}.\n{style}."
                    initial_prompts.append(prompt)

        return initial_prompts


    def evaluate_population(self, population):
        '''
        This function evaluates the fitness of the populuation by calling 'evaluate_fitness' 
        for each prompt of the population.
        '''

        fitness_dict = {}

        for prompt in population:

            fitness_dict[prompt] = self.evaluate_fitness(prompt)

        return fitness_dict


    def evaluate_fitness(self, prompt):
        '''
        This function evaluates the fitness of a prompt by calculating the accuracy of the model on the dataset.
        The current version only supports GSM8K and checks whether the final number in the output is the same as the label.
        '''

        # Calculate the accuracy
        fitness = 0
        num_of_samples = 100 

        samples = self.dataset.shuffle(seed=42).select(range(num_of_samples))

        for i,sample in enumerate(tqdm(samples)):

            question = sample['question']
            label = sample['label']

            # Construct the prompt
            if self.args.use_icl_examples:
                icl_prompt = construct_icl_examples(self.trainset, self.initial_population, self.args.num_icl_examples)
                model_input = f'''{icl_prompt}
                Question: {question}
                {prompt}
                '''
            
            else:
                model_input = f'''Question: {question}
                {prompt}
                '''

            system_message = "You are Orca, an AI language model created by Microsoft. You are a cautious assistant. You carefully follow instructions in order to solve math problems."

            prompt = f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{model_input}<|im_end|>\n<|im_start|>assistant"

            input_ids = self.tokenizer(prompt, return_tensors='pt').input_ids.to("cuda")

            output_ids = self.model.generate(input_ids)
            text_output = self.tokenizer.batch_decode(output_ids)[0]

            fitness += evaluate_GSM8K(text_output, label)

        fitness = fitness/num_of_samples

        return fitness


    def select_parents(self, fitness_dict):
        '''
        This function selects the parents of the next generation. 
        The current version supports only the roulette wheel selection method based on the fitness of the prompts.
        '''

        parents_pool = []
        fitness_weights = []

        for prompt in fitness_dict.keys():
            parents_pool.append(prompt)
            fitness_weights.append(fitness_dict[prompt])

        parents = random.choice(parents_pool, weights = fitness_weights, k = self.args.number_of_parents)

        return parents


    def crossover(self, parents):
        '''
        This function performs crossover between the parents to create a child.
        For now it will only support only the LLM crossover method.
        '''

        parent_prompt1 = parents[0]
        parent_prompt2 = parents[1]

        crossover_prompt = f'''
        I have two parent prompts for an evolutionary algorithm.
        text:
        {parent_prompt1}
        text:
        {parent_prompt2}
        Write your new text that is the child of the crossover of the old ones and has a score as high as possible. Keep it short and concise and write only the new text in square brackets.
        '''

        child_prompt = self.helper_model(crossover_prompt)

        return child_prompt


    def mutate(self, child, population):
        '''
        This function mutates the child with probability 0.5. The population will definitely be mutated.
        The current version supports only the LLM mutation method.
        '''

        new_prompts = []

        if random.random() > 0.5:
            mutated_child = self.mutate_with_LLM(child) 
            new_prompts.append(mutated_child)

        if self.args.mutate_population:

            random_prompts = random.choice(population, k = self.args.number_of_mutations)
            mutated_prompts = [self.mutate_with_LLM(prompt) for prompt in random_prompts]
            new_prompts.extend(mutated_prompts)
        
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
            

        mutated_prompt = self.helper_model(mutation_prompt)

        return mutated_prompt
    

    def build_prompt_index(self, prompts, tokenizer):
        '''
        This function builds a faiss index with the prompts of the initial population.
        '''

        prompt_embeddings = [tokenizer(prompt, return_tensors="pt")["input_ids"] for prompt in prompts]

        prompt_embeddings = np.array(prompt_embeddings)

        index = faiss.IndexFlatL2(prompt_embeddings[0].shape[1])
        index.add(prompt_embeddings)

        return index


if __name__ == "__main__":
    
    logger = setup_logger('progress_logger', 'output.log')

    parser = argparse.ArgumentParser(description='Settings for the Evolutionary Algorithms')
    parser.add_argument('--task', default='gsm8k', type=str, help='Task to be solved. Choose one of: [gsm8k, nli, open_qa]')
    parser.add_argument('--type_of_prompts', default='short', type=str, help='Type of prompts for the initial population')
    parser.add_argument('--use_icl', default=True, type=bool, help='whether to use in-context learning examples or not')
    parser.add_argument('--num_icl_examples', default=3, type=int, help='number of in-context learning examples used for evaluation')
    parser.add_argument('--num_of_samples', default=100, type=int, help='number of samples used for evaluation')
    parser.add_argument('--iterations', default=1000, type=int, help='number of iterations for the EA')
    parser.add_argument('--number_of_parents', default=2, type=int, help='number of parents to select')
    parser.add_argument('--number_of_mutations', default=2, type=int, help='number of mutations to perform')
    parser.add_argument('--mutate_population', action='store_true', help='mutate the population')
    args = parser.parse_args()
    

    model = AutoModelForCausalLM.from_pretrained("microsoft/Orca-2-7b", device_map = 'auto', load_in_8bit = True)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Orca-2-7b", use_fast = False)

    #TODO add the other datasets
    original_test_dataset = load_dataset("gsm8k", 'main', split='test')
    original_train_dataset = load_dataset("gsm8k", 'main', split='train')

    testset = original_test_dataset.map(add_label)
    trainset = original_train_dataset.map(add_label)

    prompt_engine = GenPrompt(args, trainset, testset, model, tokenizer)

    best_fitness = 0
    stagnation_count = 0
    patience = 5

    for iter in range(args.iterations):

        if iter == 0:
            logger.info(f"Evaluation of the initial population")
            population = prompt_engine.initialise_population(args)
            prompt_index = prompt_engine.build_prompt_index(population, tokenizer)
            fitness_dict = prompt_engine.evaluate_population(population)

            for prompt in population:
                logger.info(f"Generation {iter}: {prompt} with fitness {fitness_dict[prompt]}")
            logger.info(f"Genetic Algorithms starts")

        else:
            
            parents = prompt_engine.select_parents(fitness_dict)
            children = prompt_engine.crossover(parents, model, tokenizer)

            distances, indices = prompt_index.search(tokenizer(children, return_tensors="pt")["input_ids"], 2) 
            logger.info(f"Generation {iter}: {children} with fitness {fitness_dict[children]}")
            fitness_dict.update(prompt_engine.evaluate_population(children))
            population.extend(children)
            new_prompts = prompt_engine.mutate(children)
            fitness_dict.update(prompt_engine.evaluate_population(new_prompts))

            for prompt in new_prompts:
                logger.info(f"Generation {iter}: {prompt} with fitness {fitness_dict[prompt]}")

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







