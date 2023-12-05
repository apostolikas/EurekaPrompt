from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
import random
import argparse
import logging
from utils import *
from prompts import *
import openai

FORMATTER = logging.Formatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S")

def setup_logger(name, log_file, level=logging.INFO):

    handler = logging.FileHandler(log_file)        
    handler.setFormatter(FORMATTER)
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

    def initialise_population(self, args):
        '''
        This function initialises the population of prompts.
        The current version suports only prompts that will be created with external tools.
        '''

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
            raise Exception('This type of prompts is not supported')    

        return initial_prompts


    def evaluate_population(self, population, model, tokenizer):
        '''
        This function evaluates the fitness of the populuation by calling 'evaluate_fitness' 
        for each prompt of the population.
        '''

        fitness_dict = {}

        for prompt in population:

            fitness_dict[prompt] = self.evaluate_fitness(prompt, model, tokenizer, args)

        return fitness_dict


    def evaluate_fitness(self, prompt, model, tokenizer, args):
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
            if args.use_icl_examples:
                icl_prompt = construct_icl_examples(self.trainset, self.initial_population, self.args.num_icl_examples)

            model_input = f'''{icl_prompt}
            Question: {question}
            {prompt}
            '''

            # Import qlora 

            input_ids = tokenizer(model_input, return_tensors="pt").input_ids.to("cuda")
            generated_ids = model.generate(input_ids, max_length=256)
            output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

            fitness += evaluate_GSM8K(output_text, label)

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


    def crossover(self, parents, model, tokenizer):
        '''
        This function performs crossover between the parents to create a child.
        For now it will only support only the LLM crossover method.
        '''

        parent_prompt1 = parents[0]
        parent_prompt2 = parents[1]

        prompt = f'''
        I have two parent prompts for an evolutionary algorithm.
        text:
        {parent_prompt1}
        text:
        {parent_prompt2}
        Write your new text that is the child of the crossover of the old ones and has a score as high as possible. Keep it short and concise and write only the new text in square brackets.
        '''

        response = openai.ChatCompletion.create(
                            engine="gpt35_8K_DSLS_16_6_2023",
                            messages = [ {
                                "role" : "user",
                                "content" : prompt
                                        }],
                            temperature=0.3,
                            max_tokens=800,
                            top_p=0.55,
                            frequency_penalty=0,
                            presence_penalty=0,
                            stop=None)
        
        child = response.choices[0]['text']

        return child


    def mutate(self, child, population, model, tokenizer):
        '''
        This function mutates the child with probability 0.5. The population will definitely be mutated.
        The current version supports only the LLM mutation method.
        '''

        new_prompts = []

        if random.random() > 0.5:
            mutated_child = self.mutate_with_LLM(child, model, tokenizer, args) 
            new_prompts.append(mutated_child)

        if self.args.mutate_population:

            random_prompts = random.choice(population, k = self.args.number_of_mutations)
            mutated_prompts = [self.mutate_with_LLM(prompt, model, tokenizer, args) for prompt in random_prompts]
            new_prompts.extend(mutated_prompts)
        
        return new_prompts
    

    def mutate_with_LLM(self, prompt, model, tokenizer, args):
        '''
        This function mutates a prompt with the LLM method.
        '''
    
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

        response = openai.ChatCompletion.create(
                            engine="gpt35_8K_DSLS_16_6_2023",
                            messages = [ {
                                "role" : "user",
                                "content" : prompt
                                        }],
                            temperature=0.3,
                            max_tokens=800,
                            top_p=0.55,
                            frequency_penalty=0,
                            presence_penalty=0,
                            stop=None)

        mutated_prompt = response.choices[0]['text']

        return mutated_prompt


if __name__ == "__main__":
    
    logger = setup_logger('progress_logger', 'output.log')

    parser = argparse.ArgumentParser(description='Settings for the Evolutionary Algorithms')
    parser.add_argument('--type_of_prompts', default='short', type=str, help='Type of prompts for the initial population')
    parser.add_argument('--use_icl', default=True, type=bool, help='whether to use in-context learning examples or not')
    parser.add_argument('--num_icl_examples', default=3, type=int, help='number of in-context learning examples used for evaluation')
    parser.add_argument('--num_of_samples', default=100, type=int, help='number of samples used for evaluation')
    parser.add_argument('--iterations', default=1000, type=int, help='number of iterations for the EA')
    parser.add_argument('--number_of_parents', default=2, type=int, help='number of parents to select')
    parser.add_argument('--number_of_mutations', default=2, type=int, help='number of mutations to perform')
    parser.add_argument('--mutate_population', action='store_true', help='mutate the population')
    args = parser.parse_args()
    

    # Load the model and tokenizer
    # tokenizer = AutoTokenizer.from_pretrained("microsoft/Orca-2-13b", device_map = "auto")
    # model = AutoModelForCausalLM.from_pretrained("microsoft/Orca-2-13b", use_fast = True)
    
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl", device_map = "auto")

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
            fitness_dict = prompt_engine.evaluate_population(population)
            for prompt in population:
                logger.info(f"Generation {iter}: {prompt} with fitness {fitness_dict[prompt]}")
            logger.info(f"Genetic Algorithms starts")

        else:
            
            parents = prompt_engine.select_parents(fitness_dict)
            children = prompt_engine.crossover(parents, model, tokenizer)
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







