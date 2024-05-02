from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import random
import argparse
import logging
import re 
from my_utils import *
from prompts import *
import torch
from transformers.utils import logging
logging.get_logger("transformers").setLevel(logging.ERROR)


class GenPrompt:
    def __init__(self, args, testset, model, tokenizer, trainset):
        self.args = args
        self.testset = testset
        self.trainset = trainset
        self.model = model
        self.tokenizer = tokenizer
        self.initial_population = self.initialise_population()
        self.helper_model = model
        
    def initialise_population(self):

        if self.args.task == 'gsm8k':
            initial_prompts = gsm8k_initial_prompts
        elif self.args.task == 'svamp':
            initial_prompts = svamp_initial_prompts
        elif self.args.task == 'strategyqa':
            initial_prompts = strategyqa_initial_prompts
        elif self.args.task == 'csqa':
            initial_prompts = csqa_initial_prompts
        elif self.args.task in bb_tasks:
            initial_prompts = bb_initial_prompts
        else:
            raise ValueError("Task not supported")

        return initial_prompts


    def evaluate_population(self, population):

        fitness_dict = {}
        for prompt in population:
            fitness_dict[prompt] = self.evaluate_fitness(prompt)
            print(f"Evaluated prompt: {prompt} | Fitness: {fitness_dict[prompt]}")
        return fitness_dict


    def evaluate_fitness(self, prompt):

        fitness = 0

        if self.args.task == 'gsm8k':
            random.seed(self.args.seed)
            samples = random.sample(self.testset, self.args.num_of_samples)

            for sample in tqdm(samples):
                question = sample['question']
                label = sample['label']

                if self.args.use_icl_examples:
                    # include contrastive cot examples
                    icl_prompt = construct_icl_examples_gsm8k(self.trainset, self.initial_population, self.args.num_icl_examples, prompt)
                    model_input = f'''{icl_prompt}Question: {question}\nAnswer: {prompt}'''
                else:
                    model_input = f'''Question: {question}\nAnswer: {prompt}'''
                input_prompt = f'''GPT4 Correct User: {model_input}<|end_of_turn|>GPT4 Correct Assistant:'''
                text_output = self.model.get_response(input_prompt)
                fitness += evaluate_GSM8K(text_output, label)

        elif self.args.task == 'csqa':
            
            random.seed(self.args.seed)
            samples = random.sample(self.testset, self.args.num_of_samples)

            for sample in tqdm(samples):
                question = sample['question']['stem']
                choices = sample['choice_answers']
                label = sample['answerKey']

                if self.args.use_icl_examples:
                    # include contrastive cot examples
                    icl_prompt = construct_icl_examples_csqa(self.trainset, self.initial_population, self.args.num_icl_examples, prompt)
                    model_input = f'''{icl_prompt}Question: {question}\nAnswer Choices: {choices}\nAnswer: {prompt}'''
                else:
                    model_input = f'''Question: {question}\nAnswer Choices: {choices}\nAnswer: {prompt}'''
                input_prompt = f'''GPT4 Correct User: {model_input}<|end_of_turn|>GPT4 Correct Assistant:'''
                text_output = self.model.get_response(input_prompt)
                text_output = text_output.split('GPT4 Correct Assistant:')[1]
                fitness += evaluate_CSQA(text_output, label)


        elif self.args.task == 'strategyqa':
            random.seed(self.args.seed)
            samples = random.sample(self.testset, self.args.num_of_samples)

            for sample in tqdm(samples):
                question = sample['question']
                label = sample['answer']

                if self.args.use_icl_examples:
                    # include contrastive cot examples
                    icl_prompt = construct_icl_examples_strategyqa(self.trainset, self.initial_population, self.args.num_icl_examples, prompt)
                    model_input = f'''{icl_prompt}Question: Yes or no: {question}\nAnswer: {prompt}'''
                else:
                    model_input = f'''Question: Yes or no: {question}\nAnswer: {prompt}'''
                input_prompt = f'''GPT4 Correct User: {model_input}<|end_of_turn|>GPT4 Correct Assistant:'''
                text_output = self.model.get_response(input_prompt)
                text_output = text_output.split('GPT4 Correct Assistant:')[1]
                fitness += evaluate_StrategyQA(text_output, label)


        elif self.args.task == 'svamp':
            random.seed(self.args.seed)
            samples = random.sample(self.testset, self.args.num_of_samples)

            for sample in tqdm(samples):
                question = sample['full_question']
                label = sample['Answer']

                if self.args.use_icl_examples:
                    # include contrastive cot examples
                    icl_prompt = construct_icl_examples_svamp(self.trainset, self.initial_population, self.args.num_icl_examples, prompt)
                    model_input = f'''{icl_prompt}Question: {question}\nAnswer: {prompt}'''
                else:
                    model_input = f'''Question: {question}\nAnswer: {prompt}'''
                input_prompt = f'''GPT4 Correct User: {model_input}<|end_of_turn|>GPT4 Correct Assistant:'''
                text_output = self.model.get_response(input_prompt)
                text_output = text_output.split('GPT4 Correct Assistant:')[1]
                fitness += evaluate_SVAMP(text_output, label)

        elif self.args.task == 'aqua':
            random.seed(self.args.seed)
            samples = random.sample(self.testset, self.args.num_of_samples)

            for sample in tqdm(samples):
                question = sample['question']
                label = sample['correct']
                choices = sample['answer_choices']

                if self.args.use_icl_examples:
                    # include contrastive cot examples
                    icl_prompt = construct_icl_examples_aqua(self.trainset, self.initial_population, self.args.num_icl_examples, prompt)
                    model_input = f'''{icl_prompt}Question: {question}\nAnswer Choices: {choices}\nAnswer: {prompt}'''
                else:
                    model_input = f'''Question: {question}\nAnswer Choices: {choices}\nAnswer: {prompt}'''
                input_prompt = f'''GPT4 Correct User: {model_input}<|end_of_turn|>GPT4 Correct Assistant:'''
                text_output = self.model.get_response(input_prompt)
                text_output = text_output.split('GPT4 Correct Assistant:')[1]
                fitness += evaluate_CSQA(text_output, label)
        
        elif self.args.task == 'abs_nar':

            random.seed(self.args.seed)
            samples = random.sample(self.testset, self.args.num_of_samples)

            for sample in tqdm(samples):
                narrative = sample['input']
                label = sample['label']
                answer_choices = sample['answer_choices']
                model_input = f'''Question: Can you choose the most related proverb from the list of 5 proverbs given a narrative?\nNarrative: {narrative}\nAnswer choices: {answer_choices}\nAnswer: {prompt}'''
                input_prompt = f'''GPT4 Correct User: {model_input}<|end_of_turn|>GPT4 Correct Assistant:'''
                text_output = self.model.get_response(input_prompt)
                text_output = text_output.split('GPT4 Correct Assistant:')[1]
                fitness += evaluate_CSQA(text_output, label)

        elif self.args.task == 'disamb':

            random.seed(self.args.seed)
            samples = random.sample(self.testset, self.args.num_of_samples)
            question = 'Can you claritfy the meaning of the sentence with ambiguous pronouns?'

            for sample in tqdm(samples):
                context = sample['input']
                label = sample['label']
                answer_choices = sample['answer_choices']
                model_input = f'''Question: {question}\nSentence: {context}\nAnswer choices: {answer_choices}\nAnswer: {prompt}'''
                input_prompt = f'''GPT4 Correct User: {model_input}<|end_of_turn|>GPT4 Correct Assistant:'''
                text_output = self.model.get_response(input_prompt)
                text_output = text_output.split('GPT4 Correct Assistant:')[1]
                fitness += evaluate_CSQA(text_output, label)

        elif self.args.task == 'logic_ded3':

            random.seed(self.args.seed)
            samples = random.sample(self.testset, self.args.num_of_samples)
            question = 'What is the correct answer based on the context?'

            for sample in tqdm(samples):
                context = sample['input']
                label = sample['label']
                answer_choices = sample['answer_choices']
                model_input = f'''Question: {question}\nContext: {context}\nAnswer choices: {answer_choices}\nAnswer: {prompt}'''
                input_prompt = f'''GPT4 Correct User: {model_input}<|end_of_turn|>GPT4 Correct Assistant:'''
                text_output = self.model.get_response(input_prompt)
                text_output = text_output.split('GPT4 Correct Assistant:')[1]
                fitness += evaluate_CSQA(text_output, label)

        elif self.args.task == 'social_iqa' or self.args.task == 'sports_und' or self.args.task == 'date_under' or self.args.task == 'causal_judg':

            random.seed(self.args.seed)
            samples = random.sample(self.testset, self.args.num_of_samples)

            for sample in tqdm(samples):
                question = sample['input']
                label = sample['label']
                answer_choices = sample['answer_choices']
                model_input = f'''Question: {question}\nAnswer choices: {answer_choices}\nAnswer: {prompt}'''
                input_prompt = f'''GPT4 Correct User: {model_input}<|end_of_turn|>GPT4 Correct Assistant:'''
                text_output = self.model.get_response(input_prompt)
                text_output = text_output.split('GPT4 Correct Assistant:')[1]
                fitness += evaluate_CSQA(text_output, label)

        fitness = fitness/self.args.num_of_samples

        return fitness


    def select_parents(self, fitness_dict):
        random.seed() # delete
        # return random.sample(list(fitness_dict.keys()), 2)
        prompts = list(fitness_dict.keys())
        scores = list(fitness_dict.values())
        # Normalize the scores to create probabilities
        total_score = sum(scores)
        probabilities = [score / total_score for score in scores]
        # Use random.sample() to select unique prompts based on the probabilities
        population = list(zip(prompts, probabilities))
        samples = random.sample(population, 2)
        # Extract the sampled prompts
        sampled_prompts = [prompt for prompt, _ in samples]

        return sampled_prompts
    
    def crossover(self, parents):

        parent_prompt1 = parents[0]
        parent_prompt2 = parents[1]
        print(f"The parents are: {[parent_prompt1, parent_prompt2]}")
        final_prompt = crossover_dialogue(self.helper_model, parent_prompt1, parent_prompt2)
        final_prompt = final_prompt[-1]
        print(f'The child prompt is: "{final_prompt}"')
        LOGGER.info(f'Parent 1: "{parent_prompt1}"')
        LOGGER.info(f'Parent 2: "{parent_prompt2}"')
        LOGGER.info(f'Child: "{final_prompt}"')

        return final_prompt


    def mutate(self, child, population, fitness_dict, mutation_styles):

        # if self.args.task == 'csqa':
        #     mutation_styles = csqa_mutation_styles
        # elif self.args.task == 'strategyqa':
        #     mutation_styles = strategyqa_mutation_styles
        # elif self.args.task == 'gsm8k':
        #     mutation_styles = gsm8k_mutation_styles
        # elif self.args.task == 'svamp':
        #     mutation_styles = svamp_mutation_styles
        # elif self.args.task in bb_tasks:
        #     mutation_styles = bb_mutation_styles
        # else:
        #     raise ValueError("Task not supported")

        new_prompts = []

        if random.random() > 0.5:
            random.seed() # delete
            mutation_style_normal = random.choice(mutation_styles)
            print(f"Mutating the child with mutation style with index: {mutation_styles.index(mutation_style_normal)}")
            #mutated_child = mutation_dialogue(self.helper_model, mutation_style, child)
            mutated_child = mutation_dialogue(self.helper_model, mutation_style_normal, child, False, None)
            if isinstance(mutated_child, list):
                mutated_child = mutated_child[-1]
            print(f"The mutated child prompt is a {type(mutated_child)}: {mutated_child}")
            new_prompts.append(mutated_child)
            LOGGER.info(f'Mutating the child with mutation style: "{mutation_style_normal}"')
            LOGGER.info(f'The mutated child prompt is: "{mutated_child}"')
        else:
            mutation_style_normal = None
            


        if self.args.mutate_population:
            random.seed() # delete
            print("Mutating random prompts from the population")
            mutation_style = random.choice(mutation_styles)
            random_prompt = random.choice(population)
            print(f"The prompt that will be mutated is: {random_prompt}")
            use_words = True #if random.random() > 0.3 else False

            if use_words == True:

                mut_prompt = contstruct_mutation_prompt(fitness_dict)
                print(mut_prompt)
                model_input = f"GPT4 Correct System: You are a powerful AI assistant that will help me identify the crucial parts in some text.<|end_of_turn|>GPT4 Correct User: {mut_prompt}<|end_of_turn|>GPT4 Correct Assistant:"

                response_text = self.helper_model.get_response(model_input)
                
                match = re.search(r'\[(.*?)\]', response_text)
                if match:
                    list_string = match.group(1)
                    final_list = [item.strip() for item in list_string.split(",")]
                    # words_to_use = random.choice(final_list)
                    words_to_use = random.sample(final_list, 2)
                    print(f"Words that will be used for the mutation: {words_to_use}")
                else:
                    words_to_use = "step by step"
            else:
                words_to_use = None
            
            LOGGER.info(f'The random prompt that will go under guided mutation is: "{random_prompt}"')
            LOGGER.info(f'Mutation style: "{mutation_style}"')
            LOGGER.info(f'Words to use: "{words_to_use}"')

            # new_mutated_child = mutation_dialogue(self.helper_model, mutation_style, random_prompt)
            new_mutated_child = mutation_dialogue(self.helper_model, mutation_style, random_prompt, use_words, words_to_use)

            if isinstance(new_mutated_child, list):
                new_mutated_child = new_mutated_child[-1]

            new_prompts.append(new_mutated_child)
            print(f"The mutated prompt is a {type(new_mutated_child)}: {new_mutated_child}")
            LOGGER.info(f'The prompt after guided mutation is: "{new_mutated_child}"')
        
        return new_prompts, mutation_style_normal, mutation_style, words_to_use
    


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Settings for the Evolutionary Algorithms')
    parser.add_argument('--task', default='svamp', type=str, help='Task to be solved. Choose one of: [gsm8k, csqa, aqua, svamp, strategyqa]')
    parser.add_argument('--use_icl_examples', default=False, type=bool, help='whether to use in-context learning examples or not')
    parser.add_argument('--num_icl_examples', default=1, type=int, help='number of in-context learning examples used for evaluation')
    parser.add_argument('--num_of_samples', default=50, type=int, help='number of samples used for evaluation')
    parser.add_argument('--iterations', default=100, type=int, help='number of iterations for the EA')
    parser.add_argument('--number_of_mutations', default=1, type=int, help='number of mutations to perform')
    parser.add_argument('--seed', default=0, type=int, help='type of mutation')
    parser.add_argument('--mutate_population', default=True, type=bool, help='whether to mutate the population or not')
    parser.add_argument('--patience', default=5, type=int, help='after how many bad results we stop')
    args = parser.parse_args()
    print(args)
    logger_name = f"./evo_logs/Evo_{args.task}_output.log"
    LOGGER = setup_logger('progress_logger', logger_name)

    tokenizer = AutoTokenizer.from_pretrained("berkeley-nest/Starling-LM-7B-alpha")
    model = AutoModelForCausalLM.from_pretrained("berkeley-nest/Starling-LM-7B-alpha", torch_dtype = torch.float16)
    model = model.to('cuda')

    soc_model = SocraticGPT(model, tokenizer)

    trainset, testset = load_data(args.task)

    prompt_engine = GenPrompt(args, testset, soc_model, tokenizer, trainset)

    best_fitness = 0
    stagnation_count = 0
    patience = args.patience
    all_mutation_styles = load_mutation_prompts(args.task)
    print(f"Mutation styles: {all_mutation_styles}")

    for iter in range(args.iterations):

        if iter == 0:
            LOGGER.info(f"Arguments: {args}")
            LOGGER.info(f"Evaluation of the initial population")
            population = prompt_engine.initialise_population()
            initial_population = population
            print(f"The population is {population}")
            # fitness_dict = prompt_engine.evaluate_population(population)

            if args.task == 'abs_nar':
                fitness_dict = {
                    "Let's think step by step":0.56,
                    "Let's devise a plan and solve the problem step by step":0.39,
                    "Let's first prepare relevant information and make a plan. Then, let's answer the question step by step (pay attention to commonsense and logical coherence)":0.29,
                    "Let's work this out in a step by step way to be sure we have the right answer":0.52,
                    "Take a deep breath and work on this problem step-by-step":0.57,
                    "Start by dissecting the problem into its components, then address each part methodically":0.4,
                    "Dissect the problem carefully, address each part":0.34,
                    "Let's dive in and solve this challenge step by step":0.56,
                    "Approach the problem with a keen eye for detail and methodical precision":0.46,
                    "Embark on a quest for understanding, traversing the problem landscape with curiosity and logic":0.48,
                    "Your attention to detail here would mean everything":0.48,
                    "Please, let's focus and ensure we get the correct answer":0.41,
                    "Ensure that you read the question carefully and understand the problem before attempting to solve it":0.49,
                    "Dive in this problem and find the right answer":0.59,
                    "You can do this! Be careful in the calculations":0.51,
                    "Let's think slowly and carefully":0.6,
                }
            # elif args.task == 'csqa':
            #     fitness_dict = {
            #         "Let's think step by step":0.8,
            #         "Let's devise a plan and solve the problem step by step":0.8,
            #         "Let's first understand the problem, extract relevant variables and their corresponding numerals, and devise a complete plan.Then, let's carry out the plan, calculate intermediate variables (pay attention to correct numerical calculation and commonsense), solve the problem step by step, and show the answer":0.72,
            #         "Let's work this out in a step by step way to be sure we have the right answer":0.84,
            #         "Take a deep breath and work on this problem step-by-step":0.79,
            #         "Analyze this step by step":0.83,
            #         "Start by dissecting the problem into its components, then address each part methodically":0.66,
            #         "Let's approach this methodically, breaking it into smaller tasks":0.79,
            #         "Dissect the problem carefully, address each part":0.73,
            #         "Let's dive in this challenge":0.8,
            #         "Approach the problem with a keen eye for detail and methodical precision":0.81,
            #         "Embark on a quest for understanding, traversing the problem landscape with curiosity and logic":0.78,
            #         "Your attention to detail here would mean everything":0.8,
            #         "Please, let's focus and ensure we nail this down":0.77,
            #     }
            elif args.task == 'causal_judg':
                fitness_dict = {
                    "Let's think step by step": 0.66,
                    "Let's devise a plan and solve the problem step by step": 0.56,
                    "Let's first prepare relevant information and make a plan. Then, let's answer the question step by step (pay attention to commonsense and logical coherence)": 0.55,
                    "Let's work this out in a step by step way to be sure we have the right answer": 0.66,
                    "Take a deep breath and work on this problem step-by-step": 0.57,
                    "Start by dissecting the problem into its components, then address each part methodically": 0.62,
                    "Let's dive in and solve this challenge step by step": 0.66,
                    "Approach the problem with a keen eye for detail and methodical precision": 0.63,
                    "Embark on a quest for understanding, traversing the problem landscape with curiosity and logic": 0.57,
                    "Your attention to detail here would mean everything": 0.57,
                    "Please, let's focus and ensure we get the correct answer": 0.58,
                    "Ensure that you read the question carefully and understand the problem before attempting to solve it": 0.62,
                    "Dive in this problem and find the right answer": 0.59,
                    "You can do this! Be careful in the calculations": 0.56,
                    "Let's think slowly and carefully": 0.7,
                    "Let's think logically to derive the correct answer": 0.62,
                    "Focus on the challenge": 0.56,
                    "Read the question and find the right answer": 0.61,
                    "Let's put our thoughts in order": 0.63,
                }
            elif args.task == 'disamb':
                fitness_dict = {
                    "Let's think step by step":0.52,
                    "Let's devise a plan and solve the problem step by step":0.55,
                    "Let's first prepare relevant information and make a plan. Then, let's answer the question step by step (pay attention to commonsense and logical coherence)":0.55,
                    "Let's work this out in a step by step way to be sure we have the right answer":0.54,
                    "Take a deep breath and work on this problem step-by-step":0.57,
                    "Start by dissecting the problem into its components, then address each part methodically":0.52,
                    "Let's dive in and solve this challenge step by step":0.57,
                    "Approach the problem with a keen eye for detail and methodical precision":0.48,
                    "Embark on a quest for understanding, traversing the problem landscape with curiosity and logic":0.52,
                    "Your attention to detail here would mean everything":0.44,
                    "Please, let's focus and ensure we get the correct answer":0.37,
                    "Ensure that you read the question carefully and understand the problem before attempting to solve it":0.5,
                    "Dive in this problem and find the right answer":0.53,
                    "You can do this! Be careful in the calculations":0.54,
                    "Let's think slowly and carefully":0.52,
                    "Let's think logically to derive the correct answer":0.51,
                    "Focus on the challenge":0.44,
                    "Read the question and find the right answer":0.26,
                    "Let's put our thoughts in order":0.51,
                }

            # elif args.task == 'logic_ded3':
            #     fitness_dict = {
            #         "Let's think step by step": 0.62,
            #         "Let's devise a plan and solve the problem step by step": 0.62,
            #         "Let's first prepare relevant information and make a plan. Then, let's answer the question step by step (pay attention to commonsense and logical coherence)": 0.63,
            #         "Let's work this out in a step by step way to be sure we have the right answer": 0.63,
            #         "Take a deep breath and work on this problem step-by-step": 0.56,
            #         "Start by dissecting the problem into its components, then address each part methodically": 0.52,
            #         "Dissect the problem carefully, address each part": 0.52,
            #         "Let's dive in and solve this challenge step by step": 0.63,
            #         "Approach the problem with a keen eye for detail and methodical precision": 0.72,
            #         "Embark on a quest for understanding, traversing the problem landscape with curiosity and logic": 0.56,
            #         "Your attention to detail here would mean everything": 0.29,
            #         "Please, let's focus and ensure we get the correct answer": 0.67,
            #         "Ensure that you read the question carefully and understand the problem before attempting to solve it": 0.63,
            #         "Dive in this problem and find the right answer": 0.67,
            #         "You can do this! Be careful in the calculations": 0.66,
            #         "Let's think slowly and carefully": 0.66,
            #     }
            elif args.task == 'social_iqa':
                fitness_dict = {
                    "Let's think step by step": 0.68,
                    "Let's devise a plan and solve the problem step by step": 0.59,
                    "Let's first prepare relevant information and make a plan. Then, let's answer the question step by step (pay attention to commonsense and logical coherence)": 0.66,
                    "Let's work this out in a step by step way to be sure we have the right answer": 0.6,
                    "Take a deep breath and work on this problem step-by-step": 0.6,
                    "Start by dissecting the problem into its components, then address each part methodically": 0.7,
                    # "Let's dive in and solve this challenge step by step": 0.73,
                    "Approach the problem with a keen eye for detail and methodical precision": 0.65,
                    "Embark on a quest for understanding, traversing the problem landscape with curiosity and logic": 0.6,
                    "Your attention to detail here would mean everything": 0.57,
                    "Please, let's focus and ensure we get the correct answer": 0.66,
                    "Ensure that you read the question carefully and understand the problem before attempting to solve it": 0.65,
                    "Dive in this problem and find the right answer": 0.63,
                    "You can do this! Be careful in the calculations": 0.68,
                    "Let's think slowly and carefully": 0.66,
                    "Let's think logically to derive the correct answer": 0.58,
                    "Focus on the challenge": 0.65,
                    "Read the question and find the right answer": 0.61,
                    "Let's put our thoughts in order": 0.64,
                }
            elif args.task == 'sports_und':
                fitness_dict = {
                    "Let's think step by step": 0.8,
                    "Let's devise a plan and solve the problem step by step": 0.76,
                    "Let's first prepare relevant information and make a plan. Then, let's answer the question step by step (pay attention to commonsense and logical coherence)": 0.76,
                    "Let's work this out in a step by step way to be sure we have the right answer": 0.81,
                    "Take a deep breath and work on this problem step-by-step": 0.77,
                    "Start by dissecting the problem into its components, then address each part methodically": 0.78,
                    "Let's dive in and solve this challenge step by step": 0.82,
                    "Approach the problem with a keen eye for detail and methodical precision": 0.77,
                    "Embark on a quest for understanding, traversing the problem landscape with curiosity and logic": 0.77,
                    "Your attention to detail here would mean everything": 0.78,
                    "Please, let's focus and ensure we get the correct answer": 0.73,
                    "Ensure that you read the question carefully and understand the problem before attempting to solve it": 0.81,
                    "Dive in this problem and find the right answer": 0.81,
                    "You can do this! Be careful in the calculations": 0.71,
                    "Let's think slowly and carefully": 0.77,
                    "Let's think logically to derive the correct answer": 0.68,
                    "Focus on the challenge": 0.8,
                    # "Read the question and find the right answer": 0.85,
                    "Let's put our thoughts in order": 0.74,
                }
            
            else:
                fitness_dict = prompt_engine.evaluate_population(population)

            average_fitness = sum(fitness_dict.values())/len(fitness_dict)
            LOGGER.info(f"Generation: {iter} | Average fitness: {average_fitness}")
            initial_fitness_dict = fitness_dict

            for prompt in population:
                LOGGER.info(f"Population: {prompt} with fitness {fitness_dict[prompt]}")
            LOGGER.info(f"Genetic Algorithms starts")

            # create a dictionary of the mutation styles prompt
            mutation_dict = {}
            for mutation_style in all_mutation_styles:
                mutation_dict[mutation_style] = 0
    
        else:
            
            mutation_styles = list(mutation_dict.keys())

            parents = prompt_engine.select_parents(fitness_dict)
            children = prompt_engine.crossover(parents)
            print(f"\nThe child is: {children}\n")

            new_prompts, mutation_style_normal, mutation_style_guided, words_to_use = prompt_engine.mutate(children, population, fitness_dict, mutation_styles)
            print(f"\nMutated prompts: {new_prompts}\n")

            if len(new_prompts) == 1:
                child_acc = prompt_engine.evaluate_population([children])
                fitness_dict.update(child_acc)
                LOGGER.info(f'Generation {iter} Crossover: "{children}" | Fitness: {fitness_dict[children]}')
                mutated_accs = prompt_engine.evaluate_population(new_prompts)
                fitness_dict.update(mutated_accs)
                LOGGER.info(f'Generation {iter} guided-mutation prompt: "{new_prompts[0]}" | Fitness: {fitness_dict[new_prompts[0]]}')
            else:
                print("2 mutation in this generation")
                mutated_accs = prompt_engine.evaluate_population(new_prompts)
                print(f"The mutated_accs are: {mutated_accs}")
                fitness_dict.update(mutated_accs)
                LOGGER.info(f'Generation {iter} normal-mutated child: "{new_prompts[0]}" | Fitness: {fitness_dict[new_prompts[0]]}')
                LOGGER.info(f'Generation {iter} guided-mutated prompt: "{new_prompts[1]}" | Fitness: {fitness_dict[new_prompts[1]]}')

            population.extend(new_prompts)
            best_prompt = max(fitness_dict, key = fitness_dict.get)
            current_best_fitness = fitness_dict[best_prompt]

            # Remove the the 2 worst prompts
            sorted_prompts = sorted(fitness_dict, key=fitness_dict.get)
            population = [prompt for prompt in population if prompt not in sorted_prompts[:2]]
            fitness_dict = {key: value for key, value in fitness_dict.items() if key not in sorted_prompts[:2]}
            print(f"Removing the first worst prompt: {sorted_prompts[0]}")
            print(f"Removing the second worst prompt: {sorted_prompts[1]}")
            LOGGER.info(f"Removing the first worst prompt: {sorted_prompts[0]}")
            LOGGER.info(f"Removing the second worst prompt: {sorted_prompts[1]}")


            average_fitness = sum(fitness_dict.values())/len(fitness_dict)
            print(f"Generation: {iter} | Average fitness: {average_fitness}")
            LOGGER.info(f"Generation: {iter} | Average fitness: {average_fitness}")

            if len(new_prompts) > 1:
                mutated_accs = list(mutated_accs.values())
                if mutated_accs[0] > average_fitness:
                    mutation_dict[mutation_style_normal] += 1
                if mutated_accs[1] > average_fitness:
                    mutation_dict[mutation_style_guided] += 1
            else:
                mutated_accs = list(mutated_accs.values())
                if mutated_accs[0] > average_fitness:
                    mutation_dict[mutation_style_guided] += 1

            
            # Mutate the mutation prompts
            mut_prompt = mutate_mutation_prompt(mutation_dict)
            print(f"Mutation prompt: {mut_prompt}")
            model_input = f"GPT4 Correct System: You are a powerful AI assistant that follows the instructions given.<|end_of_turn|>GPT4 Correct User: {mut_prompt}<|end_of_turn|>GPT4 Correct Assistant:"
            input_ids = tokenizer(model_input, return_tensors="pt").input_ids.to('cuda')
            outputs = model.generate(
                input_ids,
                max_new_tokens=300,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            response_ids = outputs[0]
            response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
            output_mut_prompt = response_text.split('GPT4 Correct Assistant:')[1]
            try:
                pattern = r'[\[{\(](.*?)[\]}\)]'            
                new_mut_prompt = re.findall(pattern, output_mut_prompt)
                new_mut_prompt = new_mut_prompt[0]
                if new_mut_prompt not in mutation_dict:
                    mutation_dict[new_mut_prompt] = 0
                    print(f"New mutation prompt: {new_mut_prompt}")
                    LOGGER.info(f"Adding new mutation style: {new_mut_prompt}")

            except:
                print("Didn't respond with text in brackets")

            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_prompt = best_prompt
                stagnation_count = 0
                print(f"Stagnation set to 0")
            else:
                stagnation_count += 1
                print(f"Stagnation +1")

            LOGGER.info(f'Generation {iter}: Best prompt: "{best_prompt}" | Fitness {best_fitness}')

            if stagnation_count >= patience:
                if best_prompt not in initial_fitness_dict.keys():
                    LOGGER.info(f'Converged at generation {iter} with best prompt: "{best_prompt}" | Fitness {best_fitness}')
                    LOGGER.info(f'The final population is: {population}')
                    break

