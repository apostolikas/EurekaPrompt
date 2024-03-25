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
        elif self.args.task == 'aqua':
            initial_prompts = aqua_initial_prompts
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
            samples = random.sample(self.testset['examples'], self.args.num_of_samples)

            for sample in tqdm(samples):
                narrative = sample['input']
                label = sample['label']
                answer_choices = sample['answer_choices']
                model_input = f'''Question: Can you choose the most related proverb from the list of 5 proverbs given a narrative?\nNarrative: {narrative}\nAnswer choices: {answer_choices}\nAnswer: {prompt}'''
                input_prompt = f'''GPT4 Correct User: {model_input}<|end_of_turn|>GPT4 Correct Assistant:'''
                text_output = self.model.get_response(input_prompt)
                text_output = text_output.split('GPT4 Correct Assistant:')[1]
                fitness += evaluate_CSQA(text_output, label)

        elif self.args.task == 'cause_effect':

            random.seed(self.args.seed)
            samples = random.sample(self.testset['examples'], self.args.num_of_samples)
            question = sample['task_prefix']

            for sample in tqdm(samples):
                label = sample['label']
                answer_choices = sample['answer_choices']
                model_input = f'''Question: {question}\nAnswer choices: {answer_choices}\nAnswer: {prompt}'''
                input_prompt = f'''GPT4 Correct User: {model_input}<|end_of_turn|>GPT4 Correct Assistant:'''
                text_output = self.model.get_response(input_prompt)
                text_output = text_output.split('GPT4 Correct Assistant:')[1]
                fitness += evaluate_CSQA(text_output, label)


        elif self.args.task == 'disamb':

            random.seed(self.args.seed)
            samples = random.sample(self.testset['examples'], self.args.num_of_samples)
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
            samples = random.sample(self.testset['examples'], self.args.num_of_samples)
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
            samples = random.sample(self.testset['examples'], self.args.num_of_samples)

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
        return random.sample(list(fitness_dict.keys()), 2)

    def crossover(self, parents):

        parent_prompt1 = parents[0]
        parent_prompt2 = parents[1]
        print(f"The parents are: {[parent_prompt1, parent_prompt2]}")
        final_prompt = crossover_dialogue(self.helper_model, parent_prompt1, parent_prompt2)
        final_prompt = final_prompt[-1]
        print(f"child prompt is a {type(final_prompt)}: {final_prompt}")

        return final_prompt


    def mutate(self, child, population, fitness_dict):

        if self.args.task == 'csqa':
            mutation_styles = commonsense_mutation_styles
        elif self.args.task == 'strategyqa':
            mutation_styles = strategyqa_mutation_styles
        elif self.args.task == 'gsm8k':
            mutation_styles = gsm8k_mutation_styles
        elif self.args.task == 'aqua':
            mutation_styles = aqua_mutation_styles
        elif self.args.task == 'svamp':
            mutation_styles = svamp_mutation_styles
        elif self.args.task in bb_tasks:
            mutation_styles = bb_mutation_styles
        else:
            raise ValueError("Task not supported")

        new_prompts = []

        if random.random() > 0.5:
            random.seed() # delete
            mutation_style = random.choice(mutation_styles)
            print(f"Mutating the child with mutation style with index: {mutation_styles.index(mutation_style)}")
            #mutated_child = mutation_dialogue(self.helper_model, mutation_style, child)
            mutated_child = mutation_dialogue(self.helper_model, mutation_style, child, False, None)
            if isinstance(mutated_child, list):
                mutated_child = mutated_child[-1]
            print(f"The mutated child prompt is a {type(mutated_child)}: {mutated_child}")

            new_prompts.append(mutated_child)

        if self.args.mutate_population:
            random.seed() # delete
            print("Mutating random prompts from the population")
            mutation_style = random.choice(mutation_styles)
            random_prompt = random.choice(population)
            print(f"The prompt that will be mutated is: {random_prompt}")

            use_words = True if random.random() > 0.3 else False

            if use_words == True:

                mut_prompt = contstruct_mutation_prompt(fitness_dict)
                model_input = f"GPT4 Correct System: You are a powerful AI assistant that will help me identify the crucial parts in some text.<|end_of_turn|>GPT4 Correct User: {mut_prompt}<|end_of_turn|>GPT4 Correct Assistant:"

                response_text = self.helper_model.get_response(model_input)
                
                match = re.search(r'\[(.*?)\]', response_text)
                if match:
                    list_string = match.group(1)
                    final_list = [item.strip() for item in list_string.split(",")]
                    words_to_use = random.choice(final_list)
                    print(f"Words that will be used for the mutation: {words_to_use}")
                else:
                    words_to_use = "step by step"
            else:
                words_to_use = None
                            
            # new_mutated_child = mutation_dialogue(self.helper_model, mutation_style, random_prompt)
            new_mutated_child = mutation_dialogue(self.helper_model, mutation_style, random_prompt, use_words, words_to_use)

            if isinstance(new_mutated_child, list):
                new_mutated_child = new_mutated_child[-1]
                
            new_prompts.append(new_mutated_child)
            print(f"The mutated prompt is a {type(new_mutated_child)}: {new_mutated_child}")
        
        return new_prompts
    


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Settings for the Evolutionary Algorithms')
    parser.add_argument('--task', default='abs_nar', type=str, help='Task to be solved. Choose one of: [gsm8k, csqa, aqua, svamp, strategyqa]')
    parser.add_argument('--use_icl_examples', default=False, type=bool, help='whether to use in-context learning examples or not')
    parser.add_argument('--num_icl_examples', default=1, type=int, help='number of in-context learning examples used for evaluation')
    parser.add_argument('--num_of_samples', default=35, type=int, help='number of samples used for evaluation')
    parser.add_argument('--iterations', default=20, type=int, help='number of iterations for the EA')
    parser.add_argument('--number_of_mutations', default=1, type=int, help='number of mutations to perform')
    parser.add_argument('--seed', default=0, type=int, help='type of mutation')
    parser.add_argument('--mutate_population', default=True, type=bool, help='whether to mutate the population or not')
    parser.add_argument('--patience', default=5, type=int, help='after how many bad results we stop')
    args = parser.parse_args()
    print(args)
    logger_name = f"Evo_{args.task}_output.log"
    logger = setup_logger('progress_logger', logger_name)

    bb_tasks = ['abs_nar', 'causal_judg', 'cause_effect', 'date_under', 'disamb', 'logic_ded3', 'social_iqa', 'sports_und']

    tokenizer = AutoTokenizer.from_pretrained("berkeley-nest/Starling-LM-7B-alpha")
    model = AutoModelForCausalLM.from_pretrained("berkeley-nest/Starling-LM-7B-alpha", torch_dtype = torch.float16)
    model = model.to('cuda')

    soc_model = SocraticGPT(model, tokenizer)

    if args.task == 'gsm8k':
        original_test_dataset = read_jsonl('./data/gsm8k_test.jsonl')
        testset = list(map(add_label, original_test_dataset))
        original_train_dataset = read_jsonl('./data/gsm8k_train.jsonl')
        trainset = list(map(add_label, original_train_dataset))

    elif args.task == 'svamp':
        with open('./data/SVAMP.json') as f:
            testset = json.load(f)
        testset = list(map(lambda x: {**x, 'full_question': x['Body'] + ' ' + x['Question']}, testset))
        testset = list(map(lambda x: {**x, 'Answer': int(x['Answer']) if x['Answer'].is_integer() else x['Answer']}, testset))
        trainset = testset

    elif args.task == 'csqa':
        testset = read_jsonl('./data/csqa_val.jsonl')
        for item in testset:
            choices = item["question"]["choices"]
            choice_answers = ", ".join(map(format_choice, choices))
            item["choice_answers"] = choice_answers

        trainset = read_jsonl('./data/csqa_train.jsonl')
        for item in trainset:
            choices = item["question"]["choices"]
            choice_answers = ", ".join(map(format_choice, choices))
            item["choice_answers"] = choice_answers

    elif args.task == 'strategyqa':
        with open('./data/strategyqa_train.json')as f:
            testset = json.load(f) 
        trainset = testset

    elif args.task == 'aqua':
        testset = read_jsonl('./data/aqua_test.json')

        for instance in testset:
            instance['answer_choices'] = format_aqua_options(instance['options'])
        trainset = testset

    elif args.task in bb_tasks:
        with open(f'./data/{args.task}.json') as f:
            testset = json.load(f)

        testset['examples'] = list(map(process_bb_example, testset['examples']))
        trainset = testset

    else:
        raise ValueError("Task not supported")


    prompt_engine = GenPrompt(args, testset, soc_model, tokenizer, trainset)

    best_fitness = 0
    stagnation_count = 0
    patience = args.patience

    for iter in range(args.iterations):

        if iter == 0:
            logger.info(f"Arguments: {args}")
            logger.info(f"Evaluation of the initial population")
            population = prompt_engine.initialise_population()
            print(f"The population is {population}")
            fitness_dict = prompt_engine.evaluate_population(population)

            for prompt in population:
                logger.info(f"Population: {prompt} with fitness {fitness_dict[prompt]}")
            logger.info(f"Genetic Algorithms starts")

        else:
            
            parents = prompt_engine.select_parents(fitness_dict)
            children = prompt_engine.crossover(parents)
            print(f"\nThe child is: {children}\n")
            
            new_prompts = prompt_engine.mutate(children, population, fitness_dict)
            print(f"\nMutated prompts: {new_prompts}\n")

            if [children] not in population:
                fitness_dict.update(prompt_engine.evaluate_population([children]))
                population.extend([children])
            logger.info(f'Generation {iter} Crossover: "{children}" | Fitness: {fitness_dict[children]}')

            fitness_dict.update(prompt_engine.evaluate_population(new_prompts))
            for prompt in new_prompts:
                logger.info(f'Generation {iter} Mutation: "{prompt}" | Fitness: {fitness_dict[prompt]}')

            population.extend(new_prompts)
            best_prompt = max(fitness_dict, key = fitness_dict.get)
            current_best_fitness = fitness_dict[best_prompt]

            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_prompt = best_prompt
                stagnation_count = 0
            else:
                stagnation_count += 1

            logger.info(f'Generation {iter}: Best prompt: "{best_prompt}" | Fitness {best_fitness}')

            if stagnation_count >= patience:
                logger.info(f'Converged at generation {iter} with best prompt: "{best_prompt}" | Fitness {best_fitness}')
                logger.info(f'The final population is: {population}')
                break
