import random
import math
from utils import *
import argparse
import logging
from prompts.math_prompts import *
from prompts.nli_prompts import *
from prompts.open_qa_prompts import *
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from conversation import SocraticGPT
import torch

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
    def __init__(self, args, trainset, testset):
        self.args = args
        self.trainset = trainset
        self.testset = testset
        # self.model = AutoModelForCausalLM.from_pretrained("microsoft/Orca-2-7b", device_map = 'auto', torch_dtype = torch.float16)
        # self.tokenizer = AutoTokenizer.from_pretrained("microsoft/Orca-2-7b", use_fast = False)

        self.tokenizer = AutoTokenizer.from_pretrained("berkeley-nest/Starling-LM-7B-alpha")
        self.model = AutoModelForCausalLM.from_pretrained("berkeley-nest/Starling-LM-7B-alpha", torch_dtype = torch.float16)
        self.model = self.model.to('cuda')

        self.prompt_pool = self.construct_prompt_pool()
        # self.helper_tokenizer = AutoTokenizer.from_pretrained("berkeley-nest/Starling-LM-7B-alpha")
        # self.helper_model = AutoModelForCausalLM.from_pretrained("berkeley-nest/Starling-LM-7B-alpha", device_map='auto')
        self.logger = setup_logger('progress_logger', 'SA_output.log')


    def construct_prompt_pool(self):
        '''
        This function constructs the prompt pool by sampling from the training set.
        '''
        if self.args.type_of_prompts == 'short':
            initial_prompts = shorter_prompts
        elif self.args.type_of_prompts == 'normal':
            initial_prompts = mixed_prompts
        elif self.args.type_of_prompts == 'long':
            initial_prompts = long_prompts
        elif self.args.type_of_prompts == 'abstract':
            initial_prompts = abstract_prompts
        elif self.args.type_of_prompts == 'passive':
            initial_prompts = passive_voice_prompts
        elif self.args.type_of_prompts == 'standard':
            initial_prompts = standard_prompts
        else:
            raise Exception('This type of prompts is not supported')
        
        return initial_prompts


    def evaluate(self, prompt, num_of_samples):
        '''
        This function evaluates the fitness of a prompt by calculating the accuracy of the model on the dataset.
        The current version only supports GSM8K and checks whether the final number in the output is the same as the label.
        '''
        
        score = 0

        samples = self.testset.shuffle(seed=self.args.seed).select(range(num_of_samples)) 

        for sample in tqdm(samples):

            question = sample['question']
            label = sample['label']

            # Construct the prompt
            if self.args.use_icl_examples:

                if self.args.use_contrastive_cot:
                    contrastive_prompt = construct_contrastive_icl_example(contrastive_samples, self.args.num_icl_examples)
                    model_input = f'''{contrastive_prompt}
                    Question: {question}
                    {prompt}
                    '''
                else:
                    icl_prompt = construct_icl_examples(self.trainset, [prompt], self.args.num_icl_examples)
                    model_input = f'''{icl_prompt}
                    Question: {question}
                    {prompt}
                    '''
          
            else:
                    model_input = f'''Question: {question}\n\n{prompt}\n\nAnswer:'''

            ### Orca2.7b ###

            # system_message = "You are Orca, an AI language model created by Microsoft. You are a cautious assistant. You carefully follow instructions in order to solve math problems."
            # prompt = f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{model_input}<|im_end|>\n<|im_start|>assistant"
            # input_ids = self.tokenizer(prompt, return_tensors='pt').input_ids.to("cuda")
            # output_ids = self.model.generate(input_ids)
            # text_output = self.tokenizer.batch_decode(output_ids)[0]
                    
            ### Starling ###
                    
            input_ids = self.tokenizer(model_input, return_tensors="pt").input_ids.to('cuda')
            outputs = self.model.generate(
                input_ids,
                max_length=500,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            response_ids = outputs[0]
            text_output = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            
            score += evaluate_GSM8K(text_output, label)

        score = score/num_of_samples
        return score


    def generate_neighboring_solution_with_OPRO(self, prompt, scores):
        '''
        Function to generate a neighboring solution by perturbing the prompt using an LLM.
        '''
        samples = self.trainset.shuffle(seed=0).select(range(2))
        questions = samples['question']
        labels = samples['label']

        sorted_scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)}
        sample_prompts = list(sorted_scores.keys())[:1]

        perturb_prompt = f'''I have some texts along with their corresponding scores. The texts are arranged in ascending order based on their scores, where higher scores indicate better quality.
        text:
        {sample_prompts[0]}
        score: 
        {scores[sample_prompts[0]]}
        
        text:
        {prompt}
        score:
        {scores[prompt]}

        The following exemplars show how to apply your text: you replace "INS" in each input with your text, then read the input and give an output. We say your output is wrong if your output is different from the given output, and we say your output is correct if they are the same.

        input: 
        Q: {questions[0]}
        A: "INS"
        output: {labels[0]}

        input:
        Q: {questions[1]}
        A: "INS"
        output: {labels[1]}

        Write only your new single "INS" within brackets that is different from the old ones and has a score as high as possible. Write the text in square brackets.        
        '''

        input_ids = self.tokenizer(perturb_prompt, return_tensors="pt").input_ids.to('cuda')
        outputs = self.model.generate(
            input_ids,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        new_prompt = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return new_prompt
    

    def generate_neighboring_solution_with_LLM(self, prompt):

        perturb_prompt = f'''{random.choice(mutation_prompts)}
        {prompt}
        Your mutated prompt has to be within brackets.
        '''

        input_ids = self.helper_tokenizer(perturb_prompt, return_tensors="pt").input_ids.to('cuda')
        outputs = self.model.generate(
            input_ids,
            pad_token_id=self.helper_tokenizer.pad_token_id,
            eos_token_id=self.helper_tokenizer.eos_token_id,
        )
        new_prompt = self.helper_tokenizer.decode(outputs[0], skip_special_tokens=True)

        return new_prompt


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


    def run_simulated_annealing(self, initial_prompt):
        '''
        This function runs the simulated annealing algorithm to optimize the prompt.
        '''

        current_prompt = initial_prompt
        best_prompt = initial_prompt
        temperature = self.args.initial_temperature
        # sa_scores = {}
        # for initial_prompts in self.prompt_pool:
        #     sa_scores[initial_prompts] = self.evaluate(initial_prompts, self.args.num_of_samples)

        sa_scores = {
            "Let's think step by step": 0.38,
            "Let's first understand the problem and devise a plan to solve the problem. Then, let's carry out the plan and solve the problem step by step" : 0.35,
            # "Let's work this out in a step by step way to be sure we have the right answer." : 0.0,
            "Take a deep breath and work on this problem step-by-step." : 0.23,
            # "Draw parallels between the current problem and similar problems that have been solved before" : 0.0,
        }

        # with open('SA_initial_prompts_scores.txt', 'w') as f:
        #     for prompt, score in sa_scores.items():
        #         f.write("%s:%s\n" % (prompt, score))
        
        stop_step = 0 

        while temperature > 10: #1.0:

            for _ in range(self.args.iterations_per_temperature):

                if stop_step >= 2:
                    print("Stop iterating...")
                    break

                if self.args.mutation_type == 'opro':

                    neighbor_prompt = self.generate_neighboring_solution_with_OPRO(current_prompt, sa_scores)

                elif self.args.mutation_type == 'llm':

                    neighbor_prompt = self.generate_neighboring_solution_with_LLM(current_prompt)

                elif self.args.mutation_type == 'dialogue':
            
                    mutation_style = random.choice(mutation_styles)
                    neighbor_prompt = self.mutate_with_dialogue(current_prompt, mutation_style)

                current_score = self.evaluate(current_prompt, self.args.num_of_samples)
                neighbor_score = self.evaluate(neighbor_prompt, self.args.num_of_samples)

                print(f"Current Prompt: {current_prompt} with score {current_score}")
                print(f"Generated Prompt: {neighbor_prompt} with score {neighbor_score}")

                # Metropolis criterion
                if neighbor_score > current_score: #or random.uniform(0, 1) < math.exp((neighbor_score - current_score) / temperature):
                    current_prompt = neighbor_prompt
                    print(f"Replaced current prompt with neighbor prompt since {neighbor_score} > {current_score}")
                    max_score = self.evaluate(best_prompt, self.args.num_of_samples)
                    print(f"Best score so far is {max_score} with prompt {best_prompt}")
                    # Update the best prompt if needed
                    if neighbor_score > max_score:
                        best_prompt = neighbor_prompt
                        best_score = neighbor_score
                        print(f"Replaced best prompt with neighbor prompt since {neighbor_score} > {max_score}")

                else:
                    stop_step += 1
                    print("Stop step increased: ", stop_step)


            self.logger.info("After {} iterations at temperature {}, the best prompt is {} with score {}".format(self.args.iterations_per_temperature, temperature, best_prompt, best_score))

            temperature *= self.args.cooling_rate

            self.logger.info("Temperature is now {}".format(temperature))
            

        self.logger.info("The best prompt is {} with score {}".format(best_prompt, best_score))

        return best_prompt, best_score


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--type_of_prompts", default="standard", type=str, help="Type of prompts to use")
    parser.add_argument("--num_icl_examples", default=1, type=int, help="Number of ICL examples to use")
    parser.add_argument("--initial_temperature", default=100, type=float, help="Initial temperature for simulated annealing")
    parser.add_argument("--use_icl_examples", default=False, type=bool, help="Whether to use ICL examples")
    parser.add_argument('--use_contrastive_cot', default=False, type=bool, help='whether to use contrastive cot in-context learning examples or not')
    parser.add_argument("--cooling_rate", default=0.5, type=float, help="Cooling rate for simulated annealing")
    parser.add_argument("--iterations_per_temperature", default=5, type=int, help="Number of iterations per temperature for simulated annealing")
    parser.add_argument("--num_of_samples", default=15, type=int, help="Number of samples to evaluate the fitness on")
    parser.add_argument("--mutation_type", default="dialogue", type=str, help="Type of mutation to use")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    args = parser.parse_args()


    # Load the dataset
    original_test_dataset = load_dataset("gsm8k", 'main', split='test')
    original_train_dataset = load_dataset("gsm8k", 'main', split='train')

    testset = original_test_dataset.map(add_label)
    trainset = original_train_dataset.map(add_label)

    prompt_engine = SimAnneal(args, trainset, testset)

    scores = {}
    # initial_prompt = random.choice(prompt_engine.prompt_pool)
    initial_prompt = "Let's submerge ourselves in the conundrum, identify vital variables and their numerical values, and establish a plan. As we carry out the plan, let's scrutinize intermediate findings (ensure correct numerical calculations and logical reasoning), tackle the problem progressively, and unveil the answer"
    best_prompt, best_score = prompt_engine.run_simulated_annealing(initial_prompt)
    scores[best_prompt] = best_score

    # # Evaluate the prompts on the full dataset
    # for prompt in scores.keys():
    #     score = prompt_engine.evaluate(prompt, len(testset))
    #     scores[prompt] = score

    print(f"The optimal prompt is {max(scores, key = scores.get)} with a score of {scores[max(scores, key = scores.get)]}")