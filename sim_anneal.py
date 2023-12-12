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


class SimAnneal:
    def __init__(self, args, trainset, testset, model, tokenizer):
        self.args = args
        self.trainset = trainset
        self.testset = testset
        self.model = model
        self.tokenizer = tokenizer        
        self.logger = logging.basicConfig(filename='Simulated_Annealing.log', level = logging.INFO)
        self.prompt_pool = self.construct_prompt_pool()
        self.helper_model = AutoModelForCausalLM.from_pretrained("TheBloke/OpenHermes-2.5-Mistral-7B-GGUF", model_file="./openhermes-2.5-mistral-7b.Q4_K_M.gguf", model_type="mistral")

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
        else:
            raise Exception('This type of prompts is not supported')
        
        return initial_prompts


    def evaluate(self, prompt, num_of_samples):
        '''
        This function evaluates the fitness of a prompt by calculating the accuracy of the model on the dataset.
        The current version only supports GSM8K and checks whether the final number in the output is the same as the label.
        '''
        
        score = 0
        samples = self.testset.shuffle(seed=42).select(range(num_of_samples)) 

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
            
            score += evaluate_GSM8K(text_output, label)

        score = score/num_of_samples
        return score


    def generate_neighboring_solution_with_OPRO(self, prompt, scores):
        '''
        Function to generate a neighboring solution by perturbing the prompt using an LLM.
        '''
        # Have to change it to remember 2 previous prompts and then use OPRO to generate a new prompt #  
        samples = self.trainset.shuffle(seed=42).select(range(2))
        questions = samples['question']
        labels = samples['label']

        sample_prompts = random.sample(list(scores), 2)
        sample_scores = [scores[sample_prompts[0]], scores[sample_prompts[1]]]

        perturb_prompt = f'''I have some texts along with their corresponding scores. The texts are arranged in ascending order based on their scores, where higher scores indicate better quality.
        text:
        {sample_prompts[0]}
        score: 
        {sample_scores[0]}
        
        text:
        {sample_prompts[1]}
        score:
        {sample_scores[1]}

        The following exemplars show how to apply your text: you replace <INS> in each input with your text, then read the input and give an output. We say your output is wrong if your output is different from the given output, and we say your output is correct if they are the same.

        input: 
        Q: {questions[0]}
        A: <INS>
        output: 
        {labels[0]}

        input:
        Q: {questions[1]}
        A: <INS>
        output:
        {labels[1]}

        Write your new text that is different from the old ones and has a score as high as possible. Write the text in square brackets.
        '''

        new_prompt = self.helper_model(perturb_prompt)

        return new_prompt
    

    def generate_neighboring_solution_with_LLM(self, prompt):

        perturb_prompt = f'''{random.choice(mutation_prompts)}
        {prompt}
        '''

        new_prompt = self.helper_model(perturb_prompt)

        return new_prompt



    # def run_normal_simulated_annealing(self, initial_prompt):
    #     '''
    #     This function runs the simulated annealing algorithm to optimize the prompt.
    #     '''

    #     current_prompt = initial_prompt
    #     best_prompt = initial_prompt
    #     temperature = self.args.initial_temperature

    #     while temperature > 1.0:

    #         for _ in range(self.args.iterations_per_temperature):

    #             neighbor_prompt = self.generate_neighboring_solution_with_LLM(current_prompt)

    #             current_score = self.evaluate(current_prompt, self.args.num_of_samples)
    #             neighbor_score = self.evaluate(neighbor_prompt, self.args.num_of_samples)

    #             # Metropolis criterion
    #             if neighbor_score > current_score or random.uniform(0, 1) < math.exp((neighbor_score - current_score) / temperature):
    #                 current_prompt = neighbor_prompt

    #                 # Update the best prompt if needed
    #                 if neighbor_score > self.evaluate(best_prompt, self.args.num_of_samples):
    #                     best_prompt = neighbor_prompt
    #                     best_score = neighbor_score

                
    #         self.logger.info("After {} iterations at temperature {}, the best prompt is {} with score {}".format(self.args.iterations_per_temperature, temperature, best_prompt, best_score))

    #         temperature *= self.args.cooling_rate

    #         self.logger.info("Temperature is now {}".format(temperature))

    #     self.logger.info("The best prompt is {} with score {}".format(best_prompt, best_score))

    #     return best_prompt, best_score
    

    def run_boosted_simulated_annealing(self, initial_prompt):
        '''
        This function runs the simulated annealing algorithm to optimize the prompt.
        '''

        current_prompt = initial_prompt
        best_prompt = initial_prompt
        temperature = self.args.initial_temperature
        sa_scores = {}

        for initial_prompts in self.prompt_pool:
            sa_scores[initial_prompts] = self.evaluate(initial_prompts, self.args.num_of_samples)

        with open('Initial_prompts_scores.txt', 'w') as f:
            for prompt, score in sa_scores.items():
                f.write("%s:%s\n" % (prompt, score))
        
        while temperature > 1.0:

            for _ in range(self.args.iterations_per_temperature):

                neighbor_prompt = self.generate_neighboring_solution_with_OPRO(current_prompt, sa_scores)

                current_score = self.evaluate(current_prompt, self.args.num_of_samples)
                neighbor_score = self.evaluate(neighbor_prompt, self.args.num_of_samples)

                # Metropolis criterion
                if neighbor_score > current_score or random.uniform(0, 1) < math.exp((neighbor_score - current_score) / temperature):
                    current_prompt = neighbor_prompt

                    # Update the best prompt if needed
                    if neighbor_score > self.evaluate(best_prompt, self.args.num_of_samples):
                        best_prompt = neighbor_prompt
                        best_score = neighbor_score

                
            self.logger.info("After {} iterations at temperature {}, the best prompt is {} with score {}".format(self.args.iterations_per_temperature, temperature, best_prompt, best_score))

            temperature *= self.args.cooling_rate

            self.logger.info("Temperature is now {}".format(temperature))

        self.logger.info("The best prompt is {} with score {}".format(best_prompt, best_score))

        return best_prompt, best_score

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--type_of_prompts", default="short", type=str, help="Type of prompts to use")
    parser.add_argument("--num_icl_examples", default=3, type=int, help="Number of ICL examples to use")
    parser.add_argument("--initial_temperature", default=100, type=float, help="Initial temperature for simulated annealing")
    parser.add_argument("--use_icl_examples", default=True, type=bool, help="Whether to use ICL examples")
    parser.add_argument('--use_contrastive_cot', default=True, type=bool, help='whether to use contrastive cot in-context learning examples or not')
    parser.add_argument("--cooling_rate", default=0.95, type=float, help="Cooling rate for simulated annealing")
    parser.add_argument("--iterations_per_temperature", default=50, type=int, help="Number of iterations per temperature for simulated annealing")
    parser.add_argument("--num_of_samples", default=100, type=int, help="Number of samples to evaluate the fitness on")
    args = parser.parse_args()


    # Load the dataset
    original_test_dataset = load_dataset("gsm8k", 'main', split='test')
    original_train_dataset = load_dataset("gsm8k", 'main', split='train')

    testset = original_test_dataset.map(add_label)
    trainset = original_train_dataset.map(add_label)

    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Orca-2-13b", device_map = "auto")
    model = AutoModelForCausalLM.from_pretrained("microsoft/Orca-2-13b", use_fast = True)

    prompt_engine = SimAnneal(args, trainset, testset, model, tokenizer)

    scores = {}
    initial_prompt = random.choice(prompt_engine.prompt_pool)
    best_prompt, best_score = prompt_engine.run_normal_simulated_annealing(initial_prompt)
    scores[best_prompt] = best_score

    # Evaluate the prompts on the full dataset
    for prompt in scores.keys():
        score = prompt_engine.evaluate(prompt, len(testset))
        scores[prompt] = score

    print(f"The optimal prompt is {max(scores, key = scores.get)} with a score of {scores[max(scores, key = scores.get)]}")