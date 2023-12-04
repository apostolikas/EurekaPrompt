import random
import math
from eval import evaluate_GSM8K
import argparse
import logging
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

class SimAnneal:
    def __init__(self, args, dataset, model, tokenizer):
        self.args = args
        self.dataset = dataset
        self.model = model
        self.tokenizer = tokenizer
        # self.helper_model = TBD
        # self.helper_tokenizer = TBD
        
        self.logger = logging.basicConfig(filename='Simulated_Annealing.log', level = logging.INFO)


    def evaluate(self, prompt, num_of_samples):
        '''
        This function evaluates the fitness of a prompt by calculating the accuracy of the model on the dataset.
        The current version only supports GSM8K and checks whether the final number in the output is the same as the label.
        '''

        score = 0

        samples = self.dataset.shuffle(seed=42).select(range(num_of_samples))

        for i,sample in enumerate(samples):

            # Input to the model

            # Output from the model

            # Evaluate the output
            label = sample['label']
            
            # Update the fitness
            score += evaluate_GSM8K(y_pred, label)

        score = score/num_of_samples

        return score

    def generate_neighboring_solution(prompt):
        '''
        Function to generate a neighboring solution by perturbing the prompt using an LLM.
        '''
        pass


    def run_simulated_annealing(self, initial_prompt):
        '''
        This function runs the simulated annealing algorithm to optimize the prompt.
        '''

        current_prompt = initial_prompt
        best_prompt = initial_prompt
        temperature = self.args.initial_temperature

        while temperature > 1.0:

            for _ in range(self.args.iterations_per_temperature):

                neighbor_prompt = self.generate_neighboring_solution(current_prompt)

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
    parser.add_argument("--initial_temperature", default=100, type=float, help="Initial temperature for simulated annealing")
    parser.add_argument("--cooling_rate", default=0.95, type=float, help="Cooling rate for simulated annealing")
    parser.add_argument("--iterations_per_temperature", default=50, type=int, help="Number of iterations per temperature for simulated annealing")
    parser.add_argument("--num_of_samples", default=100, type=int, help="Number of samples to evaluate the fitness on")
    args = parser.parse_args()


    # Load the dataset
    dataset = load_dataset("gsm8k", 'main', split='test')

    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Orca-2-13b", device_map = "auto")
    model = AutoModelForCausalLM.from_pretrained("microsoft/Orca-2-13b", use_fast = True)

    prompt_engine = SimAnneal(args, dataset, model, tokenizer)

    prompts = [] # TBD
    scores = {}

    for initial_prompt in prompts:
        best_prompt, best_score = prompt_engine.run_simulated_annealing(initial_prompt)
        scores[best_prompt] = best_score

    # Evaluate the prompts on the full dataset

    for prompt in scores.keys():
        score = prompt_engine.evaluate(prompt, len(dataset))
        scores[prompt] = score

    print(f"The optimal prompt is {max(scores, key = scores.get)} with a score of {scores[max(scores, key = scores.get)]}")
    








