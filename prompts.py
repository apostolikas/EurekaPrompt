thinking_styles = ["Let's think step by step",
                   "Let's first understand the problem and devise a plan to solve the problem. Then, let's carry out the plan and solve the problem step by step",
                   "Let's first understand the problem, extract relevant variables and their corresponding numerals, and make a plan. Then, let's carry out the plan, calculate intermediate variables (pay attention to correct numerical calculation and commonsense), solve the problem step by step, and show the answer",
                   "Let's work this out in a step by step way to be sure we have the right answer",
                   "Take a deep breath and work on this problem step-by-step",
                   "Let's break this down step-by-step", 
                   "Let's look at this from multiple perspectives",
                   "Let's visualize the problem",
                   "Inhale deeply, exhale slowly, and embark on this problem-solving journey with a step-by-step mindset",
                   "Let's dissect this puzzle, analyzing its components methodically to arrive at a coherent solution",
                   "Let's embrace a structured thought process, navigating through the problem systematically",
                   "Our strategy involves breaking down the problem into manageable steps, ensuring a clear and accurate solution",
                   "Our approach will be to methodically work through the problem, ensuring accuracy at each step to derive the correct answer",
                   "With a calculated approach, let's dissect and solve this problem in a stepwise fashion"]

task_description = ["Solve this math problem:",
                    "Derive the solution to this math problem:",
                    "Calculate the answer to this math problem:",
                    "Compute the answer to this math problem:",
                    "Compute the solution:",
                    "Give the answer to this math problem:",
                    "Give your answer as an arabic numeral:",
                    "Provide an arabic numeral for the following math problem:",
                    "Answer with an arabic numeral the following math problem:",]

mutation_prompts = ["Modify the prompt to make it more detailed",
                     "Improve the prompt by adding helpful advice",
                     "Change the wording of the prompt in an unexpected way",
                     "Change the wording of the prompt in a way that makes it easier to understand",
                     "Make a variant of the prompt",
                     "Modify the prompt to help an LLM follow the instructions",
                     "Elaborate on the prompt to do what it wants",
                     "Create a mutated version of the prompt",
                     "Generate a mutated version of the prompt by adding more details",
                     "Use unconventional thinking and create a mutated version of the prompt",
                     "Generate a version of the prompt that introduces an element of surprise",
                     "Mutate the prompt to provide an alternative viewpoint",
                     "Mutate the prompt"]

import random
from datasets import load_dataset

thinking_styles = random.sample(thinking_styles, 5)
task_descriptions = random.sample(task_description, 5)
original_dataset = load_dataset("gsm8k", 'main', split='test')

# Define a function to add the 'label' key to each entry
def add_label(entry):
    answer_value = int(entry['answer'].split('####')[1].replace(',', ''))
    entry['label'] = answer_value
    return entry

# Use the map method to apply the function to each entry in the dataset
dataset = original_dataset.map(add_label)

# Take 100 random samples from the dataset
samples100 = dataset.shuffle(seed=42).select(range(100))


prompt = f'''
{thinking_styles[0]}. {task_description[0]} 
{samples100[1]['question']}
'''
print(prompt)

print(samples100[1]['answer'])
