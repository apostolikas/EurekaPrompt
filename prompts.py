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

mixed_prompts = [
"Break down this math problem systematically and provide the solution with a step-by-step mindset.",
"Embark on this problem-solving journey, inhaling deeply and exhaling slowly, while deriving the solution to this mathematical challenge.",
"Methodically work through this problem, computing the answer step by step with our structured approach.",
"Work on this problem step-by-step, taking a deep breath, and provide an arabic numeral for the following math problem.",
"Dissect this puzzle methodically, analyze its components, and solve the math problem, giving your answer as an arabic numeral.",
"Look at this from multiple perspectives and compute the solution to this mathematical challenge in a structured thought process.",
"Our strategy involves breaking down the problem into manageable steps, so calculate and provide the answer to this math problem.",
"Visualize the problem and derive the solution with a step-by-step mindset, answering with an arabic numeral for the following math problem.",
"With a calculated approach, dissect and solve this problem in a stepwise fashion by computing the solution to this mathematical challenge.",
"Embrace a structured thought process, navigate through the problem systematically, and give your answer as an arabic numeral to this math problem.",
]


shorter_prompts = [
"Break down and solve this math problem step by step."
"Embark on a journey to derive the solution to this math challenge."
"Methodically compute the answer to this problem with our structured approach."
"Solve this problem step by step and provide an arabic numeral."
"Dissect and solve this puzzle, giving the answer as an arabic numeral."
"Look at this math challenge from multiple perspectives, computing the solution."
"Break down the problem, calculate, and provide the answer."
"Visualize the problem, derive the solution step by step, and answer with an arabic numeral."
"Compute the solution with a calculated, stepwise approach."
"Embrace a structured thought process, navigate systematically, and give an arabic numeral answer."
]

shortest_prompts = [
"Solve this math problem step by step.",
"Derive the solution to this math challenge.",
"Compute the answer methodically.",
"Step-by-step solution, provide an arabic numeral.",
"Dissect and solve, give an arabic numeral.",
"Approach from multiple perspectives, compute the solution.",
"Break down, calculate, provide the answer.",
"Visualize, derive, answer with an arabic numeral.",
"Compute the solution stepwise.",
"Embrace a structured approach, give an arabic numeral answer.",
]

long_prompts = [
"Let's systematically break down the components of this math problem, examining each step in detail, and provide a solution by meticulously working through the problem with a step-by-step mindset.",
"Embark on a comprehensive problem-solving journey where you not only derive the solution to this challenging math problem but also delve into the underlying concepts, ensuring a thorough understanding of the mathematical principles at play.",
"Our approach involves a methodical computation of the answer to this problem, emphasizing precision and accuracy at each stage. Navigate through the problem systematically, adhering to our structured approach to arrive at a clear and well-calculated solution.",
"Approach this problem with a systematic and step-by-step solution in mind. Take a deep breath, focus on the task at hand, and provide the answer as an arabic numeral, ensuring a meticulous numerical calculation process.",
"Delve into the intricacies of this puzzle, dissecting its components methodically to arrive at a comprehensive solution. After solving the math problem, present the answer as an arabic numeral, showcasing your attention to detail in both analysis and calculation.",
"Explore this math challenge from various perspectives, considering different angles and approaches. Compute the solution with a structured thought process, ensuring a comprehensive understanding of the problem and a well-calculated answer.",
"Our strategy is to break down the problem into manageable steps, ensuring a clear and accurate solution. Calculate each step meticulously and provide the answer, showcasing a detailed and systematic approach to problem-solving.",
"Visualize the problem to gain a deeper understanding of its intricacies. Derive the solution step by step, paying careful attention to each calculation, and respond with an arabic numeral, reflecting your comprehensive and detailed problem-solving skills.",
"Approach the solution to this problem in a calculated and stepwise manner. Ensure precision at each stage of computation, navigating through the problem systematically to arrive at a well-thought-out and accurate solution.",
"Embrace a structured thought process as you navigate through this problem, ensuring a comprehensive analysis of its components. Provide an arabic numeral answer, showcasing your commitment to a detailed and meticulous problem-solving approach.",
]

abstract_prompts = [
"Let's deconstruct the layers of complexity within this mathematical challenge, carefully crafting a step-by-step solution that not only unravels the intricacies of the problem but also emphasizes a meticulous approach to numerical accuracy.",
"Embark on a thoughtful exploration of this mathematical enigma, dissecting its core elements and intricacies to derive a solution that reflects not just computational prowess but a profound understanding of the underlying mathematical principles.",
"Our chosen strategy involves a deliberate and systematic computation of the answer, navigating through the mathematical landscape with precision and clarity, ensuring that each step contributes meaningfully to the ultimate resolution of the problem.",
"In approaching this mathematical puzzle, adopt a methodical and stepwise mindset, intertwining a deep breath with the precision of numerical calculation to unravel the layers of complexity and present a solution in the form of an arabic numeral.",
"Immerse yourself in a detailed dissection of this mathematical puzzle, methodically analyzing its constituent parts to derive a solution that not only showcases numerical fluency but also a keen eye for the intricacies embedded within the problem.",
"Adopt a multidimensional perspective in navigating this mathematical challenge, weaving together various angles of approach to compute a solution that is not only methodically derived but also reflects a nuanced understanding of the problem at hand.",
"Our strategic blueprint involves the careful deconstruction of the problem, breaking it down into manageable components, and navigating through each calculation with a meticulous eye for detail, ultimately arriving at a solution that speaks to clarity and accuracy.",
"Envision the mathematical landscape before you, carefully deriving the solution step by step with a focus on both the visual and numerical aspects of the problem, culminating in an arabic numeral that encapsulates the comprehensive nature of your problem-solving prowess.",
"Embark on a calculated journey towards the solution, ensuring each step is taken with precision and purpose, resulting in a methodically derived answer that reflects both the systematic approach and the accuracy of your numerical calculations.",
"Immerse yourself in a structured thought process as you navigate through this mathematical challenge, extracting insights from each component to present an arabic numeral answer that mirrors not just the solution but the careful consideration given to every facet of the problem.",
]

passive_voice_prompts = [
"Let the math problem be systematically broken down and solved step by step.",
"Allow the solution to this math challenge to be derived on this comprehensive problem-solving journey.",
"The answer to this problem can be methodically computed with our structured approach.",
"This problem can be approached with a systematic and step-by-step solution in mind.",
"The puzzle components can be methodically dissected, and the answer presented as an arabic numeral.",
"This math challenge can be explored from various perspectives, and the solution computed with a structured thought process.",
"The strategy involves breaking down the problem into manageable steps, ensuring a clear and accurate solution.",
"The problem can be visualized to gain a deeper understanding, and the solution derived step by step.",
"The solution to this problem can be approached in a calculated and stepwise manner.",
"A structured thought process can be embraced as the problem is navigated, and an arabic numeral answer provided.",
]

