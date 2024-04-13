### Initial Prompts ###

gsm8k_initial_prompts = [
    "Let's think step by step",
    "Let's first understand the problem and devise a plan to solve the problem. Then, let's carry out the plan and solve the problem step by step",
    "Let's first understand the problem, extract relevant variables and their corresponding numerals, and devise a plan. Then, let's carry out the plan, calculate the intermediate results (pay attention to calculation and common sense), solve the problem step by step, and show the answer",
    "Let's work this out in a step by step way to be sure we have the right answer",
    "Take a deep breath and work on this problem step-by-step",
    "Break this down",
    "A little bit of arithmetic and a logical approach will help us quickly arrive at the solution to this problem",
    "Let's combine our numerical command and clear thinking to quickly and accurately decipher the answer",
    "Let's be very precise and accurate in our calculations",
    "Let's create a simplified version of the problem to gain insights and test potential solutions",
    "Embark on a journey to derive the solution to this problem",
    "Compute the solution with a calculated, stepwise approach",
    "Let's be very precise and accurate in our calculations",
    "Our approach will be to methodically work through the problem, ensuring accuracy at each step to derive the correct answer",
    "Slow down, let's break this down into manageable steps",
    "Inhale deeply, exhale slowly, and embark on this problem-solving journey with a step-by-step mindset",
    "Inhale deeply, then utilize a step-by-step logical reasoning process and mathematical tools to deconstruct the problem into manageable components, ultimately leading to a more efficient resolution; let's ensure our calculations are precise and accurate"
]

svamp_initial_prompts = [
    "Let's think step by step",
    "Let's first understand the problem and devise a plan to solve the problem. Then, let's carry out the plan and solve the problem step by step",
    "Let's first understand the problem, extract relevant variables and their corresponding numerals, and make a complete plan. Then, let's carry out the plan, calculate intermediate variables (pay attention to correct numerical calculation and commonsense), solve the problem step by step, and show the answer",
    "Let's work this out in a step by step way to be sure we have the right answer",
    "Take a deep breath and work on this problem step-by-step",
    "Break this down",
    "A little bit of arithmetic and a logical approach will help us quickly arrive at the solution to this problem",
    "Let's combine our numerical command and clear thinking to quickly and accurately decipher the answer",
    "Let's be very precise and accurate in our calculations",
    "Let's create a simplified version of the problem to gain insights and test potential solutions",
    "Embark on a journey to derive the solution to this problem",
    "Compute the solution with a calculated, stepwise approach",
    "Let's be very precise and accurate in our calculations",
    "Our approach will be to methodically work through the problem, ensuring accuracy at each step to derive the correct answer",
    "Slow down, let's break this down into manageable steps",
    "Inhale deeply, exhale slowly, and embark on this problem-solving journey with a step-by-step mindset",
]

csqa_initial_prompts = [
    "Let's think step by step",
    "Let's devise a plan and solve the problem step by step",
    "Let's first understand the problem, extract relevant variables and their corresponding numerals, and devise a complete plan.Then, let's carry out the plan, calculate intermediate variables (pay attention to correct numerical calculation and commonsense), solve the problem step by step, and show the answer",
    "Let's work this out in a step by step way to be sure we have the right answer",
    "Take a deep breath and work on this problem step-by-step",
    "Analyze this step by step",    
    "Start by dissecting the problem into its components, then address each part methodically",
    "Let's approach this methodically, breaking it into smaller tasks",
    "Dissect the problem carefully, address each part",
    "Let's dive in this challenge",
    "Approach the problem with a keen eye for detail and methodical precision",
    "Embark on a quest for understanding, traversing the problem landscape with curiosity and logic",
    "Your attention to detail here would mean everything",
    "Please, let's focus and ensure we nail this down",
]

strategyqa_initial_prompts = [
    "Let's think step by step",
    "Let's devise a plan and solve the problem step by step",
    "Let's first prepare relevant information and make a plan. Then, let's answer the question step by step (pay attention to commonsense and logical coherence)",
    "Let's work this out in a step by step way to be sure we have the right answer",
    "Take a deep breath and work on this problem step-by-step",
    "Analyze this step by step",    
    "Start by dissecting the problem into its components, then address each part methodically",
    "Let's approach this methodically, breaking it into smaller tasks",
    "Dissect the problem carefully, address each part",
    "Let's dive in this challenge",#and solve this challenge",
    # "Let's dive in and solve this challenge step by step",
    "Approach the problem with a keen eye for detail and methodical precision",
    "Embark on a quest for understanding, traversing the problem landscape with curiosity and logic",
    "Your attention to detail here would mean everything",
    "Please, let's focus and ensure we nail this down",
]

bb_initial_prompts = [
    "Let's think step by step",
    "Let's devise a plan and solve the problem step by step",
    "Let's first prepare relevant information and make a plan. Then, let's answer the question step by step (pay attention to commonsense and logical coherence)",
    "Let's work this out in a step by step way to be sure we have the right answer",
    "Take a deep breath and work on this problem step-by-step",
    "Start by dissecting the problem into its components, then address each part methodically",
    "Dissect the problem carefully, address each part",
    "Let's dive in and solve this challenge step by step",
    "Approach the problem with a keen eye for detail and methodical precision",
    "Embark on a quest for understanding, traversing the problem landscape with curiosity and logic",
    "Your attention to detail here would mean everything",
    "Please, let's focus and ensure we get the correct answer",
    "Ensure that you read the question carefully and understand the problem before attempting to solve it",
    "Dive in this problem and find the right answer",
    "You can do this! Be careful in the calculations",
    "Let's think slowly and carefully",
]




### Mutation Styles ###

gsm8k_mutation_styles = [
    "The mutation is a variant of the input prompt that introduces a structured thought process.",
    "The mutation is a variant of the input prompt using unconventional thinking.",
    "The mutation is a variant of the input prompt that provides an alternative viewpoint.",
    "The mutation presents a tweaked version of the task, emphasizing logical steps in problem-solving.",
    "This variant of the prompt, through mutation, offers a fresh perspective on the problem, focusing on strategic thinking.",
    "Through mutation, the prompt is altered to showcase problem-solving strategies and logical reasoning.",
    "The mutation introduces a revised version of the prompt, aiming to illuminate the process of logical reasoning and problem-solving.",
    ]


svamp_mutation_styles = [
    "The mutation is a short variant of the input prompt that introduces a structured thought process.",
    "The mutation is a short variant of the input prompt that provides an alternative viewpoint.",
    "The mutation presents a tweaked version of the task, emphasizing logical steps in problem-solving.",
    "This variant of the prompt, through mutation, offers a fresh perspective on the problem, focusing on quickness and strategic thinking.",
    "Through mutation, the prompt is altered to showcase motivation and logical reasoning.",
    "The mutation introduces a revised version of the prompt, aiming to focus on logic and swiftness.",
]

csqa_mutation_styles = [
    "The mutation is a variant of the input prompt that introduces a structured thought process.",
    "The mutation is a variant of the input prompt that provides an alternative viewpoint.",
    "The mutation presents a tweaked version of the task, emphasizing logical steps in problem-solving.",
    "This variant of the prompt, through mutation, offers a fresh perspective on the problem, focusing on quickness and strategic thinking.",
    "Through mutation, the prompt is altered to showcase motivation and logical reasoning.",
    "The mutation introduces a revised version of the prompt, aiming to focus on logic and swiftness.",
    "The mutation is a revised version of the prompt that emphasizes the importance of logical reasoning and problem-solving strategies.",
    "The mutation is a short variant of the prompt aiming to promote a positive thinking and common sense."
]

strategyqa_mutation_styles = [
    "The mutation is a variant of the input prompt that introduces a structured thought process.",
    "The mutation is a variant of the input prompt that provides an alternative viewpoint.",
    "The mutation presents a tweaked version of the task, emphasizing logical steps in problem-solving.",
    "This variant of the prompt, through mutation, offers a fresh perspective on the problem, focusing on quickness and strategic thinking.",
    "Through mutation, the prompt is altered to showcase motivation and logical reasoning.",
    "The mutation introduces a revised version of the prompt, aiming to focus on logic and swiftness.",
    "The mutation is a revised version of the prompt that emphasizes the importance of logical reasoning and problem-solving strategies.",
    "The mutation is a short variant of the prompt aiming to promote a positive thinking and common sense."
]


bb_mutation_styles = [
    "The mutation is a concise variant of the input prompt that introduces a structured thought process.",
    "The mutation is a short and concise variant of the input prompt that provides an alternative viewpoint.",
    "The mutation is a paraphrased version of the prompt that focuses on quickness and strategic thinking.",
    "The mutation is a paraphrased version of the prompt that includes all the information of the initial prompt.",
    "The mutation is a detailed paraphrased version of the prompt which will make it easier to understand.",
    "This variant of the prompt, through mutation, offers a fresh perspective on the problem, focusing on quickness and strategic thinking.",
    "Through mutation, the prompt is altered to showcase motivation and logical reasoning.",
    "The mutation introduces a revised version of the prompt, aiming to focus on logic and swiftness.",
    "The mutation is a short variant of the prompt aiming to promote a positive thinking and common sense."
]



### Inference Prompts ###

gsm8k_inference_prompts = [
    "Let's think step by step",
    "Let's first understand the problem and devise a plan to solve the problem. Then, let's carry out the plan and solve the problem step by step",
    "Let's first understand the problem, extract relevant variables and their corresponding numerals, and devise a plan. Then, let's carry out the plan, calculate the intermediate results (pay attention to calculation and common sense), solve the problem step by step, and show the answer",
    "Let's work this out in a step by step way to be sure we have the right answer",
    "Take a deep breath and work on this problem step-by-step",
    "Break this down",
    "A little bit of arithmetic and a logical approach will help us quickly arrive at the solution to this problem",
    "Let's combine our numerical command and clear thinking to quickly and accurately decipher the answer",
    # Prompt obtained by evoalg
    "Focus on strategic thinking to swiftly find the accurate solution"
]

svamp_inference_prompts = [
    "Let's think step by step",
    "Let's first understand the problem and devise a plan to solve the problem. Then, let's carry out the plan and solve the problem step by step",
    "Let's first understand the problem, extract relevant variables and their corresponding numerals, and make a complete plan. Then, let's carry out the plan, calculate intermediate variables (pay attention to correct numerical calculation and commonsense), solve the problem step by step, and show the answer",
    "Let's work this out in a step by step way to be sure we have the right answer",
    "Take a deep breath and work on this problem step-by-step",
    "Break this down",
    "A little bit of arithmetic and a logical approach will help us quickly arrive at the solution to this problem",
    "Let's combine our numerical command and clear thinking to quickly and accurately decipher the answer",
    # Prompt obtained by evoalg
    "Let's be very precise and accurate in our calculations: Compute the solution with a stepwise approach"
]

csqa_inference_prompts = [
    "Let's think step by step",
    "Let's devise a plan and solve the problem step by step",
    "Let's first understand the problem, extract relevant variables and their corresponding numerals, and devise a complete plan.Then, let's carry out the plan, calculate intermediate variables (pay attention to correct numerical calculation and commonsense), solve the problem step by step, and show the answer",
    "Let's work this out in a step by step way to be sure we have the right answer",
    "Take a deep breath and work on this problem step-by-step",
    # Prompt obtained by evoalg
    "Your keen eye for detail, combined with methodical precision, would mean everything in approaching the problem"
]

abs_nar_inference_prompts = [
    "Let's think step by step",
    "Let's devise a plan and solve the problem step by step",
    "Let's first prepare relevant information and make a plan. Then, let's answer the question step by step (pay attention to commonsense and logical coherence)",
    "Let's first understand the problem, extract relevant variables and their corresponding numerals, and devise a complete plan.Then, let's carry out the plan, calculate intermediate variables (pay attention to correct numerical calculation and commonsense), solve the problem step by step, and show the answer",
    "Let's work this out in a step by step way to be sure we have the right answer",
    "Take a deep breath and work on this problem step-by-step",
    # Prompt obtained by evoalg
    "Let's think slowly and carefully, collect data, strategize, and maintain logical coherence and practicality while engaging with the question"
]

causal_judg_inference_prompts = [
    "Let's think step by step",
    "Let's devise a plan and solve the problem step by step",
    "Let's first prepare relevant information and make a plan. Then, let's answer the question step by step (pay attention to commonsense and logical coherence)",
    "Let's first understand the problem, extract relevant variables and their corresponding numerals, and devise a complete plan.Then, let's carry out the plan, calculate intermediate variables (pay attention to correct numerical calculation and commonsense), solve the problem step by step, and show the answer",
    "Let's work this out in a step by step way to be sure we have the right answer",
    "Take a deep breath and work on this problem step-by-step",
    # Prompt obtained by evoalg
    "To swiftly locate the accurate response, let's inhale deeply and tackle this methodically"
]

date_under_inference_prompts = [
    "Let's think step by step",
    "Let's devise a plan and solve the problem step by step",
    "Let's first prepare relevant information and make a plan. Then, let's answer the question step by step (pay attention to commonsense and logical coherence)",
    "Let's first understand the problem, extract relevant variables and their corresponding numerals, and devise a complete plan.Then, let's carry out the plan, calculate intermediate variables (pay attention to correct numerical calculation and commonsense), solve the problem step by step, and show the answer",
    "Let's work this out in a step by step way to be sure we have the right answer",
    "Take a deep breath and work on this problem step-by-step",
    # Prompt obtained by evoalg
    "Swiftly traverse the problem landscape, employing strategic thinking and a keen attention to detail, to ensure a precise and accurate resolution"
]

social_iqa_inference_prompts = [
    "Let's think step by step",
    "Let's devise a plan and solve the problem step by step",
    "Let's first prepare relevant information and make a plan. Then, let's answer the question step by step (pay attention to commonsense and logical coherence)",
    "Let's first understand the problem, extract relevant variables and their corresponding numerals, and devise a complete plan.Then, let's carry out the plan, calculate intermediate variables (pay attention to correct numerical calculation and commonsense), solve the problem step by step, and show the answer",
    "Let's work this out in a step by step way to be sure we have the right answer",
    "Take a deep breath and work on this problem step-by-step",
    # Prompt obtained by evoalg
    "Quick-thinking and strategic instincts for accurate calculations"
]

sports_und_inference_prompts = [
    "Let's think step by step",
    "Let's devise a plan and solve the problem step by step",
    "Let's first prepare relevant information and make a plan. Then, let's answer the question step by step (pay attention to commonsense and logical coherence)",
    "Let's first understand the problem, extract relevant variables and their corresponding numerals, and devise a complete plan.Then, let's carry out the plan, calculate intermediate variables (pay attention to correct numerical calculation and commonsense), solve the problem step by step, and show the answer",
    "Let's work this out in a step by step way to be sure we have the right answer",
    "Take a deep breath and work on this problem step-by-step",
    # Prompt obtained by evoalg
    "Embark on a journey to uncover knowledge, igniting curiosity and logic, while maintaining a positive mindset and focusing on understanding the problem, devising a strategy with organized data to dissect challenges into smaller parts, navigating the landscape methodically, and achieving both favorable outcomes and exceptional results, all while being deliberate in our approach and prioritizing strategic thinking"
]


def load_inference_prompts(task):
    if task == 'gsm8k':
        return gsm8k_inference_prompts
    elif task == 'svamp':
        return svamp_inference_prompts
    elif task == 'csqa':
        return csqa_inference_prompts
    elif task == 'abs_nar':
        return abs_nar_inference_prompts
    elif task == 'causal_judg':
        return causal_judg_inference_prompts
    elif task == 'date_under':
        return date_under_inference_prompts
    elif task == 'social_iqa':
        return social_iqa_inference_prompts
    elif task == 'sports_und':
        return sports_und_inference_prompts
    else:
        raise ValueError("Task not supported")
    
    