thinking_styles = ["Let's think step by step",
                   "Let's first understand the problem and devise a plan to solve the problem. Then, let's carry out the plan and solve the problem step by step",
                   "Let's first understand the problem, extract relevant variables and their corresponding numerals, and make a plan. Then, let's carry out the plan, calculate intermediate variables (pay attention to correct numerical calculation and common sense), solve the problem step by step, and show the answer",
                   "Let's work this out in a step by step way to be sure we have the right answer",
                   "Take a deep breath and work on this problem step-by-step",
                   "Let's break this down step-by-step", 
                   "Let's first understand the relationship between the variables and devise a plan to solve the problem. Then, let's carry out the plan and solve the problem step by step",
                   "Take a breath and focus on the task",
                   "Let's be very precise and accurate in our calculations",
                   "Let's be very precise, because this is very important",
                   "If you find the correct answer, you will be rewarded",
                   "Inhale, exhale and find the correct answer",
                   "This is very important, so do your best",
                   "Take a deep breath, understand the problem, devise a plan to solve it, and then carry out the plan",
                   "We can not afford to make a mistake here",
                   "DO NOT GET THIS WRONG PLEASE",
                   "Use an abstract and unconventional thinking style",
                   "Let's look at this from multiple perspectives",
                   "Let's visualize the problem",
                   "Inhale deeply, exhale slowly, and embark on this problem-solving journey with a step-by-step mindset",
                   "Derive the solution by dividing and conquering",
                   "Let's divide and conquer",
                   "Let's break this down into smaller parts and solve each part separately",
                   "Let's start with the desired outcome and work backwards",
                   "Create visual diagrams that represent the relationships between different aspects of the problem",
                   "Draw parallels between the current problem and similar problems that have been solved before"
                   "Let's create a simplified version or model of the problem to gain insights and test potential solutions",
                   "Let's dissect this puzzle, analyzing its components methodically to arrive at a coherent solution",
                   "Let's focus on the big picture and then work our way down to the details",
                   "Let's focus on logic and reasoning to arrive at a well-considered solution."
                   "Let's use holistic thinking",
                   "Let's use analogical thinking and draw parallels between the current problem and similar problems that have been solved before",
                   "Let's embrace a structured thought process, navigating through the problem systematically",
                   "Our strategy involves breaking down the problem into manageable steps, ensuring a clear and accurate solution",
                   "Our approach will be to methodically work through the problem, ensuring accuracy at each step to derive the correct answer",
                   "With a calculated approach, let's dissect and solve in a stepwise fashion"]

task_description = ["Solve this math problem.",
                    "Derive the solution to this math problem.",
                    "Calculate the answer to this math problem.",
                    "Compute the solution.",
                    "Give your answer as an arabic numeral:",
                    "Answer with an arabic numeral the following math problem.",]

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
                     "Mutate the prompt to provide an alternative viewpoint",]




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
"Break down and solve this math problem step by step.",
"Embark on a journey to derive the solution to this math challenge.",
"Methodically compute the answer to this problem with our structured approach.",
"Solve this problem step by step and provide an arabic numeral.",
"Dissect and solve this puzzle, giving the answer as an arabic numeral.",
"Look at this math challenge from multiple perspectives, computing the solution.",
"Break down the problem, calculate, and provide the answer.",
"Visualize the problem, derive the solution step by step, and answer with an arabic numeral.",
"Compute the solution with a calculated, stepwise approach.",
"Embrace a structured thought process, navigate systematically, and give an arabic numeral answer.",
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

standard_prompts = [
"Let's think step by step.",
"Let's first understand the problem and devise a plan to solve the problem. Then, let's carry out the plan and solve the problem step by step.",
"Let's first understand the problem, extract relevant variables and their corresponding numerals, and make a plan. Then, let's carry out the plan, calculate intermediate variables (pay attention to correct numerical calculation and common sense), solve the problem step by step, and show the answer.",
"Let's work this out in a step by step way to be sure we have the right answer.",
"Take a deep breath and work on this problem step-by-step.",
]

mutation_styles = [
    "The mutation is a variant of the input prompt using unconventional thinking.",
    "The mutation is a variant of the input prompt that introduces an element of surprise.",
    "The mutation is a variant of the input prompt that provides an alternative viewpoint.",
    "The mutation is a variant of the input prompt that makes it easier to understand.",
    "The mutation is a variant of the input prompt that helps an AI assistant follow the instructions.",
    "The mutation is a variant of the input prompt that adds more details.",
    "The mutation is a variant of the input prompt that motivates the AI assistant to approach the task with enthusiasm and a can-do attitude.",]


contrastive_samples = [
    {
        "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "explanation": "There are 15 trees originally. Then there were 21 trees after the Grove workers planted some more. So there must have been 21 - 15 = 6 trees that were planted.",
        "answer": "6",
        "wrong_explanation": "There are 21 - 15 = 6 trees originally. Then there were 15 trees after the Grove workers planted some more. So there must have been 21 trees that were planted.",
        "wrong_answer": "21"
    },
    {
        "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "explanation": "There are originally 3 cars. Then 2 more cars arrive. Now 3 + 2 = 5 cars are in the parking lot.",
        "answer": "5",
        "wrong_explanation": "There are originally 3 + 2 = 5 cars. Then 3 more cars arrive. Now 2 cars are in the parking lot.",
        "wrong_answer": "2"
    },
    {
        "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "explanation": "Originally, Leah had 32 chocolates and her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39 pieces left in total.",
        "answer": "39",
        "wrong_explanation": "Originally, Leah had 32 + 42 = 74 chocolates and her sister had 32. So in total they had 74 - 35 = 39. After eating 35, they had 42 pieces left in total.",
        "wrong_answer": "42"
    },
    {
        "question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "explanation": "Jason had 20 lollipops originally. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8 lollipops.",
        "answer": "8",
        "wrong_explanation": "Jason had 20 - 12 = 8 lollipops originally. Then he had 20 after giving some to Denny. So he gave Denny 12 lollipops.",
        "wrong_answer": "12"
    },
    {
        "question": "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?",
        "explanation": "He writes each friend 3*2=6 pages a week. So he writes 6*2=12 pages every week. That means he writes 12*52=624 pages a year.",
        "answer": "624",
        "wrong_explanation": "He writes each friend 12*52=624 pages a week. So he writes 3*2=6 pages every week. That means he writes 6*2=12 pages a year.",
        "wrong_answer": "12"
    },
    {
        "question": "Henry made two stops during his 60-mile bike trip. He first stopped after 20 miles. His second stop was 15 miles before the end of the trip. How many miles did he travel between his first and second stops?",
        "explanation" : "Henry traveled a total of 60 miles on his bike trip. His second stop was 15 miles before the end of the trip, so he had already traveled 60 - 15 = 45 miles. His first stop was after 20 miles, so he traveled 45 - 20 = 25 miles between his first and second stops.",
        "answer" : "25",
        "wrong_explanation": "Henry traveled a total of 60 miles on his bike trip. His first stop was after 20 miles, so he traveled 20 miles between his starting point and his first stop. His second stop was 15 miles before the end of the trip, so he traveled 60 - 15 = 45 miles between his second stop and the end of the trip. Therefore, he traveled a total of 20 + 45 = 65 miles between his first and second stops.",
        "wrong_answer" : "65"
    },
]