import transformers
import re
import time
import random

class SocraticGPT:
    def __init__(self, role, model, tokenizer, mutation_style, n_round=1):
        self.role = role
        self.tokenizer = tokenizer
        self.model = model
        self.n_round = n_round
        self.mutation_style = mutation_style
        self.other_role = "Theaetetus" if role == "Socrates" else "Socrates"
        self.history = []

    def set_problem(self, problem):
        self.history.append({
            "role": "system",
            "content": f"{self.role} and {self.other_role} are two AI assistants for Tony to mutate a prompt. {self.mutation_style} The initial prompt is: \"[{problem}]\".\n\n{self.role} and {self.other_role} will engage in multi-round dialogue to mutate the prompt for an evolutionary algorithm. The final mutated prompt has to be within brackets."
        })
        self.history.append({
            "role": "assistant",
            "content": f"Hi {self.other_role}, let's work together to mutate the prompt. {self.mutation_style} Both of us can suggest improvements or mutations. The mutated prompt has to be short and concise. The mutated prompt has to be within brackets."
        })

    def get_response(self):
        input_prompt = "".join([f"{msg['role']}: {msg['content']}" for msg in self.history])
        input_ids = self.tokenizer(input_prompt, return_tensors="pt").input_ids.to('cuda')
        outputs = self.model.generate(
            input_ids,
            max_length=1000,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        response_ids = outputs[0]
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        self.history.append({
            "role": "assistant",
            "content": response_text
        })
        return response_text

def dialogue():

    mut_style = "The mutation is a variant of the input prompt that motivates the AI assistant to approach the task with enthusiasm and a can-do attitude."

    tokenizer = transformers.AutoTokenizer.from_pretrained("berkeley-nest/Starling-LM-7B-alpha")
    model = transformers.AutoModelForCausalLM.from_pretrained("berkeley-nest/Starling-LM-7B-alpha", device_map='auto')

    # Initialize SocraticGPT instances
    socrates = SocraticGPT(role="Socrates", mutation_style=mut_style, model=model, tokenizer=tokenizer)
    theaetetus = SocraticGPT(role="Theaetetus", mutation_style=mut_style, model=model, tokenizer=tokenizer)

    # Define the initial prompt
    initial_prompt =  "Let's work this out in a step by step way to be sure we have the right answer"

    # Set the initial prompt for both Socrates and Theaetetus
    socrates.set_problem(initial_prompt)
    theaetetus.set_problem(initial_prompt)

    start = time.time()

    # Begin the dialogue for multiple rounds
    for round_num in range(socrates.n_round):

        print(f"Round {round_num+1}:")

        # Socrates provides a response
        socrates_response = socrates.get_response()
        print(f"{socrates.role}: {socrates_response}")

        if "final" in socrates_response.lower():
            print("Socrates mentioned 'final'. Ending the dialogue.")
            break

        # Theaetetus provides a response
        theaetetus_response = theaetetus.get_response()
        print(f"{theaetetus.role}: {theaetetus_response}")


    end = time.time()
    print(f"All rounds lasted: {end-start}")

    # Print the final mutated prompt
    final_prompt = socrates_response  

    mutated_prompt = re.findall(r'"([^"]*)"', final_prompt)

    # Ignore the first mutation, which is the initial prompt
    final_prompt = mutated_prompt[1:]

    # Remove brackets and replace " with ' for the final prompt
    final_prompt = [prompt.replace('"',"'").replace("[","").replace("]","").replace("!",".") for prompt in final_prompt]

    final_prompt = random.sample(final_prompt, 1)

    return final_prompt

    
if __name__ == '__main__':
    new_prompts = dialogue()
    print(new_prompts)
