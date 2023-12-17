import transformers
import re
import time

tokenizer = transformers.AutoTokenizer.from_pretrained("berkeley-nest/Starling-LM-7B-alpha")
model = transformers.AutoModelForCausalLM.from_pretrained("berkeley-nest/Starling-LM-7B-alpha")

class SocraticGPT:
    def __init__(self, role, n_round=1, model="berkeley-nest/Starling-LM-7B-alpha"):
        self.role = role
        self.model = model
        self.n_round = n_round
        self.other_role = "Theaetetus" if role == "Socrates" else "Socrates"
        self.history = []

    def set_problem(self, problem):
        self.history.append({
            "role": "system",
            "content": f"{self.role} and {self.other_role} are two AI assistants for Tony to mutate a prompt by using unconventional and abstract thinking. The initial prompt is: \"[{problem}]\".\n\n{self.role} and {self.other_role} will engage in multi-round dialogue to mutate the prompt for an evolutionary algorithm. The mutated prompt has to be within brackets."
        })
        self.history.append({
            "role": "assistant",
            "content": f"Hi {self.other_role}, let's work together to mutate the prompt by using unconventional and abstract thinking. Please feel free to suggest mutations or improvements. The mutated prompt has to be within brackets."
        })

    def get_response(self):
        input_prompt = "".join([f"{msg['role']}: {msg['content']}" for msg in self.history])
        input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids
        outputs = model.generate(
            input_ids,
            max_length=512,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        response_ids = outputs[0]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
        self.history.append({
            "role": "assistant",
            "content": response_text
        })
        return response_text

def main():
    # Initialize SocraticGPT instances
    socrates = SocraticGPT(role="Socrates")
    theaetetus = SocraticGPT(role="Theaetetus")

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

        # Mutate the prompt based on responses
        # mutated_prompt = random.choice([socrates_response, theaetetus_response])
        # print(f"Mutated Prompt: {mutated_prompt}")

    end = time.time()
    print(f"Round lasted: {end-start}")

    # Print the final mutated prompt
    final_prompt = socrates_response  

    matches = re.findall(r'\[([^]]+)\]', final_prompt)

    if matches:
        last_brackets_text = matches[-1]
        print("Final mutated prompt:", last_brackets_text)
    else:
        print("No text within brackets found.")

if __name__ == '__main__':
    main()
