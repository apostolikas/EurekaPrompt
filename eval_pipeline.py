import transformers
import re
import time

class SocraticGPT:
    def __init__(self, role, model, tokenizer, task, n_round=1):
        self.role = role
        self.tokenizer = tokenizer
        self.model = model
        self.n_round = n_round
        self.other_role = "Theaetetus" if role == "Socrates" else "Socrates"
        self.task = task
        self.history = []

    def set_problem(self, gen_text, label):
        self.history.append({
            "role": "system",
            # "content": f"{self.role} and {self.other_role} are two AI assistants for Tony to evaluate another AI assistant on a {self.task} task. {self.role} and {self.other_role} will engage in multi-round dialogue to determine whether what the AI assistant generated is the same as the true label. The assistant's output is '{gen_text}' and the true label is '{label}'. The final answer has to be a yes or no and must be within brackets."
            # "content": f'{self.role} and {self.other_role} are two AI assistants for Tony to evaluate another AI assistant on a {self.task} task. {self.role} and {self.other_role} will engage in multi-round dialogue to determine whether what some text is semantically equivalent to the label. The text is "{gen_text}". The label is "{label}". The answer is either a YES or NO and must be within brackets.'
            "content": f"{self.role} and {self.other_role} are two AI assistants for Tony to help him on a {self.task} task. {self.role} and {self.other_role} will engage in multi-round dialogue to determine whether what the AI assistant generated (premise) is the same as the true label (hypothesis). The assistant's output is '{gen_text}' and the true label is '{label}'. The final answer has to be 1 if they premise entails the hypothesis or 0 if the premise contradicts the hypotehsis. The arabic numeral must be within brackets."

        })
        self.history.append({
            "role": "assistant",
            "content": f"\nHi {self.other_role}, let's work together to determine if the text given is the same as the label. Both of us have to say and justify our opinion, but the final answer must be 1 or 0 has to be within brackets."
        })

    def get_response(self):
        input_prompt = "".join([f"{msg['role']}: {msg['content']}" for msg in self.history])
        input_ids = self.tokenizer(input_prompt, return_tensors="pt").input_ids.to('cuda')
        outputs = self.model.generate(
            input_ids,
            max_length=750,
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

def dialogue(model, tokenizer, task, generated_text, label):

    socrates = SocraticGPT(role="Socrates", model=model, tokenizer=tokenizer, task=task)
    theaetetus = SocraticGPT(role="Theaetetus", model=model, tokenizer=tokenizer, task=task)

    socrates.set_problem(generated_text, label)
    theaetetus.set_problem(generated_text, label)

    start = time.time()

    for round_num in range(socrates.n_round):

        print(f"Round {round_num+1}:")

        socrates_response = socrates.get_response()
        print(f"{socrates.role}: {socrates_response}")

        theaetetus_response = theaetetus.get_response()
        print(f"{theaetetus.role}: {theaetetus_response}")

        if "[1]" or "[0]" in theaetetus_response.lower() or "final" in theaetetus_response.lower():
            print("Ending the dialogue.")
            break


    end = time.time()
    print(f"Time taken: {end-start:.2f}s")

    final_answer = socrates_response  
    final_answer = re.findall(r'"([^"]*)"', final_answer)
    final_answer = [prompt.replace('"',"'").replace("[","").replace("]","").replace("!",".") for prompt in final_answer]

    return final_answer

    
if __name__ == '__main__':

    file_path = './Starling_Embase_NLI_0shot_instruct_seed_0.txt'

with open(file_path, 'r') as file:
    # Read the entire content of the file
    file_content = file.read()

# Split the text into examples using '---'
examples = file_content.split('---\n')

# Remove the last --- from the last example
examples[-1] = examples[-1].strip('---\n')

# Lists to store labels and text_after_answer
labels = []
text_after_answer = []

# Process each example
for example in examples:
    # Find the position of "Answer:"
    answer_pos = example.find("Answer:")

    if answer_pos != -1:
        # Extract the text after "Answer:"
        answer_text = example[answer_pos + len("Answer:"):].strip()

        # Exclude the line with "entailment" from answer_text
        answer_lines = answer_text.split('\n')
        last_line = answer_lines[-1].strip()

        # Remove the last line from answer_text
        answer_lines = answer_lines[:-1]
        answer_text = '\n'.join(answer_lines)

        # Split the text into lines and get the last word as label
        lines = answer_text.split('\n')
        label = last_line

        # Append label and text_after_answer to the respective lists
        labels.append(label)
        text_after_answer.append(answer_text)



    task = "Natural Language Inference"

    tokenizer = transformers.AutoTokenizer.from_pretrained("berkeley-nest/Starling-LM-7B-alpha")
    model = transformers.AutoModelForCausalLM.from_pretrained("berkeley-nest/Starling-LM-7B-alpha")
    model = model.to('cuda')


    for i in range(len(text_after_answer)):

        print(f"Example {i+1}:")

        generated_text = text_after_answer[i]
        label = labels[i]

        final_answer = dialogue(model, tokenizer, task, generated_text, label)

        print(f"Result of the dialogue: {final_answer}\n")

        break


    