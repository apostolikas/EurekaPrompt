from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch
from my_utils import *
from prompts import *
import evaluate
import numpy as np
import re
import torch.nn.functional as F

import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from tqdm import tqdm

class Perplexity:
    def __init__(self, model, tokenizer, batch_size=16, add_start_token=True, device=None, max_length=None):
        if device is not None:
            assert device in ["gpu", "cpu", "cuda"], "device should be either gpu or cpu."
            if device == "gpu":
                device = "cuda"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = model
        self.model = self.model.to(device)
        self.tokenizer = tokenizer

        if self.tokenizer.pad_token is None and batch_size > 1:
            existing_special_tokens = list(self.tokenizer.special_tokens_map_extended.values())
            assert len(existing_special_tokens) > 0, "If batch_size > 1, model must have at least one special token to use for padding. Please use a different model or set batch_size=1."
            self.tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

        self.batch_size = batch_size
        self.add_start_token = add_start_token
        self.max_length = max_length
        self.device = device

    def compute(self, predictions):
        if self.add_start_token and self.max_length:
            assert self.tokenizer.bos_token is not None, "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
            max_tokenized_len = self.max_length - 1
        else:
            max_tokenized_len = self.max_length

        encodings = self.tokenizer(
            predictions,
            add_special_tokens=False,
            padding=True,
            truncation=True if max_tokenized_len else False,
            max_length=max_tokenized_len,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(self.device)

        encoded_texts = encodings["input_ids"]
        attn_masks = encodings["attention_mask"]

        if self.add_start_token:
            assert torch.all(torch.ge(attn_masks.sum(1), 1)), "Each input text must be at least one token long."
        else:
            assert torch.all(torch.ge(attn_masks.sum(1), 2)), "When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings."

        ppls = []
        loss_fct = CrossEntropyLoss(reduction="none")

        for start_index in tqdm(range(0, len(encoded_texts), self.batch_size)):
            end_index = min(start_index + self.batch_size, len(encoded_texts))
            encoded_batch = encoded_texts[start_index:end_index]
            attn_mask = attn_masks[start_index:end_index]

            if self.add_start_token:
                bos_tokens_tensor = torch.tensor([[self.tokenizer.bos_token_id]] * encoded_batch.size(dim=0)).to(self.device)
                encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
                attn_mask = torch.cat(
                    [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(self.device), attn_mask], dim=1
                )

            labels = encoded_batch

            with torch.no_grad():
                out_logits = self.model(encoded_batch, attention_mask=attn_mask).logits
                                        
            shift_logits = out_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

            perplexity_batch = torch.exp(
                (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
                / shift_attention_mask_batch.sum(1)
            )

            ppls += perplexity_batch.tolist()

        return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}


def cosine_similarity(a, b):
    return torch.dot(a, b) / (torch.norm(a) * torch.norm(b))

def compute_average_similarity(response_embedding, cluster_embeddings):
    similarities = [cosine_similarity(response_embedding, emb) for emb in cluster_embeddings]
    return sum(similarities) / len(similarities)

def all_equal(final_result, cluster_final_results):
    return all(fr == final_result for fr in cluster_final_results)

def cluster_responses(response_embeddings, final_results, similarity_threshold):
    clusters = []
    for i in range(len(response_embeddings)):
        response_embedding = response_embeddings[i]
        final_result = final_results[i]
        cluster_found = False
        for cluster in clusters:
            cluster_indices = [idx for idx, _, _ in cluster]
            cluster_embeddings = [response_embeddings[idx] for idx in cluster_indices]
            cluster_final_results = [final_results[idx] for idx in cluster_indices]
            avg_similarity = compute_average_similarity(response_embedding, cluster_embeddings)
            if avg_similarity > similarity_threshold and all_equal(final_result, cluster_final_results):
                cluster.append((i, response_embedding, final_result))
                cluster_found = True
                break
        if not cluster_found:
            new_cluster = [(i, response_embedding, final_result)]
            clusters.append(new_cluster)
    return clusters


def create_model_input(task, question, instruction, choices, narrative):

    if task in ['gsm8k', 'svamp']:
        model_input = f"GPT4 Correct User: Q:{question}\nA: {instruction}<|end_of_turn|>GPT4 Correct Assistant:"

    elif task in ['csqa', 'social_iqa', 'sports_und', 'date_under', 'causal_judg']:
        input_prompt = f'''Question: {question}\nAnswer Choices: {choices}\nAnswer: {instruction}'''
        model_input = f'''GPT4 Correct User: {input_prompt}<|end_of_turn|>GPT4 Correct Assistant:'''

    elif task == 'abs_nar':
        input_prompt = f'''Question: Can you choose the most related proverb from the list of 5 proverbs given a narrative?\nNarrative: {narrative}\nAnswer choices: {choices}\nAnswer: {instruction}'''
        model_input = f'''GPT4 Correct User: {input_prompt}<|end_of_turn|>GPT4 Correct Assistant:'''

    elif task == 'disamb':
        input_prompt = f'''Question: Can you clarify the meaning of the sentence with ambiguous pronouns?\nSentence: {question}\nAnswer choices: {choices}\nAnswer: {instruction}'''
        model_input = f'''GPT4 Correct User: {input_prompt}<|end_of_turn|>GPT4 Correct Assistant:'''

    return model_input

def generate_response(task, decode_strategy, model, tokenizer, question, instruction, choices=None, narrative=None):

    model_input = create_model_input(task, question, instruction, choices, narrative)
    
    inputs = tokenizer(model_input, return_tensors="pt").to('cuda')

    if decode_strategy == 'greedy':
        outputs = model.generate(**inputs, do_sample = False, num_beams = 1, max_new_tokens = 700, pad_token_id = tokenizer.pad_token_id, eos_token_id = tokenizer.eos_token_id)
        generated_text= tokenizer.batch_decode(outputs, skip_special_tokens=True)

    elif decode_strategy == 'contrastive_search':
        outputs = model.generate(**inputs, penalty_alpha=0.6, top_k=4, max_new_tokens = 700, pad_token_id = tokenizer.pad_token_id, eos_token_id = tokenizer.eos_token_id)
        generated_text= tokenizer.batch_decode(outputs, skip_special_tokens=True)

    elif decode_strategy == 'multinomial_sampling':
        outputs = model.generate(**inputs, do_sample=True, num_beams=1, max_new_tokens = 700, pad_token_id = tokenizer.pad_token_id, eos_token_id = tokenizer.eos_token_id)
        generated_text= tokenizer.batch_decode(outputs, skip_special_tokens=True)

    elif decode_strategy == 'beam_search':
        outputs = model.generate(**inputs, num_beams=5, do_sample = False, max_new_tokens = 700, pad_token_id = tokenizer.pad_token_id, eos_token_id = tokenizer.eos_token_id)
        generated_text= tokenizer.batch_decode(outputs, skip_special_tokens=True)

    elif decode_strategy == 'beam_search_with_multinomial_sampling':
        outputs = model.generate(**inputs, num_beams=5, do_sample=True, max_new_tokens = 700, pad_token_id = tokenizer.pad_token_id, eos_token_id = tokenizer.eos_token_id)
        generated_text= tokenizer.batch_decode(outputs, skip_special_tokens=True)

    elif decode_strategy == 'top_k_sampling':
        outputs = model.generate(**inputs, do_sample=True, top_k=50, max_new_tokens = 700, pad_token_id = tokenizer.pad_token_id, eos_token_id = tokenizer.eos_token_id)
        generated_text= tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    elif decode_strategy == 'top_p_sampling':
        outputs = model.generate(**inputs, do_sample=True, top_p=0.9, max_new_tokens = 700, pad_token_id = tokenizer.pad_token_id, eos_token_id = tokenizer.eos_token_id)
        generated_text= tokenizer.batch_decode(outputs, skip_special_tokens=True)

    elif decode_strategy == 'sampling0.25':
        outputs = model.generate(**inputs, do_sample=True, top_k = 0, max_new_tokens = 700, pad_token_id = tokenizer.pad_token_id, eos_token_id = tokenizer.eos_token_id, temperature=0.25)
        generated_text= tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    elif decode_strategy == 'sampling0.5':
        outputs = model.generate(**inputs, do_sample=True, top_k = 0, max_new_tokens = 700, pad_token_id = tokenizer.pad_token_id, eos_token_id = tokenizer.eos_token_id, temperature=0.5)
        generated_text= tokenizer.batch_decode(outputs, skip_special_tokens=True)

    elif decode_strategy == 'sampling0.75':
        outputs = model.generate(**inputs, do_sample=True, top_k = 0, max_new_tokens = 700, pad_token_id = tokenizer.pad_token_id, eos_token_id = tokenizer.eos_token_id, temperature=0.75)
        generated_text= tokenizer.batch_decode(outputs, skip_special_tokens=True)

    else:
        raise ValueError("Invalid decoding strategy")
    
    return generated_text, model_input

if __name__ == '__main__':

    tokenizer = AutoTokenizer.from_pretrained("berkeley-nest/Starling-LM-7B-alpha")
    model = AutoModelForCausalLM.from_pretrained("berkeley-nest/Starling-LM-7B-alpha", torch_dtype=torch.bfloat16).to('cuda')
    decode_strategies = ["greedy", "contrastive_search", "multinomial_sampling", "beam_search", "beam_search_with_multinomial_sampling", "top_k_sampling", "top_p_sampling", "sampling0.25", "sampling0.5", "sampling0.75"]
    sentence_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    sentence_model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2').to('cuda')
    # sentence_model = sentence_model
    # perplexity = evaluate.load("perplexity", module_type="metric")
    perplexity = Perplexity(model=model, tokenizer=tokenizer, batch_size=16, add_start_token=True, device="cuda", max_length=None)

    tasks = ['abs_nar']#, 'csqa', 'abs_nar', 'causal_judg', 'social_iqa', 'date_under', 'sports_und']

    for task in tasks:

        _, testset = load_data(task)
        num_of_samples = 100 #len(testset) / 4
        prompts = load_inference_prompts(task)
        random.seed(0)
        samples = random.sample(testset, num_of_samples)
        file_name = f"./relations/relation_{task}1.txt"

        with open(file_name, 'w') as f:

            for prompt in prompts:
                f.write(f"Prompt: {prompt}\n")
                entropy_list = []
                accuracy_list = []
                perplexity_list = []

                for i, sample in enumerate(samples):

                    responses = []
                    final_results = []

                    for decode_strategy in decode_strategies:
                        if task == 'gsm8k':
                            question = sample['question']
                            label = sample['answer'].split('#### ')[1]
                            response_text, model_input = generate_response(task, decode_strategy, model, tokenizer, question, prompt)

                        elif task == 'svamp':
                            question = sample['full_question']
                            label = sample['Answer']
                            response_text, model_input = generate_response(task, decode_strategy, model, tokenizer, question, prompt)

                        elif task == 'csqa':
                            question = sample['question']['stem']
                            choices = sample['choice_answers']
                            label = sample['answerKey']
                            response_text, model_input = generate_response(task, decode_strategy, model, tokenizer, question, prompt, choices)

                        elif task == 'abs_nar':
                            narrative = sample['input']
                            choices = sample['answer_choices']
                            label = sample['label']
                            response_text, model_input = generate_response(task, decode_strategy, model, tokenizer, question = None, instruction = prompt, choices = choices, narrative = narrative)

                        elif task == 'disamb':
                            question = sample['input']
                            choices = sample['answer_choices']
                            label = sample['label']
                            response_text, model_input = generate_response(task, decode_strategy, model, tokenizer, question = question, instruction = prompt, choices = choices)

                        elif task in ['causal_judg', 'social_iqa', 'date_under', 'sports_und']:
                            question = sample['input']
                            answer_choices = sample['answer_choices']
                            label = sample['label']
                            response_text, model_input = generate_response(task, decode_strategy, model, tokenizer, question = question, instruction = prompt , choices = answer_choices, narrative = None)
                        # print(f"Reponse text is {type(response_text)}")
                        # print(response_text)
                        responses.append(response_text[0])
                        final_result = extract_final_results(task, response_text[0], label)
                  
                        final_results.append(final_result)

                    results = perplexity.compute(predictions=model_input)
                    # results = perplexity.compute(model_id="berkeley-nest/Starling-LM-7B-alpha",
                    #                             add_start_token=False,
                    #                             predictions=model_input)
                    ppl = results['mean_perplexity']
                    encoded_input = sentence_tokenizer(responses, padding=True, truncation=True, return_tensors='pt').to('cuda')
                    with torch.no_grad():
                        model_output = sentence_model(**encoded_input)
                    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
                    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
                    
                    similarity_threshold = 0
                    clusters = cluster_responses(sentence_embeddings, final_results, similarity_threshold)
                    cluster_counts = [len(cluster) for cluster in clusters]
                    entropy = calculate_entropy(cluster_counts)
                    accuracy_of_responses = sum([1 for fr in final_results if str(fr) == str(label)]) / len(final_results)

                    print(f"Sample: {i} | Entropy: {entropy} | Accuracy: {accuracy_of_responses} | Perplexity: {ppl}")
                    f.write(f"Sample: {i} | Entropy: {entropy} | Accuracy: {accuracy_of_responses} | Perplexity: {ppl}\n")

                    entropy_list.append(entropy)
                    accuracy_list.append(accuracy_of_responses)
                    perplexity_list.append(ppl)

                print(f"Prompt: {prompt}")
                print(f"Average entropy: {np.mean(entropy_list)}")
                print(f"Average accuracy: {np.mean(accuracy_list)}")
                print(f"Average perplexity: {np.mean(perplexity_list)}")
                print()
                f.write(f"Average entropy: {np.mean(entropy_list)}\n")
                f.write(f"Average accuracy: {np.mean(accuracy_list)}\n")
                f.write(f"Average perplexity: {np.mean(perplexity_list)}\n")
                f.write("--------------------------------------------\n")
                

                




