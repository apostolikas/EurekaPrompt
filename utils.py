import random
import re
from prompts.math_prompts import *
from nltk import PorterStemmer

def add_label(entry):

    ans_value = f'''
        {entry['answer'].split("####")[0]} The answer is {int(entry['answer'].split('####')[1].replace(',', '',))}
    '''
    entry['ans'] = ans_value
    answer_value = int(entry['answer'].split('####')[1].replace(',', ''))
    entry['label'] = answer_value
    
    return entry


def construct_icl_examples(samples, sample_prompts, number_of_examples):

    icl_examples = random.sample(list(samples), number_of_examples)
    icl_thinking_styles = random.sample(sample_prompts, number_of_examples)
    icl_prompt = f''''''

    for i in range(len(icl_examples)):

        icl_prompt += f'''Question: {icl_examples[i]['question']}
{icl_thinking_styles[i]}
{icl_examples[i]['ans']}
'''
    return icl_prompt



def evaluate_GSM8K(y_pred, label):
    pattern = r"\[Answer\]:\s*(.*?)\n(\[Question\]:|$)"
    match = re.search(pattern, y_pred)
    if match:
        y_pred = match.group(1)
    pred = y_pred.replace(",", "")
    pred = [s for s in re.findall(r"-?\d+\.?\d*", pred)]
    if pred == []:
        return 0
    pred = pred[-1]
    pred = pred.replace(",", "").replace(".", "").replace(" ", "")
    if int(pred) == int(label):
        return 1
    else:
        return 0



def construct_contrastive_icl_example(samples, number_of_examples):

    wrong_samples = random.sample(samples, number_of_examples)

    contrastive_prompt = f''''''

    for i in range(len(wrong_samples)):

        contrastive_prompt += f'''
Question: {wrong_samples[i]['question']}
Explanation: {wrong_samples[i]['explanation']}
Answer: {wrong_samples[i]['answer']}
Wrong explanation: {wrong_samples[i]['wrong_explanation']}
Wrong answer: {wrong_samples[i]['wrong_answer']}
        '''
    
    return contrastive_prompt



def evaluate_nli(y_pred:str, label:str) -> int:
    stemmer = PorterStemmer()
    stemmed_y_pred = stemmer.stem(y_pred)
    stemmed_label = stemmer.stem(label)

    return 1 if stemmed_label in stemmed_y_pred else 0