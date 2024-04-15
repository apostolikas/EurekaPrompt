from statsmodels.stats.contingency_tables import mcnemar
from scipy import stats

def read_answers(filename):
    data = {}
    with open(filename, 'r') as file:
        lines = file.readlines()
        current_prompt = None
        for line in lines:
            line = line.strip()
            if line.startswith('Prompt:'):
                current_prompt = line.split('Prompt: ')[1]
                data[current_prompt] = []
            elif line.startswith('Question:'):
                _, result = line.split('Result: ')
                if current_prompt:
                    data[current_prompt].append(int(result))
    return data


def calculate_contingency_table(prompt1_results, prompt2_results):
    assert len(prompt1_results) == len(prompt2_results)
    both_correct = both_incorrect = prompt1_correct = prompt2_correct = 0
    for r1, r2 in zip(prompt1_results, prompt2_results):
        if r1 == 1 and r2 == 1:
            both_correct += 1
        elif r1 == 0 and r2 == 0:
            both_incorrect += 1
        elif r1 == 1 and r2 == 0:
            prompt1_correct += 1
        elif r1 == 0 and r2 == 1:
            prompt2_correct += 1
    return [[both_correct, prompt1_correct], [prompt2_correct, both_incorrect]]

if __name__ == '__main__':

    tasks = ['gsm8k', 'svamp', 'csqa', 'abs_nar', 'date_under', 'causal_judg', 'social_iqa', 'sports_und']

    for task in tasks:

        file_name = f'./inference_logs/{task}_answers.txt'

        print(f"Analyzing task: {task}...")

        data = read_answers(file_name)
        prompts = list(data.keys())
        best_prompt = prompts[-1]
        prompts = prompts[:-1]
        alpha = 0.05

        print("Performing McNermar's test")

        for prompt in prompts:
        
            print(f"Comparing prompts '{prompt}' and '{best_prompt}'")
            contingency_table = calculate_contingency_table(data[prompt], data[best_prompt])
            result = mcnemar(contingency_table, correction=True)
            print(f"McNemar test between '{prompt}' and '{best_prompt}':")
            print(f"Test statistic: {result.statistic}")
            print(f"p-value: {result.pvalue}")
            print()

            if result.pvalue < alpha:
                print("The null hypothesis is rejected. There is a significant difference between the correct and incorrect prediction counts for different prompts.")
            else:
                print("The null hypothesis cannot be rejected. There is no significant difference between the correct and incorrect prediction counts for different prompts.")
            print(" ")

        print("Performing t-test")

        for prompt in prompts:
            print(f"Comparing prompts '{prompt}' and '{best_prompt}'")
            t_stat, p_val = stats.ttest_ind(data[prompt], data[best_prompt])
            print(f"t-statistic is: {t_stat}")
            print(f"p-value is: {p_val}")
            if p_val < alpha:
                print("The null hypothesis is rejected. There is a significant difference between the correct and incorrect prediction counts for different prompts.")
            else:
                print("The null hypothesis cannot be rejected. There is no significant difference between the correct and incorrect prediction counts for different prompts.")
            print(" ")

        print(f"Finished analyzing task: {task}!\n")
