from transformers import AutoTokenizer, AutoModel
import torch
from my_utils import *
from prompts import *
import torch.nn.functional as F

if __name__ == '__main__':

    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    decode_strategies = ["greedy", "contrastive_search", "multinomial_sampling", "beam_search", "beam_search_with_multinomial_sampling", "top_k_sampling", "top_p_sampling", "sampling0.25", "sampling0.5", "sampling0.75"]

    task = 'gsm8k'
    num_of_samples = 5
    prompts = load_inference_prompts(task)
    filename = f'./decoded/decode_results_{task}.txt'
    answer_dict = extract_responses(filename, 'starling')
    acc_filename = f"./inference_logs/{task}_answers.txt"

    acc_dict = read_answers(acc_filename)
    for prompt in acc_dict:
        acc_dict[prompt] = acc_dict[prompt][:num_of_samples]

    _, testset = load_data(task)

    random.seed(0)
    samples = random.sample(testset, num_of_samples)

    entropy_dict = {}
    similarity_threshold = 0.95

    for prompt in prompts:
        entropy_list = []
        for k, sample in enumerate(samples):
            result = acc_dict[prompt][k]
            responses = []
            for decode_strategy in decode_strategies:
                responses.append(answer_dict[prompt][decode_strategy][f'{k}'])

            encoded_input = tokenizer(responses, padding=True, truncation=True, return_tensors='pt')
            with torch.no_grad():
                model_output = model(**encoded_input)
            sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

            clusters = []
            for i in range(len(responses)):
                if not clusters:
                    clusters.append([i])
                else:
                    max_similarity = -1
                    max_cluster_index = -1
                    for j, cluster in enumerate(clusters):
                        cluster_embedding = torch.mean(sentence_embeddings[cluster], axis=0)
                        similarity = F.cosine_similarity(sentence_embeddings[i].unsqueeze(0), cluster_embedding.unsqueeze(0))[0]
                        if similarity > max_similarity:
                            max_similarity = similarity
                            max_cluster_index = j

                    if max_similarity >= similarity_threshold:
                        clusters[max_cluster_index].append(i)
                    else:
                        clusters.append([i])

            cluster_counts = [len(cluster) for cluster in clusters]
            entropy = calculate_entropy(cluster_counts)
            entropy_list.append((i, entropy))

            print(f"Clusters for sample {k}:")
            for i, cluster in enumerate(clusters):
                print(f"Cluster {i+1} | {len(cluster)} responses | {[responses.index(responses[j]) for j in cluster]}")
            print(f"Sample: {k} | Result: {result} | Entropy: {entropy}\n")

        entropy_dict[prompt] = entropy_list
        average_entropy = sum([entropy for _, entropy in entropy_list]) / len(entropy_list)
        print(f"Average entropy for prompt {prompt}: {average_entropy}")