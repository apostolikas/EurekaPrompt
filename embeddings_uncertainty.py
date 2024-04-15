from transformers import AutoTokenizer, AutoModel
import torch
from my_utils import *
from prompts import *
import torch.nn.functional as F

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



if __name__ == '__main__':

    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    decode_strategies = ["greedy", "contrastive_search", "multinomial_sampling", "beam_search", "beam_search_with_multinomial_sampling", "top_k_sampling", "top_p_sampling", "sampling0.25", "sampling0.5", "sampling0.75"]

    task = 'gsm8k'
    num_of_samples = 50
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
            label = sample['answer'].split('#### ')[1]
            result = acc_dict[prompt][k]
            responses = []
            final_results = []
            for decode_strategy in decode_strategies:
                response = answer_dict[prompt][decode_strategy][f'{k}']
                final_result = extract_final_results(task, response)
                final_results.append(final_result)            
                responses.append(response)

            encoded_input = tokenizer(responses, padding=True, truncation=True, return_tensors='pt')
            with torch.no_grad():
                model_output = model(**encoded_input)
            sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
            
            # Cluster responses based on similarity and final results
            clusters = cluster_responses(sentence_embeddings, final_results, similarity_threshold)
            
            # Print the clusters and their final results
            for i, cluster in enumerate(clusters):
                print(f"Cluster {i+1}:")
                for idx, _, final_result in cluster:
                    print(f"  Response Index: {idx}")
                    print(f"  Response: {responses[idx]}")
                    print(f"  Final Result of the response: {final_result}")
                    
                print()

            # Calculate entropy for the clusters
            cluster_counts = [len(cluster) for cluster in clusters]
            entropy = calculate_entropy(cluster_counts)
            entropy_list.append((i, entropy))
            print(f"Sample: {k} | Inference eval: {result} | Entropy: {entropy} | Acc of responses: {sum([1 for fr in final_results if str(fr) == label]) / len(final_results)}\n")
            print("--------------------------------------------------")

        entropy_dict[prompt] = entropy_list
        average_entropy = sum([entropy for _, entropy in entropy_list]) / len(entropy_list)
        print(f"Average entropy for prompt {prompt}: {average_entropy}")
        print()
