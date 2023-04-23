import json

import scipy
import torch
import numpy as np
from transformers import BertTokenizer, BertModel, BertConfig
from sklearn.metrics.pairwise import cosine_similarity



def get_bert_correlation_layer_wise(model, tokenizer, dataset, n_layer):

    sim=[]
    for val in dataset['X']:
        sentence1, sentence2 = val

        tokens1 = ['[CLS]'] + tokenizer.tokenize(sentence1) + ['[SEP]']
        tokens2 = ['[CLS]'] + tokenizer.tokenize(sentence2) + ['[SEP]']
        token_ids1 = tokenizer.convert_tokens_to_ids(tokens1)
        token_ids2 = tokenizer.convert_tokens_to_ids(tokens2)
        input_ids1 = torch.tensor([token_ids1])
        input_ids2 = torch.tensor([token_ids2])

        with torch.no_grad():
            outputs1 = model(input_ids1)
            outputs2 = model(input_ids2)

        embeddings1 = outputs1['hidden_states'][n_layer].squeeze(0).numpy()
        embeddings2 = outputs2['hidden_states'][n_layer].squeeze(0).numpy()

        avg_embeddings1 = np.mean(embeddings1, axis=0)
        avg_embeddings2 = np.mean(embeddings2, axis=0)

        similarity = cosine_similarity([avg_embeddings1], [avg_embeddings2])
        sim.append(similarity[0][0])
        
    return scipy.stats.spearmanr(sim, dataset['y']).correlation


def bert_layers_benchmark(datasets):

    config = BertConfig.from_pretrained("bert-base-uncased", output_hidden_states=True)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased', config=config)

    for n_layer in [1,2,11,12]:
        print(f"Layer {n_layer}")
        
        dataset_names = ['TR9856', 'MEN', 'SimLex999']
        for data_name in dataset_names:
            print('Dataset size:',len(datasets[data_name]['X']))
            print("Spearman correlation of scores on {} {}".format(
                data_name, get_bert_correlation_layer_wise(model, tokenizer, datasets[data_name], n_layer))
                )

def main():
    with open('all_datasets.json', 'rb') as file:
        datasets = json.load(file)
    bert_layers_benchmark(datasets)


if __name__ == '__init__':
    main()