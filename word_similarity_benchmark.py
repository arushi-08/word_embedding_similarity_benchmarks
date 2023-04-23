import json
from six import iteritems
import torch
from transformers import AutoTokenizer, AutoModel, GPT2Tokenizer, GPT2Model
import scipy
from sklearn.metrics.pairwise import cosine_similarity

from data import save_word_similarity_dataset
from static_embedding import StaticEmbeddings


def compare_model_human_correlation(model, tokenizer, dataset):
    sim=[]
    for val in dataset['X']:
        word_list = val
        embeddings = []
        for word in word_list:
            input_ids = tokenizer.encode(word, add_special_tokens=False, return_tensors='pt')
            with torch.no_grad():
                outputs = model(input_ids)
            last_hidden_states = outputs.last_hidden_state
            embedding = last_hidden_states[0][0]
            embeddings.append(embedding)
        similarity = cosine_similarity(embeddings[0].reshape(1,-1), embeddings[1].reshape(1,-1))
        sim.append(similarity[0][0])
    return scipy.stats.spearmanr(sim, dataset['y']).correlation


def bert_similarity_benchmark(datasets):
    """ pre-trained BERT model compute cosine similarity then correlation with human annotation """

    tokenizer = AutoTokenizer.from_pretrained(model_name='bert-base-uncased')
    model = AutoModel.from_pretrained(model_name='bert-base-uncased')

    for data_name in datasets.keys():
        print('Dataset size:',len(datasets[data_name]['X']))
        print("Spearman correlation of scores on {} {}".format(
            data_name, compare_model_human_correlation(model, tokenizer, datasets[data_name]))
            )
    
    
def gpt2_similarity_benchmark(datasets):
    """ pre-trained GPT2 model compute cosine similarity then correlation with human annotation """

    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    gpt2_model = GPT2Model.from_pretrained('gpt2', output_hidden_states=True)
    
    for data_name in datasets.keys():
        print('Dataset size:',len(datasets[data_name]['X']))
        print("Spearman correlation of scores on {} {}".format(
            data_name, compare_model_human_correlation(gpt2_model, gpt2_tokenizer, datasets[data_name]))
            )

def static_similarity_benchmark(model, datasets):
    for name, data in iteritems(datasets):
        print(
            "Spearman correlation of scores on {} {}".format(
                name, evaluate_similarity(model, data.X, data.y)
            )
        )


def main():
    save_word_similarity_dataset()

    with open('all_datasets.json', 'r') as file:
        datasets = json.load(file)

    se = StaticEmbeddings()

    word2vec = se.load_word2vec()
    static_similarity_benchmark(word2vec, datasets)
    glove = se.load_glove()
    static_similarity_benchmark(glove, datasets)
    fasttext = se.load_fasttext()
    static_similarity_benchmark(fasttext, datasets)

    bert_similarity_benchmark(datasets)
    gpt2_similarity_benchmark(datasets)


if __name__ == '__init__':
    main()