import json
from six import iteritems
import torch
from transformers import AutoTokenizer, AutoModel, GPT2Tokenizer, GPT2Model
import scipy
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
from allennlp.modules.elmo import Elmo, batch_to_ids

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

def get_elmo_correlation(elmo, dataset):
    sim=[]
    for val in dataset['X']:
        character_ids = batch_to_ids(val)

        embeddings = elmo(character_ids)

        # extract the ELMo embeddings for the two sentences
        emb1 = embeddings['elmo_representations'][0][0]
        emb2 = embeddings['elmo_representations'][0][1]

        # calculate cosine similarity between the two ELMo embeddings
        cosine_sim = F.cosine_similarity(emb1, emb2, dim=0)
        mean_cosine_sim = torch.mean(cosine_sim)
        sim.append(mean_cosine_sim.item())

    return scipy.stats.spearmanr(sim, dataset['y']).correlation

def get_elmo_benchmarks(datasets):
    # Calculate results using helper function
    options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json"
    weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"

    elmo = Elmo(options_file, weight_file, 2, dropout=0)

    for data_name in datasets.keys():
        print('Dataset size:',len(datasets[data_name]['X']))
        print("Spearman correlation of scores on {} {}".format(
            data_name, get_elmo_correlation(elmo, datasets[data_name]))
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
    get_elmo_benchmarks(datasets)


if __name__ == '__init__':
    main()