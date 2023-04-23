import math
import functools as ft
from collections import Counter
import util
import torch
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModel, GPT2Model, GPT2Tokenizer

from data import Sentence, doc_frequencies
from sentence_similarity import download_and_load_sts_data, download_sick
from static_embedding import StaticEmbeddings



def run_avg_benchmark(sentences1, sentences2, model=None, use_stoplist=False, doc_freqs=None): 

    if doc_freqs is not None:
        N = doc_freqs["NUM_DOCS"]
    
    sims = []
    for (sent1, sent2) in zip(sentences1, sentences2):
        tokens1 = sent1.tokens_without_stop if use_stoplist else sent1.tokens
        tokens2 = sent2.tokens_without_stop if use_stoplist else sent2.tokens
        tokens1 = [token for token in tokens1 if token in model]
        tokens2 = [token for token in tokens2 if token in model]
        if len(tokens1) == 0 or len(tokens2) == 0:
            sims.append(0)
            continue
        tokfreqs1 = Counter(tokens1)
        tokfreqs2 = Counter(tokens2)
        weights1 = [tokfreqs1[token] * math.log(N/(doc_freqs.get(token, 0)+1)) 
                    for token in tokfreqs1] if doc_freqs else None
        weights2 = [tokfreqs2[token] * math.log(N/(doc_freqs.get(token, 0)+1)) 
                    for token in tokfreqs2] if doc_freqs else None
        embedding1 = np.average([model[token] for token in tokfreqs1], axis=0, weights=weights1).reshape(1, -1)
        embedding2 = np.average([model[token] for token in tokfreqs2], axis=0, weights=weights2).reshape(1, -1)
        sim = cosine_similarity(embedding1, embedding2)[0][0]
        sims.append(sim)

    return sims


def run_transformer_benchmark(sentence_1, sentence_2, model, tokenizer):
    
    embedding_1, embedding_2 = [], []
    for sent1, sent2 in zip(sentence_1, sentence_2):
        
        # tokenize the sentences and convert them into PyTorch tensors
        inputs_1 = tokenizer(sent1, padding=True, truncation=True, return_tensors="pt")
        inputs_2 = tokenizer(sent2, padding=True, truncation=True, return_tensors="pt")

        # generate the embeddings for the sentences
        with torch.no_grad():
            outputs_1 = model(**inputs_1)
            outputs_2 = model(**inputs_2)

        # extract the embeddings for the [CLS] token for each sentence
        embedding_1.append(outputs_1.last_hidden_state[:, 0, :].numpy())
        embedding_2.append(outputs_2.last_hidden_state[:, 0, :].numpy())
    
    sims = []
    for (emb1, emb2) in zip(embedding_1, embedding_2): 
        sim = cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]
        sims.append(sim)
        
    return sims


def run_sentence_transformer_benchmarks(sentences_1, sentences_2, model):
    embeddings1, embeddings2 = [], []
    sims = []
    for sent1, sent2 in zip(sentences_1, sentences_2):
        embeddings1 = model.encode(sent1, convert_to_tensor=True)
        embeddings2 = model.encode(sent2, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(embeddings1, embeddings2)
        sims.append(similarity.item())
        
    return sims

def run_experiment(df, benchmarks): 
    
    sentences1 = [Sentence(s) for s in df['sent_1']]
    sentences2 = [Sentence(s) for s in df['sent_2']]
    
    pearson_cors, spearman_cors = [], []
    for label, method in benchmarks:
        if "BERT" in label:
            sentences1 = [s for s in df['sent_1']]
            sentences2 = [s for s in df['sent_2']]
            
        sims = method(sentences1, sentences2)
        pearson_correlation = scipy.stats.pearsonr(sims, df['sim'])[0]
        print(label, pearson_correlation)
        pearson_cors.append(pearson_correlation)
        spearman_correlation = scipy.stats.spearmanr(sims, df['sim'])[0]
        spearman_cors.append(spearman_correlation)
        
    return pearson_cors, spearman_cors

def compile_correlation_results(results, benchmarks):
    df = (
        pd.DataFrame(results)
        .transpose()
        .rename(columns={i:b[0] for i, b in enumerate(benchmarks)})
        )
    print(df)
    return df

def main():
    sts_dev, sts_test = download_and_load_sts_data()
    sick_dev = download_sick("https://raw.githubusercontent.com/alvations/stasis/master/SICK-data/SICK_trial.txt")
    sick_test = download_sick("https://raw.githubusercontent.com/alvations/stasis/master/SICK-data/SICK_test_annotated.txt")

    se = StaticEmbeddings()
    word2vec = se.load_word2vec()
    glove = se.load_glove()
    fasttext = se.load_fasttext()

    bert_tokenizer = AutoTokenizer.from_pretrained(model_name='bert-base-uncased')
    bert_model = AutoModel.from_pretrained(model_name='bert-base-uncased')
    bert_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    gpt2_model = GPT2Model.from_pretrained('gpt2', output_hidden_states=True)
    gpt2_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    bert_sent_model = SentenceTransformer('bert-base-nli-mean-tokens')

    benchmarks = [("AVG-W2V", ft.partial(run_avg_benchmark, model=word2vec, use_stoplist=False)),
                ("AVG-W2V-STOP", ft.partial(run_avg_benchmark, model=word2vec, use_stoplist=True)),
                ("AVG-W2V-TFIDF", ft.partial(run_avg_benchmark, model=word2vec, use_stoplist=False, doc_freqs=doc_frequencies)),
                ("AVG-W2V-TFIDF-STOP", ft.partial(run_avg_benchmark, model=word2vec, use_stoplist=True, doc_freqs=doc_frequencies)),
                ("AVG-GLOVE", ft.partial(run_avg_benchmark, model=glove, use_stoplist=False)),
                ("AVG-GLOVE-STOP", ft.partial(run_avg_benchmark, model=glove, use_stoplist=True)),
                ("AVG-GLOVE-TFIDF", ft.partial(run_avg_benchmark, model=glove, use_stoplist=False, doc_freqs=doc_frequencies)),
                ("AVG-GLOVE-TFIDF-STOP", ft.partial(run_avg_benchmark, model=glove, use_stoplist=True, doc_freqs=doc_frequencies)),
                ("AVG-FT", ft.partial(run_avg_benchmark, model=fasttext, use_stoplist=False)),
                ("AVG-FT-STOP", ft.partial(run_avg_benchmark, model=fasttext, use_stoplist=True)),
                ("AVG-FT-TFIDF", ft.partial(run_avg_benchmark, model=fasttext, use_stoplist=False, doc_freqs=doc_frequencies)),
                ("AVG-FT-TFIDF-STOP", ft.partial(run_avg_benchmark, model=fasttext, use_stoplist=True, doc_freqs=doc_frequencies)),
                ("BERT", ft.partial(run_transformer_benchmark, model=bert_model, tokenizer=bert_tokenizer)),
                ("GPT2", ft.partial(run_transformer_benchmark, model=gpt2_model, tokenizer=gpt2_tokenizer)),
                ("BERT-SENT", ft.partial(run_sentence_transformer_benchmarks, model=bert_sent_model))
                ]

    pearson_results, spearman_results = {}, {}
    pearson_results["SICK-DEV"], spearman_results["SICK-DEV"] = run_experiment(sick_dev, benchmarks)
    pearson_results["SICK-TEST"], spearman_results["SICK-TEST"] = run_experiment(sick_test, benchmarks)
    pearson_results["STS-DEV"], spearman_results["STS-DEV"] = run_experiment(sts_dev, benchmarks)
    pearson_results["STS-TEST"], spearman_results["STS-TEST"] = run_experiment(sts_test, benchmarks)  

    plt.rcParams['figure.figsize'] = (10,5)

    print('pearson_results_df')
    pearson_results_df = compile_correlation_results(pearson_results, benchmarks)
    print('-'*30)

    print('pearson_results_df')
    spearman_results_df = compile_correlation_results(spearman_results, benchmarks)
    print('-'*30)

    pearson_results_df.plot(kind="bar").legend(loc="lower left")
    spearman_results_df.plot(kind="bar").legend(loc="lower left")


if __name__ == '__main__':
    main()