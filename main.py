# main.py
from fastapi import FastAPI
from pydantic import BaseModel
#import numpy as np
#import pandas as pd
import pickle
import spacy  # Plotting tools

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

import gensim
from gensim.utils import simple_preprocess

import spacy.cli
spacy.cli.download("en_core_web_sm")
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])


nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
stop_words = stopwords.words('english')
stop_words.extend(['code'])
stop_w = list(set(stopwords.words('english'))) + ['[', ']', ',', '.', ':', '?', '(', ')', 'code']

app = FastAPI()

with open('lda_id2word.pkl', 'rb') as f:
    lda_id2word = pickle.load(f)

with open('lda_model.pkl', 'rb') as f:
    lda_model = pickle.load(f)

'''with open('lda_scaler.pkl', 'rb') as f:
    lda_scaler = pickle.load(f)

with open('lda_xgboost_model.pkl', 'rb') as f:
    lda_xgboost_model = pickle.load(f)'''


def tokenizer_fct(sentence):
    # print(sentence)
    sentence_clean = sentence.replace('-', ' ').replace('+', ' ').replace('/', ' ').replace('#', ' ')
    word_tokens = word_tokenize(sentence_clean)
    return word_tokens


def stop_word_filter_fct(list_words):
    filtered_w = [w for w in list_words if not w in stop_w]
    filtered_w2 = [w for w in filtered_w if len(w) > 2]
    return filtered_w2


# lower case et alpha
def lower_start_fct(list_words):
    lw = [w.lower() for w in list_words if (not w.startswith("@")) and (not w.startswith("http"))]
    return lw


# Lemmatizer (base d'un mot)
from nltk.stem import WordNetLemmatizer


def lemma_fct(list_words):
    lemmatizer = WordNetLemmatizer()
    lem_w = [lemmatizer.lemmatize(w) for w in list_words]
    return lem_w


# Fonction de préparation du texte pour le bag of words (Countvectorizer et Tf_idf, Word2Vec)
def transform_bow_fct(desc_text):
    word_tokens = tokenizer_fct(desc_text)
    sw = stop_word_filter_fct(word_tokens)
    lw = lower_start_fct(sw)
    # lem_w = lemma_fct(lw)
    transf_desc_text = ' '.join(lw)
    return transf_desc_text


# Fonction de préparation du texte pour le bag of words avec lemmatization
def transform_bow_lem_fct(desc_text):
    word_tokens = tokenizer_fct(desc_text)
    sw = stop_word_filter_fct(word_tokens)
    lw = lower_start_fct(sw)
    lem_w = lemma_fct(lw)
    transf_desc_text = ' '.join(lem_w)
    return transf_desc_text


# Define function for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]


def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


def transform_lem_sentence_fct(text):
    text_lem = [gensim.utils.simple_preprocess(text)]

    # nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    # Remove Stop Words
    data_words_nostops_test = remove_stopwords(text_lem)

    # Do lemmatization keeping only noun, adj, vb, adv
    return lemmatization(data_words_nostops_test, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])[0]

def transform_num(txt):
    tt = transform_lem_sentence_fct(txt)
    return lda_id2word.doc2bow(tt)


def best_topics(min_proba=0.1, lda_topics=[]):
    return sorted(list(filter(lambda p: p[1] > min_proba, [c for c in lda_topics])), key=lambda v: v[1], reverse=True)


def key_words(text, min_prob_topics=0.1, min_prob_words=0.015):
    text_1 = transform_bow_lem_fct(text)

    text_num = transform_num(text_1)

    text_lda = lda_model[text_num]

    best_2_topics = [t[0] for t in best_topics(min_prob_topics, text_lda[0])][:2]

    return best_2_topics, {t: list(filter(lambda p: p[1] > min_prob_words, lda_model.show_topic(t))) for t in best_2_topics}


def get_matrix(topics):
    MATRIX = []
    for l in topics:
        LINE = []
        for v in l:
            while len(LINE) < v[0]:
                LINE.append(0.0)
            LINE.append(v[1])
        LINE.extend([0.0] * (9 - len(LINE))) if len(LINE) < 9 else None
        MATRIX.append(LINE)
    return MATRIX


class Sentence(BaseModel):
    question: str


@app.get("/")
def hello():
    return {"message": "Welcome! to the IK-P5-APP"}

@app.post("/Words_proposition")
async def propose(request: Sentence):

    topics = key_words(request.question)

    return {number: [w[0] for w in words] for number, words in topics[1].items()}

'''@app.post("/Tags_prediction")
async def predict(request: Sentence):

    sentense_ = transform_bow_lem_fct(request.question)

    text_num = transform_lem_sentence_fct(sentense_)

    corpus = lda_id2word.doc2bow(text_num)

    vector = lda_model[corpus]

    vect = get_matrix([vector[0]])

    vect_scaled = lda_scaler.transform(vect)

    pred = lda_xgboost_model.predict(vect_scaled)

    index = [i for i in range(10) if pred[0][i] == 1]

    tags = {
        0: 'javascript',
        1: 'java',
        2: 'c#',
        3: 'python',
        4: 'php',
        5: 'android',
        6: 'html',
        7: 'jquery',
        8: 'c++',
        9: 'css'}

    print([tags[i] for i in index])

    #results = [val for val in np.array(tab)[np.argwhere(y_pred==1)][0]]

    results = [tags[i] for i in index]

    # Return the prediction
    return {"tags": results}'''