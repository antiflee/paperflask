from django.conf import settings

import gensim.models.word2vec as w2v
import pickle
import numpy as np
import os

os.chdir(os.path.join(settings.BASE_DIR,'paperflask','analysis'))

bigram = pickle.load(open("bigram.p","rb"))
titles2vec = w2v.Word2Vec.load("titles2vec_25.w2v")
titlesDict = pickle.load(open('title_vec_dict', 'rb'))

def word_tokenize(s):
    res = []
    prev = 0
    for i in range(len(s)+1):
        if i == len(s) or not s[i].isalnum():
            res.append(s[prev:i])
            prev = i + 1
    return res

def cos_sim(v1, v2):
    """ Returns a custmized cosine similarity of two vectors"""
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    norm_product = norm1 * norm2
    if norm_product == 0:
        return 0

    # It is found that the vallina cosine similarity doesn't do
    # a good job, partially due to the fact that it ignores the
    # difference between the lengths of the two vectors. Here I
    # try to add a pre-factor to the original cos_sim to repre-
    # sent the difference between two lengths.
    norm_ratio = min(norm1, norm2) / max(norm1, norm2)

    return np.dot(v1, v2) * np.sqrt(norm_ratio) / norm_product

def clean_title(title):
    # To do: convert molecular formula
    words = word_tokenize(title)
    try:
        words[0] = words[0].lower()
    except Exception as e:
        pass
    words = ' '.join(words)
    words = bigram[words.split()]
    return " ".join([w for w in words if w not in r':-.,'])

def title_to_vec(title, model=titles2vec, dim=25):
    result = np.zeros(dim)
    words = word_tokenize(title)
    try:
        words[0] = words[0].lower()
    except Exception as e:
        pass
    words = ' '.join(words)
    words = bigram[words.split()]
    words = [w for w in words if w not in r':-.,']

    for word in words:
        if word in model.wv.vocab:
            w2v = model.wv[word]
            result += w2v

    return result

def get_most_similar_title(new_title, titlesDict=titlesDict, k=5):
    """ Returns the k most similar titles to the new_title"""
    new_title = clean_title(new_title)
    new_vec = title_to_vec(new_title)

    # Return the indices of the largest k elements, ordered by their values.
    from heapq import heappush, heappop
    res = []
    for key, val in titlesDict.items():
        sim = cos_sim(new_vec, val)
        if not res or len(res) < k or res[0][0] < sim:
            heappush(res, (sim, key))
        if len(res) > k:
            heappop(res)
    similar_titles = [i for n, i in sorted(res, reverse=True)]

    return similar_titles
