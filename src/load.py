import json
import pickle
from collections import Counter

import numpy as np
from tqdm import tqdm


def loadfile(fn, num=1):
    print('loading a file...' + fn)
    ret = []
    with open(fn, encoding='utf-8') as f:
        for line in f:
            th = line[:-1].split('\t')
            x = []
            for i in range(num):
                x.append(int(th[i]))
            ret.append(tuple(x))
    return ret


def get_ids(fn):
    ids = []
    with open(fn, encoding='utf-8') as f:
        for line in f:
            th = line[:-1].split('\t')
            ids.append(int(th[0]))
    return ids

def get_ent2id(fns):
    ent2id = {}
    for fn in fns:
        with open(fn, 'r', encoding='utf-8') as f:
            for line in f:
                th = line[:-1].split('\t')
                ent2id[th[1]] = int(th[0])
    return ent2id


def load_attr(fns, e, ent2id, topA=1000):
    cnt = {}
    # fns:[attrs_1, attrs_2]
    for fn in fns:
        with open(fn, 'r', encoding='utf-8') as f:
            for line in f:
                th = line[:-1].split('\t')
                if th[0] not in ent2id:
                    continue
                for i in range(1, len(th)):
                    if th[i] not in cnt:
                        cnt[th[i]] = 1
                    else:
                        cnt[th[i]] += 1
    fre = [(k, cnt[k]) for k in sorted(cnt, key=cnt.get, reverse=True)]
    if len(fre) < 1000:
        topA = len(fre)
    attr2id = {}
    for i in range(topA):
        attr2id[fre[i][0]] = i
    attr = np.zeros((e, topA), dtype=np.float32)

    for fn in fns:
        with open(fn, 'r', encoding='utf-8') as f:
            for line in f:
                th = line[:-1].split('\t')
                if th[0] in ent2id:
                    for i in range(1, len(th)):
                        if th[i] in attr2id:
                            attr[ent2id[th[0]]][attr2id[th[i]]] = 1.0
    return attr


def load_top_relation(e, triples, topR=1000):
    rel_mat = np.zeros((e, topR), dtype=np.float32)
    rels = np.array(triples)[:, 1]
    top_rels = Counter(rels).most_common(topR)
    rel_index_dict = {r: i for i, (r, cnt) in enumerate(top_rels)}
    for tri in triples:
        h = tri[0]
        r = tri[1]
        o = tri[2]
        if r in rel_index_dict:
            rel_mat[h][rel_index_dict[r]] += 1.
            rel_mat[o][rel_index_dict[r]] += 1.
    return np.array(rel_mat)


def load_relation(e_num, r_num, triples, topR=1000):
    rel_mat_in = np.zeros((e_num, r_num), dtype=np.float32)
    rel_mat_out = np.zeros((e_num, r_num), dtype=np.float32)
    for (h, r, t) in triples:
        rel_mat_in[t][r] += 1
        rel_mat_out[h][r] += 1
    return np.array(rel_mat_in), np.array(rel_mat_out)


def load_word_emb(filename):
    """
        please download the zip file from "http://nlp.stanford.edu/data/glove.6B.zip"
        and choose "glove.6B.300d.txt" as the word vectors.
    """
    word_vecs = {}
    print("load word_emb......")
    with open(filename, encoding='UTF-8') as f:
        for line in tqdm(f.readlines()):
            line = line.split()
            word_vecs[line[0]] = np.array([float(x) for x in line[1:]])
    return word_vecs


def load_trans_ent_name(file_path):
    # translated entity names
    ent_names = json.load(open(file_path, "r"))
    return ent_names


def word_char_f(node_size, ent_names, word_vecs):
    d = {}
    count = 0
    for _, name in ent_names:
        for word in name:
            word = word.lower()
            for idx in range(len(word) - 1):
                if word[idx:idx + 2] not in d:
                    d[word[idx:idx + 2]] = count
                    count += 1

    ent_vec = np.zeros((node_size, 300))
    char_vec = np.zeros((node_size, len(d)))

    for i, name in ent_names:
        k = 0
        for word in name:
            word = word.lower()
            if word in word_vecs:
                ent_vec[i] += word_vecs[word]
                k += 1

            for idx in range(len(word) - 1):
                char_vec[i, d[word[idx:idx + 2]]] += 1
        if k:
            ent_vec[i] /= k
        else:
            ent_vec[i] = np.random.random(300) - 0.5

        if np.sum(char_vec[i]) == 0:
            char_vec[i] = np.random.random(len(d)) - 0.5
        ent_vec[i] = ent_vec[i] / np.linalg.norm(ent_vec[i])
        char_vec[i] = char_vec[i] / np.linalg.norm(char_vec[i])

    return ent_vec, char_vec


def load_json_embd(path):
    embd_dict = {}
    with open(path) as f:
        for line in f:
            example = json.loads(line.strip())
            vec = np.array([float(e) for e in example['feature'].split()])
            embd_dict[int(example['guid'])] = vec
    return embd_dict