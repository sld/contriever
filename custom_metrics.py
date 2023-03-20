import os
import json
import pandas as pd
import numpy as np
import time
import faiss
import argparse
import tqdm
import torch
from src.contriever import Contriever
from transformers import AutoTokenizer


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="test.json")
    parser.add_argument("--model_path", type=str, default="facebook/mcontriever-msmarco")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_path", type=str, default="data/test_results.json")
    args = parser.parse_args()
    return args


def filter_sent(sentence):
    # faqs should already be filtered
    return sentence


def main():
    args = _parse_args()
    print('torch.cuda.is_available(): ', torch.cuda.is_available())
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    mcontriever_msmarco = Contriever.from_pretrained(args.model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    data_path = args.data_path
    with open(data_path) as f:
        data = json.load(f)
    y_train = []
    y_test = []
    X = []

    for row in data:
        a = filter_sent(row['answer'])
        y_train.append(a)
        y_test.append(a)
        X.append(filter_sent(row['question']))
        for p in row['paraphrases']:
            X.append(filter_sent(p['text']))
            y_test.append(a)

    print(len(y_train), len(y_test), len(X))

    train_embeddings = []
    for i in tqdm.tqdm(range(0, len(y_train), args.batch_size)):
        inputs = tokenizer(y_train[i:i + args.batch_size], padding=True, truncation=True, return_tensors="pt").to(device)
        emb = mcontriever_msmarco(**inputs).detach().cpu().numpy()
        train_embeddings.extend(emb)

    test_embeddings = []
    for i in tqdm.tqdm(range(0, len(X), args.batch_size)):
        inputs = tokenizer(X[i:i + args.batch_size], padding=True, truncation=True, return_tensors="pt").to(device)
        emb = mcontriever_msmarco(**inputs).detach().cpu().numpy()
        test_embeddings.extend(emb)

    index = faiss.IndexFlatIP(len(train_embeddings[0]))

    vectors = np.array(train_embeddings, dtype=np.float32)
    index.add(vectors)

    test_search_results = index.search(np.array(test_embeddings), 3)

    top1 = 0
    top3 = 0
    for i, (ind1, ind2, ind3) in enumerate(test_search_results[1]):
        if y_train[ind1] == y_test[i]:
            top1 += 1
        if y_train[ind1] == y_test[i] or y_train[ind2] == y_test[i] or y_train[ind3] == y_test[i]:
            top3 += 1
    print(top1 / len(y_test), top3 / len(y_test), len(y_test))


if __name__ == '__main__':
    main()
