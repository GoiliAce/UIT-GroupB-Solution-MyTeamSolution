from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModel
import torch

import numpy as np


class BM25Retrieval:   
    def get_score(self, query, doc):
        tokenized_corpus = [doc.split(" ")]
        bm25 = BM25Okapi(tokenized_corpus)
        query = query.split(" ")
        return bm25.get_scores(query)[0] 
    def get_scores(self, query, docs):
        tokenized_corpus = [doc.split(" ") for doc in docs]
        bm25 = BM25Okapi(tokenized_corpus)
        query = query.split(" ")
        return bm25.get_scores(query)
    
    def get_topk(self, query, docs, k=5):
        scores = self.get_scores(query, docs)
        topk_idx = np.argsort(scores)[::-1][:k]
        return [docs[i] for i in topk_idx]
    def get_top_rate(self, query, docs, rate=0.5):
        scores = self.get_scores(query, docs)
        scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
        topk_idx = np.argsort(scores)[::-1]
        topk_idx = topk_idx[scores[topk_idx] >= rate]
        return [docs[i] for i in topk_idx]
    
class TFIDFRetrieval:
    def get_score(self, query, doc):
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform([doc])
        query = vectorizer.transform([query])
        return cosine_similarity(X, query).ravel()[0]
    def get_scores(self, query, docs):
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(docs)
        query = vectorizer.transform([query])
        return cosine_similarity(X, query).ravel()
    def get_topk(self, query, docs, k=5):
        scores = self.get_scores(query, docs)
        topk_idx = np.argsort(scores)[::-1][:k]
        return [docs[i] for i in topk_idx]
    def get_top_rate(self, query, docs, rate=0.5):
        scores = self.get_scores(query, docs)
        scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
        topk_idx = np.argsort(scores)[::-1]
        topk_idx = topk_idx[scores[topk_idx] >= rate]
        return [docs[i] for i in topk_idx]
    
class SentenceRetrieval:
    def __init__(self, model_name="bkai-foundation-models/vietnamese-bi-encoder", device="cuda"):
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.tokenzier = AutoTokenizer.from_pretrained(model_name)
        self.device = device
    
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    def get_scores(self, query, context):
        context_input = context + [query]
        encoded_input = self.tokenzier(context_input, padding=True, return_tensors='pt').to(self.device)
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask']).cpu().numpy()
        return cosine_similarity(sentence_embeddings[-1].reshape(1, -1), sentence_embeddings[:-1])[0]
        
    def get_top_rate(self, query, context, rate=0.2):
        scores = self.get_scores(query, context)
        topk_idx = np.argsort(scores)[::-1]
        topk_idx = topk_idx[scores[topk_idx] >= rate]
        return [context[idx] for idx in topk_idx]
    
    def get_topk(self, query, context, k=5):
        scores = self.get_scores(query, context)
        top_idx = np.argsort(scores)[::-1]
        return [context[idx] for idx in top_idx[:k]]
    