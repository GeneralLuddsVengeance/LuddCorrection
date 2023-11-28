from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn

class SWV(nn.Module):

    def __init__(self, model='all-mpnet-base-v2'):
        self.wv = SentenceTransformer(model)

    def forward(self, text):
        return self.wv(text)
