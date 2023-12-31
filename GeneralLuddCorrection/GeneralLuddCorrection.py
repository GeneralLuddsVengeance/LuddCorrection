import torch
import torch.nn as nn
from tqdm import tqdm
from .sentence_vector_model import SWV

class GeneralLuddCorrection():

    def __init__(self, k, u, wv_model):
        super(GeneralLuddCorrection, self).__init__()
        
        self.cos_similarity = nn.CosineSimilarity(dim=-1)

        if isinstance(wv_model, str):
            self.wv = SWV(wv_model)
        else:
            self.wv = wv_model


        with torch.no_grad():
            print('Generating known example vectors')
            kwv = torch.cat([torch.FloatTensor(self.wv(text)).view(1,-1) for text in tqdm(k)], dim=0)

            print('Generating unknown example vectors')
            uwv = torch.cat([torch.FloatTensor(self.wv(text)).view(1,-1) for text in tqdm(u)], dim=0)

            print('Fitting labels')
            self.labels, probs = self.__fit(kwv, uwv)

    def __fit(self, k, u):
        labels = torch.zeros(size=(u.shape[0],))
        last_max = torch.zeros(size=(u.shape[0],)) -1

        for i, wv in tqdm(enumerate(k)):
            res = self.cos_similarity(wv, u)
            labels[res > last_max] = i
            last_max[res > last_max] = res[res>last_max]

        return labels, last_max

