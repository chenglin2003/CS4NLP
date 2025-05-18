import torch
import torch.nn as nn

class TripletLoss(nn.Module):

    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, pos_dist, neg_dist):
        return torch.mean(torch.clamp(pos_dist - neg_dist + self.margin, min=0.0))
    
def cosine_distance(a, b):
    return 1 - torch.nn.functional.cosine_similarity(a, b)

