import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):

    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, pos_dist, neg_dist):
        return torch.mean(torch.clamp(pos_dist - neg_dist + self.margin, min=0.0))
    
def cosine_distance(a, b):
    return 1 - (torch.nn.functional.cosine_similarity(a, b) + 1 ) / 2

def cosine_similarity(a, b):
    a = F.normalize(a, p=2, dim=1)
    b = F.normalize(b, p=2, dim=1)
    return a @ b.T


class ImprovedContrastiveLoss(nn.Module):
    """
    Improved Contrastive Loss
    https://arxiv.org/pdf/2308.03281
    https://github.com/UKPLab/sentence-transformers/issues/2774
    """

    def __init__(self, temperature=0.1):
        super(ImprovedContrastiveLoss, self).__init__()
        self.t = temperature
        self.s = nn.CosineSimilarity(dim=2)

    def forward(self, q, d):

        q = F.normalize(q, p=2, dim=1)
        d = F.normalize(d, p=2, dim=1)

        similarity_q_d = cosine_similarity(q, d)
        similarity_q_q = cosine_similarity(q, q)
        similarity_d_d = cosine_similarity(d, d)

        # Compute the partition function
        exp_sim_q_d = torch.exp(similarity_q_d / self.t)
        exp_sim_q_q = torch.exp(similarity_q_q / self.t)
        exp_sim_d_d = torch.exp(similarity_d_d / self.t)

        # Ensure the diagonal is not considered in negative samples
        mask = torch.eye(similarity_q_d.size(0), device=similarity_q_d.device).bool()
        exp_sim_q_q = exp_sim_q_q.masked_fill(mask, 0)
        exp_sim_d_d = exp_sim_d_d.masked_fill(mask, 0)

        partition_function = exp_sim_q_d.sum(dim=1) + exp_sim_q_d.sum(dim=0) + exp_sim_q_q.sum(dim=1) + exp_sim_d_d.sum(dim=0)

        # Compute the loss
        loss = -torch.log(exp_sim_q_d.diag() / partition_function).mean()

        return loss
    
class MultipleNegativesSymmetricRankingLoss(nn.Module):
    """
    Multiple Negatives Symmetric Ranking Loss
    https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/losses/MultipleNegativesSymmetricRankingLoss.py#L13-L98
    """

    def __init__(self, scale = 20.0):
        super(MultipleNegativesSymmetricRankingLoss, self).__init__()
        self.scale = scale
        self.CE = nn.CrossEntropyLoss()

    def forward(self, a, c):
        scores = cosine_similarity(a, c) * self.scale
        labels = torch.tensor(
            range(len(scores)), dtype=torch.long, device=scores.device
        )

        anchor_positive_scores = scores[:, 0 : len(c)]
        forward_loss = self.CE(scores, labels)
        backward_loss = self.CE(anchor_positive_scores.transpose(0, 1), labels)
        return (forward_loss + backward_loss) / 2
        
