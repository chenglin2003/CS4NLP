import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseModel(torch.nn.Module):
    def __init__(self, seq_model, pooling='mean'):
        super(SiameseModel, self).__init__()

        self.seq_model = seq_model

        self.styolometric = nn.Sequential(
            nn.Linear(58, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(seq_model.config.hidden_size+256, 512),
            nn.ReLU(),
            nn.Linear(512, 768),
            nn.ReLU(),
            nn.Linear(768, 1024),
        )

        if pooling == 'mean':
            self.pooling = self.mean_pooling
        elif pooling == 'cls':
            self.pooling = self.cls_pooling

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def cls_pooling(self, model_output, _):
        token_embeddings = model_output[0]
        return token_embeddings[:, 0, :]

    def forward(self, text, styolometric_features):

        x = self.seq_model(**text)

        x = self.pooling(x, text['attention_mask'])

        x = F.normalize(x, p=2, dim=1)

        s = self.styolometric(styolometric_features)

        s = F.normalize(s, p=2, dim=1)

        x = torch.cat((x, s), dim=1)

        x = self.fc(x)

        x = F.normalize(x, p=2, dim=1)

        return x