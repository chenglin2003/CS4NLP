import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseModel(torch.nn.Module):
    def __init__(self, seq_model, pooling='last'):
        super(SiameseModel, self).__init__()

        self.seq_model = seq_model

        self.styolometric = nn.Sequential(
            nn.Linear(58, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
        )

        self.fc = nn.Sequential(
            nn.Linear(seq_model.config.hidden_size+128, 512),
            nn.ReLU(),
            nn.Linear(512, 768),
        )

        if pooling == 'mean':
            self.pooling = self.mean_pooling
        elif pooling == 'cls':
            self.pooling = self.cls_pooling
        elif pooling == 'last':
            self.pooling = self.last_pooling
        else:
            raise ValueError(f"Pooling method '{pooling}' is not supported. Choose from 'mean', 'cls', or 'last'.")

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def cls_pooling(self, model_output, _):
        token_embeddings = model_output[0]
        return token_embeddings[:, 0, :]
    
    def last_pooling(self, model_output, _):
        return model_output.last_hidden_state[:, 0]

    def forward(self, text, styolometric_features):

        x = self.seq_model(**text)

        x = self.pooling(x, text['attention_mask'])

        s = self.styolometric(styolometric_features)

        x = torch.cat((x, s), dim=1)

        x = self.fc(x)

        return x