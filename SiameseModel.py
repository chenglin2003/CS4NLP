import torch
import torch.nn as nn

class SiameseModel(torch.nn.Module):
    def __init__(self, seq_model):
        super(SiameseModel, self).__init__()

        self.seq_model = seq_model

        self.fc = nn.Sequential(
            nn.Linear(seq_model.config.hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
        )

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, text):

        x = self.seq_model(**text)

        x = self.mean_pooling(x, text['attention_mask'])

        x = self.fc(x)

        return x