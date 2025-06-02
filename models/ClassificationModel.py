import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassificationModel(torch.nn.Module):
    def __init__(self, embedding_model, num_classes, embedding_dim=768):
        super(ClassificationModel, self).__init__()

        self.embedding_model = embedding_model

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, num_classes),
        )

    def forward(self, text, styolometric_features):
        # Get embeddings from the embedding model
        embeddings = self.embedding_model(text, styolometric_features)

        # Classify using the classifier
        logits = self.classifier(embeddings)

        return logits