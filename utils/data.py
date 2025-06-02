from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
import torch
import numpy as np
from utils.loss import cosine_distance
from itertools import combinations
import torch.nn.functional as F
import os
from tqdm import tqdm

from transformers.utils import logging
logger = logging.get_logger("transformers")



class TripletDataset(Dataset):
    def __init__(self, id, tokens, styolometric_features, test=False):
        super().__init__()
        self.id = id
        self.input_ids = tokens['input_ids']
        self.attention_mask = tokens['attention_mask']
        self.id_set = set(self.id)
        self.id_to_indices = {id: np.where(self.id == id)[0]
                                    for id in self.id_set}
        
        self.styolometric_features = torch.from_numpy(styolometric_features).float()

        # We fix all the triplet samples in the dataset
        self.test_idx = []
        self.test = test
        if test:
            np.random.seed(0)
            for idx in range(len(self.id)):
                anchor_id = self.id[idx]
                positive_indices = self.id_to_indices[anchor_id]
                positive_indices = positive_indices[positive_indices != idx]
                positive_idx = np.random.choice(positive_indices)
                negative_id = np.random.choice(list(self.id_set - {anchor_id}))
                negative_indices = self.id_to_indices[negative_id]
                negative_idx = np.random.choice(negative_indices)

                self.test_idx.append([idx, positive_idx, negative_idx])

        self.test_idx = np.array(self.test_idx)
        
    def __len__(self):
        return len(self.id)
    
    def __getitem__(self, idx):

        if self.test:
            idx, positive_idx, negative_idx = self.test_idx[idx]
        else:
            anchor_id = self.id[idx]
            # Get a positive sample
            positive_indices = self.id_to_indices[anchor_id]
            positive_indices = positive_indices[positive_indices != idx]
            positive_idx = np.random.choice(positive_indices)

            # Get a negative sample
            negative_id = np.random.choice(list(self.id_set - {anchor_id}))
            negative_indices = self.id_to_indices[negative_id]
            negative_idx = np.random.choice(negative_indices)

        # Return the triplet      
        anchor_input_ids = self.input_ids[idx]
        anchor_attention_mask = self.attention_mask[idx]
        anchor_styolometric_features = self.styolometric_features[idx]

        positive_input_ids = self.input_ids[positive_idx]
        positive_attention_mask = self.attention_mask[positive_idx]
        positive_styolometric_features = self.styolometric_features[positive_idx]

        negative_input_ids = self.input_ids[negative_idx]
        negative_attention_mask = self.attention_mask[negative_idx]
        negative_styolometric_features = self.styolometric_features[negative_idx]

        return anchor_input_ids, positive_input_ids, negative_input_ids, \
               anchor_attention_mask, positive_attention_mask, negative_attention_mask, \
               anchor_styolometric_features, positive_styolometric_features, negative_styolometric_features 
    
class PairDataset(Dataset):
    def __init__(self, id, tokens, styolometric_features):
        super().__init__()
        self.id = id
        self.input_ids = tokens['input_ids']
        self.attention_mask = tokens['attention_mask']
        
        self.styolometric_features = torch.from_numpy(styolometric_features).float()
        self.id_set = set(self.id)
        self.id_to_indices = {id: np.where(self.id == id)[0]
                                    for id in self.id_set}
        
    def __len__(self):
        return len(self.id)
    
    def __getitem__(self, idx):
        anchor_input_ids = self.input_ids[idx]
        anchor_attention_mask = self.attention_mask[idx]
        anchor_id = self.id[idx]
        anchor_styolometric_features = self.styolometric_features[idx]

        # Get a positive sample
        positive_indices = self.id_to_indices[anchor_id]
        positive_indices = positive_indices[positive_indices != idx]
        positive_idx = np.random.choice(positive_indices)
        positive_input_ids = self.input_ids[positive_idx]
        positive_attention_mask = self.attention_mask[positive_idx]
        positive_styolometric_features = self.styolometric_features[positive_idx]

        return anchor_id, anchor_input_ids, anchor_attention_mask, anchor_styolometric_features, \
               anchor_id, positive_input_ids, positive_attention_mask, positive_styolometric_features
    
    
class BatchDataset(Dataset):
    def __init__(self, id, tokens, styolometric_features):
        super().__init__()
        self.id = id
        self.input_ids = tokens['input_ids']
        self.attention_mask = tokens['attention_mask']
        
        self.styolometric_features = torch.from_numpy(styolometric_features).float()
        
    def __len__(self):
        return len(self.id)
    
    def __getitem__(self, idx):
        anchor_input_ids = self.input_ids[idx]
        anchor_attention_mask = self.attention_mask[idx]
        anchor_id = self.id[idx]
        anchor_styolometric_features = self.styolometric_features[idx]

        return anchor_id, anchor_input_ids, anchor_attention_mask, anchor_styolometric_features
    
def semihard_negative(loss_values, margin):
    semihard_negatives = np.where(np.logical_and(loss_values < margin, loss_values > 0))[0]
    return np.random.choice(semihard_negatives) if len(semihard_negatives) > 0 else None


class FunctionNegativeTripletSelector():

    def __init__(self, margin, negative_selection_fn=semihard_negative, cpu=False):
        super(FunctionNegativeTripletSelector, self).__init__()
        self.cpu = cpu
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn

    def get_triplets(self, embeddings, labels):
        
        embeddings = embeddings.detach()
        similarity_matrix = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1)

        distance_matrix = 1 - (similarity_matrix + 1) / 2
        distance_matrix = distance_matrix.cpu().numpy()

        # logger.warn(distance_matrix)

        triplets = []

        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs
            anchor_positives = np.array(anchor_positives)

            ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]
            for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
                loss_values = ap_distance - distance_matrix[torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(negative_indices)] + self.margin
                hard_negative = self.negative_selection_fn(loss_values, self.margin)
                if hard_negative is not None:
                    hard_negative = negative_indices[hard_negative]
                    triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])

        # if len(triplets) == 0:
        #     triplets.append([anchor_positive[0], anchor_positive[1], negative_indices[0]])

        triplets = np.array(triplets)

        return torch.LongTensor(triplets)
    

class NCEDataset(Dataset):
    def __init__(self, id, tokens, styolometric_features):
        super().__init__()
        self.id = id
        self.input_ids = tokens['input_ids']
        self.attention_mask = tokens['attention_mask']
        
        self.styolometric_features = torch.from_numpy(styolometric_features).float()
        self.id_set = set(self.id)
        self.id_list = list(self.id_set)
        self.id_to_indices = {id: np.where(self.id == id)[0]
                                    for id in self.id_set}

        
    def __len__(self):
        return len(self.id_list)
    
    def __getitem__(self, author_id):
        author_indices = self.id_to_indices[self.id_list[author_id]]
        idx = np.random.choice(author_indices, 2, replace=False)

        anchor_input_ids = self.input_ids[idx[0]]
        anchor_attention_mask = self.attention_mask[idx[0]]
        anchor_id = self.id[idx[0]]
        anchor_styolometric_features = self.styolometric_features[idx[0]]

        # Get a positive sample
        positive_input_ids = self.input_ids[idx[1]]
        positive_attention_mask = self.attention_mask[idx[1]]
        positive_styolometric_features = self.styolometric_features[idx[1]]

        return {
            'anchor_id': anchor_id,
            'anchor_input_ids': anchor_input_ids,
            'anchor_attention_mask': anchor_attention_mask,
            'anchor_styolometric_features': anchor_styolometric_features,
            'positive_input_ids': positive_input_ids,
            'positive_attention_mask': positive_attention_mask,
            'positive_styolometric_features': positive_styolometric_features
        }
    
def tokenize_df(tokenizer, texts):
    # Split into chunks of 512 texts
    for i in tqdm(range(0, len(texts), 512)):
        chunk = texts[i:i+512]
        chunk = tokenizer(chunk.to_list(), return_tensors="pt", padding=True, truncation=True, max_length=512)
        if i == 0:
            train_tokens = chunk
        else:
            train_tokens['input_ids'] = torch.cat((train_tokens['input_ids'], chunk['input_ids']), dim=0)
            train_tokens['attention_mask'] = torch.cat((train_tokens['attention_mask'], chunk['attention_mask']), dim=0)

    return train_tokens

class ClassificationDataset(Dataset):
    def __init__(self, id, tokens, stylometric_features):
        super().__init__()
        self.id = id
        self.input_ids = tokens['input_ids']
        self.attention_mask = tokens['attention_mask']
        self.stylometric_features = torch.from_numpy(stylometric_features).float()
        
    def __len__(self):
        return len(self.id)
    
    def __getitem__(self, idx):
        label = self.id[idx]
        input_ids = self.input_ids[idx]
        attention_mask = self.attention_mask[idx]
        stylometric_features = self.stylometric_features[idx]

        return label, input_ids, attention_mask, stylometric_features