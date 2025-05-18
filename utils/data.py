from torch.utils.data import Dataset, DataLoader
import numpy as np

class TripletDataset(Dataset):
    def __init__(self, id, tokens):
        super().__init__()
        self.id = id
        self.input_ids = tokens['input_ids']
        self.attention_mask = tokens['attention_mask']
        self.id_set = set(self.id)
        self.id_to_indices = {id: np.where(self.id == id)[0]
                                    for id in self.id_set}
        
    def __len__(self):
        return len(self.id)
    
    def __getitem__(self, idx):
        anchor_input_ids = self.input_ids[idx]
        anchor_attention_mask = self.attention_mask[idx]
        anchor_id = self.id[idx]

        # Get a positive sample
        positive_indices = self.id_to_indices[anchor_id]
        positive_indices = positive_indices[positive_indices != idx]
        positive_idx = np.random.choice(positive_indices)
        positive_input_ids = self.input_ids[positive_idx]
        positive_attention_mask = self.attention_mask[positive_idx]

        # Get a negative sample
        negative_id = np.random.choice(list(self.id_set - {anchor_id}))
        negative_indices = self.id_to_indices[negative_id]
        negative_idx = np.random.choice(negative_indices)
        negative_input_ids = self.input_ids[negative_idx]
        negative_attention_mask = self.attention_mask[negative_idx]

        return anchor_input_ids, positive_input_ids, negative_input_ids, \
               anchor_attention_mask, positive_attention_mask, negative_attention_mask