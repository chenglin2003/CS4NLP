import torch
import numpy as np
from utils.loss import cosine_distance
from utils.metrics import f1

def eval_model(model, dataloader, criterion, device, pos_thresh=0.5, test_percent=0.01):
    model.eval()
    f1_scores = []
    acc_scores = []
    pos_dists = []
    neg_dists = []

    stop = int(len(dataloader) * test_percent)

    with torch.no_grad():
        for j, batch in enumerate(dataloader):
            if j > stop:
                break

            a_input_ids, p_input_ids, n_input_ids, \
            a_attention_mask, p_attention_mask, n_attention_mask, \
            a_styolometric_features, p_styolometric_features, n_styolometric_features = batch

            a_input_ids = a_input_ids.to(device)
            p_input_ids = p_input_ids.to(device)
            n_input_ids = n_input_ids.to(device)
            a_attention_mask = a_attention_mask.to(device)
            p_attention_mask = p_attention_mask.to(device)
            n_attention_mask = n_attention_mask.to(device)
            a_styolometric_features = a_styolometric_features.to(device)
            p_styolometric_features = p_styolometric_features.to(device)
            n_styolometric_features = n_styolometric_features.to(device)

            # Forward pass
            anchor = {
                'input_ids': a_input_ids,
                'attention_mask': a_attention_mask
            }
            positive = {
                'input_ids': p_input_ids,
                'attention_mask': p_attention_mask
            }
            negative = {    
                'input_ids': n_input_ids,
                'attention_mask': n_attention_mask
            }

            # Forward pass
            anchor_output = model(anchor, a_styolometric_features)
            positive_output = model(positive, p_styolometric_features)
            negative_output = model(negative, n_styolometric_features)

            loss = criterion(anchor_output, positive_output, negative_output)

            pos_dist = cosine_distance(anchor_output, positive_output)
            neg_dist = cosine_distance(anchor_output, negative_output)

            pos_dists.append(pos_dist)
            neg_dists.append(neg_dist)

            pos_preds = pos_dist < pos_thresh
            neg_preds = neg_dist < pos_thresh
            
            pos_labels = torch.ones_like(pos_preds)
            neg_labels = torch.zeros_like(neg_preds)

            preds = torch.cat((pos_preds, neg_preds), dim=0)
            labels = torch.cat((pos_labels, neg_labels), dim=0)

            # Calculate F1 score
            f1_score = f1(preds, labels)

            f1_scores.append(f1_score)

            acc = (preds == labels).float().mean().cpu().item()

            acc_scores.append(acc)

    model.train()

    return {
        'f1_score': np.mean(f1_scores),
        'accuracy': np.mean(acc_scores),
        'pos_dist': torch.median(torch.cat(pos_dists)),
        'neg_dist': torch.median(torch.cat(neg_dists))
    }