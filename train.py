import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from transformers import AutoModel, AutoTokenizer
import os
import time
import wandb


from utils.data import TripletDataset
from utils.loss import TripletLoss, cosine_distance
from utils.metrics import f1
from SiameseModel import SiameseModel

from transformers.utils import logging
logging.set_verbosity_warning()

def main():

    # Logging 
    logger = logging.get_logger("transformers")
    wandb_run = wandb.init(
        entity="b3nguin",
        project="CS4NLP",
        config={
        },
    )

    # Load Dataset
    logger.warning("Loading Dataset...")
    train_df = pd.read_csv("data/blogtext_train.csv")
    test_df = pd.read_csv("data/blogtext_test.csv")

    # Load Model and Tokenizer
    logger.warning("Loading Model and Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", use_fast=False)
    minilm = AutoModel.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2", 
        trust_remote_code=True, 
        # torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
    )

    # Freeze the model parameters
    for param in minilm.parameters():
        param.requires_grad = False

    logger.warning("Tokenizing Dataset...")
    train_ids = train_df['id']
    train_texts = train_df['text']
    if os.path.exists("data/train_ids.pt") and os.path.exists("data/train_attention_mask.pt"):
        logger.warning("Loading pre-tokenized train dataset...")
        train_tokens = torch.load("data/train_ids.pt")
        train_attention_mask = torch.load("data/train_attention_mask.pt")
        train_tokens = {
            'input_ids': train_tokens,
            'attention_mask': train_attention_mask
        }
    else:
        logger.warning("Tokenizing train dataset...")
        train_tokens = tokenizer(train_texts.to_list(), return_tensors="pt", padding=True, truncation=True, max_length=512)
        torch.save(train_tokens['input_ids'], "data/train_ids.pt")
        torch.save(train_tokens['attention_mask'], "data/train_attention_mask.pt")

    test_ids = test_df['id']
    test_text = test_df['text']
    if os.path.exists("data/test_ids.pt") and os.path.exists("data/test_attention_mask.pt"):
        logger.warning("Loading pre-tokenized test dataset...")
        test_tokens = torch.load("data/test_ids.pt")
        test_attention_mask = torch.load("data/test_attention_mask.pt")
        test_tokens = {
            'input_ids': test_tokens,
            'attention_mask': test_attention_mask
        }
    else:
        logger.warning("Tokenizing test dataset...")
        test_tokens = tokenizer(test_text.to_list(), return_tensors="pt", padding=True, truncation=True, max_length=512)
        torch.save(test_tokens['input_ids'], "data/test_ids.pt")
        torch.save(test_tokens['attention_mask'], "data/test_attention_mask.pt")

    logger.warning("Creating Dataset and DataLoader...")

    # Create Dataset and DataLoader
    train_dataset = TripletDataset(train_ids.to_numpy(), train_tokens)
    test_dataset = TripletDataset(test_ids.to_numpy(), test_tokens)

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, pin_memory=True)

    # Create Model
    model = SiameseModel(minilm)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.warning("Using device: " + str(device))
    model.to(device)

    # Loss Function and Optimizer
    criterion = nn.TripletMarginWithDistanceLoss(
            distance_function=cosine_distance,
            margin=1.5,
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)

    epochs = 3

    logger.warning("Training Model...")
    # Training Loop
    for epoch in range(epochs):
        model.train()
        for i, batch in enumerate(train_dataloader):
            start_time = time.time()

            anchor_input_ids, positive_input_ids, negative_input_ids, \
            anchor_attention_mask, positive_attention_mask, negative_attention_mask = batch

            anchor_input_ids = anchor_input_ids.to(device)
            positive_input_ids = positive_input_ids.to(device)
            negative_input_ids = negative_input_ids.to(device)
            anchor_attention_mask = anchor_attention_mask.to(device)
            positive_attention_mask = positive_attention_mask.to(device)
            negative_attention_mask = negative_attention_mask.to(device)

            # Forward pass
            anchor = {
                'input_ids': anchor_input_ids,
                'attention_mask': anchor_attention_mask
            }
            positive = {
                'input_ids': positive_input_ids,
                'attention_mask': positive_attention_mask
            }
            negative = {
                'input_ids': negative_input_ids,
                'attention_mask': negative_attention_mask
            }

            # Forward pass
            anchor_output = model(anchor)
            positive_output = model(positive)
            negative_output = model(negative)

            loss = criterion(anchor_output, positive_output, negative_output)

            # Backward pass
            loss.backward()

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            
            if (i+1) % 100 == 0:
                pos_dist = torch.nn.functional.cosine_similarity(anchor_output, positive_output)
                neg_dist = torch.nn.functional.cosine_similarity(anchor_output, negative_output)

                pos_preds = pos_dist > 0
                neg_preds = neg_dist > 0

                pos_labels = torch.ones_like(pos_preds)
                neg_labels = torch.zeros_like(neg_preds)

                preds = torch.cat((pos_preds, neg_preds), dim=0) * 1.0
                labels = torch.cat((pos_labels, neg_labels), dim=0)

                # Calculate F1 score
                f1_score = f1(preds, labels)

                acc = (preds == labels).float().mean()

                logger.warning(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}, F1: {f1_score:.4f}, Acc: {acc:.4f} , Time: {time.time() - start_time:.2f}s")
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": loss.item(),
                    "train_f1": f1_score,
                    "train_acc": acc,
                })
                logger.warning(str(torch.mean(pos_dist)))
                logger.warning(str(torch.mean(neg_dist)))

        scheduler.step()
    
        # Evaluation Loop
        model.eval()
        f1_scores = []
        with torch.no_grad():
            for i, batch in enumerate(test_dataloader):
                anchor, positive, negative = batch
                anchor = anchor.to(device)
                positive = positive.to(device)
                negative = negative.to(device)

                # Forward pass
                pos_dist = model(anchor, positive)
                neg_dist = model(anchor, negative)

                loss = criterion(pos_dist, neg_dist)

                pos_preds = pos_dist < 0.5
                neg_preds = neg_dist < 0.5

                pos_labels = torch.ones_like(pos_preds)
                neg_labels = torch.zeros_like(neg_preds)

                preds = torch.cat((pos_preds, neg_preds), dim=0)
                labels = torch.cat((pos_labels, neg_labels), dim=0)

                # Calculate F1 score
                f1_score = f1(preds, labels)

                f1_scores.append(f1_score)

            logger.warning(f"Test F1 Score: {np.mean(f1_scores):.4f}")
            logger.warning(f"Test Loss: {np.mean(loss):.4f}")
            wandb.log({
                "test_loss": np.mean(loss),
                "test_f1": np.mean(f1_scores),
            })

        # Save Model
        if not os.path.exists("models"):
            os.makedirs("models")
        torch.save(model.state_dict(), f"models/model_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    main()