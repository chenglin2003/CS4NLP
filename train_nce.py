import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from transformers import AutoModel, AutoTokenizer
import os
import time
import wandb
from tqdm import tqdm

from utils.data import NCEDataset
from utils.loss import cosine_similarity, MultipleNegativesSymmetricRankingLoss
from utils.metrics import f1
from models.SiameseModel import SiameseModel

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

    VARIANT = "blogtext"
    ENCODER = "Alibaba-NLP/gte-base-en-v1.5"
    print("Using variant:", VARIANT)
    print("Using encoder:", ENCODER)

    # Load Dataset
    logger.warning("Loading Dataset...")
    train_df = pd.read_csv(f"dataset/{VARIANT}_train.csv")
    test_df = pd.read_csv(f"dataset/{VARIANT}_test.csv")

    # Load Model and Tokenizer
    logger.warning("Loading Model and Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        ENCODER, 
        use_fast=True,
        unpad_inputs=True,
        use_memory_efficient_attention=True,
        torch_dtype=torch.float16
    )
    encoder = AutoModel.from_pretrained(
        ENCODER, 
        trust_remote_code=True, 
    )

    # Freeze the encoder parameters
    for param in encoder.parameters():
        param.requires_grad = False

    # Pre tokenize Dataset
    logger.warning("Tokenizing Dataset...")
    train_ids = train_df['id']
    train_texts = train_df['text']
    if os.path.exists(f"data/{VARIANT}_train_ids.pt") and os.path.exists(f"data/{VARIANT}_train_attention_mask.pt"):
        logger.warning("Loading pre-tokenized train dataset...")
        train_tokens = torch.load(f"data/{VARIANT}_train_ids.pt")
        train_attention_mask = torch.load(f"data/{VARIANT}_train_attention_mask.pt")
        train_tokens = {
            'input_ids': train_tokens,
            'attention_mask': train_attention_mask
        }
    else:
        logger.warning("Tokenizing train dataset...")
            # Split into chunks of 512 texts
        for i in tqdm(range(0, len(train_texts), 512)):
            chunk = train_texts[i:i+512]
            chunk = tokenizer(chunk.to_list(), return_tensors="pt", padding="max_length", truncation=True, max_length=512)
            if i == 0:
                train_tokens = chunk
            else:
                train_tokens['input_ids'] = torch.cat((train_tokens['input_ids'], chunk['input_ids']), dim=0)
                train_tokens['attention_mask'] = torch.cat((train_tokens['attention_mask'], chunk['attention_mask']), dim=0)
        # train_tokens = tokenizer(train_texts.to_list(), return_tensors="pt", padding=True, truncation=True, max_length=512)
        torch.save(train_tokens['input_ids'], f"data/{VARIANT}_train_ids.pt")
        torch.save(train_tokens['attention_mask'], f"data/{VARIANT}_train_attention_mask.pt")

    test_ids = test_df['id']
    test_text = test_df['text']
    if os.path.exists(f"data/{VARIANT}_test_ids.pt") and os.path.exists(f"data/{VARIANT}_test_attention_mask.pt"):
        logger.warning("Loading pre-tokenized test dataset...")
        test_tokens = torch.load(f"data/{VARIANT}_test_ids.pt")
        test_attention_mask = torch.load(f"data/{VARIANT}_test_attention_mask.pt")
        test_tokens = {
            'input_ids': test_tokens,
            'attention_mask': test_attention_mask
        }
    else:
        logger.warning("Tokenizing test dataset...")
        test_tokens = tokenizer(test_text.to_list(), return_tensors="pt", padding="max_length", truncation=True, max_length=512)
        torch.save(test_tokens['input_ids'], f"data/{VARIANT}_test_ids.pt")
        torch.save(test_tokens['attention_mask'], f"data/{VARIANT}_test_attention_mask.pt")

    logger.warning("Creating Dataset and DataLoader...")

    # Load Stylometric, ignore text and id columns
    train_stylometric = train_df.drop(columns=['id', 'text'])
    test_stylometric = test_df.drop(columns=['id', 'text'])
    train_stylometric = train_stylometric.to_numpy()
    test_stylometric = test_stylometric.to_numpy()

    # Create Dataset and DataLoader
    train_dataset = NCEDataset(train_ids.to_numpy(), train_tokens, train_stylometric)
    test_dataset = NCEDataset(test_ids.to_numpy(), test_tokens, test_stylometric)

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, pin_memory=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, pin_memory=True)

    # Create Model
    model = SiameseModel(encoder)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.warning("Using device: " + str(device))
    model.to(device)

    # Loss Function and Optimizer
    criterion = MultipleNegativesSymmetricRankingLoss(scale=10.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    epochs = 30
    accumulation_steps = 1
    current_step = 0
    mini_test_size = 0.005 * len(train_dataloader)
    frozen = True
    freeze_until = 10

    logger.warning("Training Model...")
    # Training Loop
    for epoch in range(epochs):

        # Start fine-tuning encoder
        if frozen and (epoch+1) >= freeze_until:
            logger.warning("Unfreezing encoder parameters...")
            for param in model.parameters():
                param.requires_grad = True
            frozen = False
            for g in optimizer.param_groups:
                g['lr'] = 2e-6
            train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, pin_memory=True, drop_last=True)

        model.train()
        for i, batch in enumerate(train_dataloader):

            start_time = time.time()

            a_input_ids = batch['anchor_input_ids']
            p_input_ids = batch['positive_input_ids']
            a_attention_mask = batch['anchor_attention_mask']
            p_attention_mask = batch['positive_attention_mask']
            a_styolometric_features = batch['anchor_styolometric_features']
            p_styolometric_features = batch['positive_styolometric_features']

            a_input_ids = a_input_ids.to(device)
            p_input_ids = p_input_ids.to(device)
            a_attention_mask = a_attention_mask.to(device)
            p_attention_mask = p_attention_mask.to(device)
            a_styolometric_features = a_styolometric_features.to(device)
            p_styolometric_features = p_styolometric_features.to(device)

            # Forward pass
            anchor = {
                'input_ids': a_input_ids,
                'attention_mask': a_attention_mask
            }
            positive = {
                'input_ids': p_input_ids,
                'attention_mask': p_attention_mask
            }

            # Forward pass
            anchor_output = model(anchor, a_styolometric_features)
            positive_output = model(positive, p_styolometric_features)

            loss = criterion(anchor_output, positive_output)

            # Backward pass
            loss.backward()

            current_step += 1

            # Gradient accumulation
            if current_step == accumulation_steps:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                current_step = 0

            
            if (i+1) % 50 == 0:
                dists = cosine_similarity(anchor_output, positive_output)
                pos_dist = dists.diagonal()
                neg_dist = dists.flatten()[1:].view(dists.shape[0]-1, dists.shape[0]+1)[:,:-1].flatten()

                preds = dists.argmax(dim=1) == torch.arange(dists.shape[0]).to(device)
                labels = torch.ones_like(preds)

                # Calculate F1 score
                f1_score = f1(preds, labels)

                acc = (preds == labels).float().mean()

                # Compute MRR
                rank = dists.argsort(dim=1,descending=True, stable=True).argsort(dim=1).diagonal()
                rank = 1 / (rank+1)
                mrr = torch.mean(rank)

                logger.warning(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}, F1: {f1_score:.4f}, Acc: {acc:.4f}, MRR {mrr:.4f} , Time: {time.time() - start_time:.2f}s")
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
        if epoch % 5 == 0:
            logger.warning("Peforming full evaluation...")
            j = len(test_dataloader)
        else:
            logger.warning("Peforming mini evaluation...")
            j = mini_test_size # Using a subset of the test set for faster evaluation
        model.eval()
        f1_scores = []
        acc_scores = []
        mrr_scores = []
        pos_dists = []
        neg_dists = []
        with torch.no_grad():
            start_time = time.time()
            for j, batch in enumerate(test_dataloader):
                if j >= mini_test_size:
                    break

                a_input_ids = batch['anchor_input_ids']
                p_input_ids = batch['positive_input_ids']
                a_attention_mask = batch['anchor_attention_mask']
                p_attention_mask = batch['positive_attention_mask']
                a_styolometric_features = batch['anchor_styolometric_features']
                p_styolometric_features = batch['positive_styolometric_features']

                a_input_ids = a_input_ids.to(device)
                p_input_ids = p_input_ids.to(device)
                a_attention_mask = a_attention_mask.to(device)
                p_attention_mask = p_attention_mask.to(device)
                a_styolometric_features = a_styolometric_features.to(device)
                p_styolometric_features = p_styolometric_features.to(device)

                # Forward pass
                anchor = {
                    'input_ids': a_input_ids,
                    'attention_mask': a_attention_mask
                }
                positive = {
                    'input_ids': p_input_ids,
                    'attention_mask': p_attention_mask
                }

                # Forward pass
                anchor_output = model(anchor, a_styolometric_features)
                positive_output = model(positive, p_styolometric_features)

                dists = cosine_similarity(anchor_output, positive_output)
                pos_dist = dists.diagonal()
                neg_dist = dists.flatten()[1:].view(dists.shape[0]-1, dists.shape[0]+1)[:,:-1].flatten()
                preds = dists.argmax(dim=1) == torch.arange(dists.shape[0]).to(device)
                labels = torch.ones_like(preds)

                # Calculate F1 score
                f1_score = f1(preds, labels)

                acc = (preds == labels).float().mean()

                # Compute MRR
                rank = dists.argsort(dim=1,descending=True, stable=True).argsort(dim=1).diagonal()
                rank = 1 / (rank+1)
                mrr = torch.mean(rank)

                f1_scores.append(f1_score)
                acc_scores.append(acc.cpu().item())
                mrr_scores.append(mrr.cpu().item())
                pos_dists.append(pos_dist)
                neg_dists.append(neg_dist)

            end_time = time.time()
            logger.warning(f"Test F1 Score: {np.mean(f1_scores):.4f}")
            logger.warning(f"Test Acc: {np.mean(acc_scores):.4f}")
            logger.warning(f"Test MRR: {np.mean(mrr_scores):.4f}")
            logger.warning(f"Test Pos Dist: {torch.median(torch.cat(pos_dists)):.4f}")
            logger.warning(f"Test Neg Dist: {torch.median(torch.cat(neg_dists)):.4f}")
            logger.warning(f"Test Time per step: {(end_time - start_time) / j:.2f}s")

        model.train()

        # Save Model
        if not os.path.exists("models"):
            os.makedirs("models")
        torch.save(model.state_dict(), f"models/embed/{VARIANT}_model_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    main()