import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from transformers import AutoModel, AutoTokenizer
import os
import time
import wandb
from info_nce import InfoNCE


from utils.data import BatchDataset, FunctionNegativeTripletSelector, TripletDataset, PairDataset
from utils.loss import TripletLoss, cosine_distance
from utils.metrics import f1
from utils.eval import eval_model
from models.SiameseModel import SiameseModel, StylometricModel

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
    tokenizer = AutoTokenizer.from_pretrained(
        "Alibaba-NLP/gte-base-en-v1.5", 
        use_fast=True,
        unpad_inputs=True,
        use_memory_efficient_attention=True,
        torch_dtype=torch.float16
    )
    encoder = AutoModel.from_pretrained(
        "Alibaba-NLP/gte-base-en-v1.5", 
        trust_remote_code=True, 
    )

    # Freeze the model parameters
    for param in encoder.parameters():
        param.requires_grad = False

    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = "left"

    # train_df['text'] = "query: " + train_df['text']

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

        # Split into chunks of 512 texts
        for i in range(0, len(train_texts), 512):
            chunk = train_texts[i:i+512]
            chunk = tokenizer(chunk.to_list(), return_tensors="pt", padding=True, truncation=True, max_length=512)
            if i == 0:
                train_tokens = chunk
            else:
                train_tokens['input_ids'] = torch.cat((train_tokens['input_ids'], chunk['input_ids']), dim=0)
                train_tokens['attention_mask'] = torch.cat((train_tokens['attention_mask'], chunk['attention_mask']), dim=0)
        # train_tokens = tokenizer(train_texts.to_list(), return_tensors="pt", padding=True, truncation=True, max_length=512)
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

    # Load Stylometric, ignore text and id columns
    train_stylometric = train_df.drop(columns=['id', 'text'])
    test_stylometric = test_df.drop(columns=['id', 'text'])
    train_stylometric = train_stylometric.to_numpy()
    test_stylometric = test_stylometric.to_numpy()

    # Create Dataset and DataLoader
    train_dataset = PairDataset(train_ids.to_numpy(), train_tokens, train_stylometric)
    test_dataset = TripletDataset(test_ids.to_numpy(), test_tokens, test_stylometric, test=True)

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, pin_memory=True)

    # Create Model
    model = SiameseModel(encoder)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.warning("Using device: " + str(device))
    model.to(device)

    # Loss Function and Optimizer
    # criterion = nn.TripletMarginWithDistanceLoss(
    #         distance_function=cosine_distance,
    #         margin=0.7,
    #     )
    criterion = InfoNCE(
        temperature=1e-3,
        negative_mode="paired",
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    triplet_selector = FunctionNegativeTripletSelector(
                margin=0.7,
            )

    epochs = 3
    pos_thresh = 0.4
    num_triplets_thresh = 16
    num_triplets = 0
    frozen = True
    freeze_until = 5000

    anchor_acc = None
    positive_acc = None
    negative_acc = None

    logger.warning("Training Model...")

    # Eval for baseline
    # baseline = eval_model(
    #     model,
    #     test_dataloader,
    #     criterion,
    #     device,
    #     pos_thresh=pos_thresh,
    # )
    # logger.warning(f"Baseline F1 Score: {baseline['f1_score']:.4f}")
    # logger.warning(f"Baseline Acc: {baseline['accuracy']:.4f}")
    # logger.warning(f"Baseline Pos Dist: {baseline['pos_dist']:.4f}")
    # logger.warning(f"Baseline Neg Dist: {baseline['neg_dist']:.4f}")

    # Training Loop
    for epoch in range(epochs):
        model.train()
        for i, batch in enumerate(train_dataloader):

            # Start fine-tuning encoder
            if frozen and i > freeze_until:
                logger.warning("Unfreezing encoder parameters...")
                for param in model.parameters():
                    param.requires_grad = True
                frozen = False
                for g in optimizer.param_groups:
                    g['lr'] = 1e-5

            start_time = time.time()

            a_ids, a_input_ids, a_attention_mask, a_styolometric_features, \
            p_ids, p_input_ids, p_attention_mask, p_styolometric_features = batch

            # Combine the anchor and positive samples
            a_ids = torch.cat((a_ids, p_ids), dim=0)
            a_input_ids = torch.cat((a_input_ids, p_input_ids), dim=0)
            a_attention_mask = torch.cat((a_attention_mask, p_attention_mask), dim=0)
            a_styolometric_features = torch.cat((a_styolometric_features, p_styolometric_features), dim=0)

            a_input_ids = a_input_ids.to(device)
            a_attention_mask = a_attention_mask.to(device)
            a_styolometric_features = a_styolometric_features.to(device)

            # Forward pass
            anchor = {
                'input_ids': a_input_ids,
                'attention_mask': a_attention_mask
            }

            # Forward pass
            anchor_output = model(anchor, a_styolometric_features)

            triplets = triplet_selector.get_triplets(anchor_output, a_ids)

            if len(triplets) == 0:
                continue

            anchor = anchor_output[triplets[:, 0]]
            positive = anchor_output[triplets[:, 1]]
            negative = anchor_output[triplets[:, 2]]

            # if anchor_acc is None:
            #     anchor_acc = anchor
            #     positive_acc = positive
            #     negative_acc = negative
            # else:
            #     anchor_acc = torch.cat((anchor_acc, anchor), dim=0)
            #     positive_acc = torch.cat((positive_acc, positive), dim=0)
            #     negative_acc = torch.cat((negative_acc, negative), dim=0)

            negative = negative.unsqueeze(1)

            loss = criterion(anchor, positive, negative)

            # Backward pass
            loss.backward()

            num_triplets += len(triplets)
            if num_triplets >= num_triplets_thresh:

                optimizer.step()
                
                optimizer.zero_grad(set_to_none=True)

                anchor_acc = None
                positive_acc = None
                negative_acc = None

                num_triplets = 0

            
            if (i+1) % 100 == 0:
                pos_dist = cosine_distance(anchor, positive)
                neg_dist = cosine_distance(anchor, negative.squeeze(1))

                pos_preds = pos_dist < pos_thresh
                neg_preds = neg_dist < pos_thresh

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

            if (i+1) % 1000 == 0:
                # Save Model
                if not os.path.exists("models"):
                    os.makedirs("models")
                torch.save(model.state_dict(), f"models/model_epoch_{epoch+1}_step_{i+1}.pt")

                # (Mini) Evaluation Loop
                model.eval()
                f1_scores = []
                acc_scores = []
                pos_dists = []
                neg_dists = []

                stop = 0.01 * len(test_dataloader)

                with torch.no_grad():
                    for j, batch in enumerate(test_dataloader):
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

                        # loss = criterion(anchor_output, positive_output, negative_output)

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

                    logger.warning(f"Test F1 Score: {np.mean(f1_scores):.4f}")
                    # logger.warning(f"Test Loss: {np.mean(loss):.4f}")
                    logger.warning(f"Test Acc: {np.mean(acc_scores):.4f}")
                    logger.warning(f"Test Pos Dist: {torch.median(torch.cat(pos_dists)):.4f}")
                    logger.warning(f"Test Neg Dist: {torch.median(torch.cat(neg_dists)):.4f}")

                    wandb.log({
                        "mini_test_f1": np.mean(f1_scores),
                        "mini_test_acc": np.mean(acc_scores),
                        "mini_test_pos_dist": torch.median(torch.cat(pos_dists)),
                        "mini_test_neg_dist": torch.median(torch.cat(neg_dists)),
                    })

                model.train()

        scheduler.step()
    
        # Evaluation Loop
        model.eval()
        f1_scores = []
        acc_scores = []
        with torch.no_grad():
            for j, batch in enumerate(test_dataloader):

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

            logger.warning(f"Test F1 Score: {np.mean(f1_scores):.4f}")
            logger.warning(f"Test Acc: {np.mean(acc_scores):.4f}")
            wandb.log({
                # "test_loss": np.mean(loss),s
                "test_f1": np.mean(f1_scores),
                "test_acc": np.mean(acc_scores),
            })

        # Save Model
        if not os.path.exists("models"):
            os.makedirs("models")
        torch.save(model.state_dict(), f"models/model_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    main()