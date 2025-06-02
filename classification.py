import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import numpy as np

from models.SiameseModel import SiameseModel
from models.ClassificationModel import ClassificationModel
from utils.data import tokenize_df, ClassificationDataset

from transformers.utils import logging

def main():

    # Logger
    logger = logging.get_logger("transformers")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load top-k dataset
    # Load top-k dataset
    DATA_PATH = 'dataset'
    DATASET = "blogtext"
    VARIANT = "10"
    MODEL = "bert"
    NUM_CLASSES = 50
    print(f"Using dataset: {DATASET}, variant: {VARIANT}, model: {MODEL}, num_classes: {NUM_CLASSES}")
    train_df = pd.read_csv(os.path.join(DATA_PATH, DATASET + "_train_" + VARIANT + ".csv"))
    test_df = pd.read_csv(os.path.join(DATA_PATH, DATASET + "_test_" + VARIANT + ".csv"))

    # Load Model and Tokenizer
    logger.warning("Loading Model and Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "distilbert/distilbert-base-uncased" if MODEL == "bert" else "Alibaba-NLP/gte-base-en-v1.5",
        use_fast=True,
        unpad_inputs=True,
        use_memory_efficient_attention=True,
        torch_dtype=torch.float16
    )
    encoder = AutoModel.from_pretrained(
        "distilbert/distilbert-base-uncased" if MODEL == "bert" else "Alibaba-NLP/gte-base-en-v1.5",
        trust_remote_code=True, 
    )
    embedding_model = SiameseModel(encoder)

    # Load model weights
    model_path = f"models/{MODEL}.pt"
    embedding_model.load_state_dict(torch.load(model_path, map_location=device))

    # Freeze the embedding model
    for param in embedding_model.parameters():
        param.requires_grad = False

    model = ClassificationModel(embedding_model, num_classes=NUM_CLASSES)

    logger.warning("Tokenizing Dataset...")
    train_ids = train_df['id']
    train_texts = train_df['text']
    if os.path.exists(os.path.join(DATA_PATH, DATASET + "_train_" + VARIANT + "_" + MODEL +  "_ids.pt")) and os.path.exists(os.path.join(DATA_PATH, DATASET + "_train_" + VARIANT + "_attention_mask.pt")):
        logger.warning("Loading pre-tokenized train dataset...")
        train_tokens = torch.load(os.path.join(DATA_PATH, DATASET + "_train_" + VARIANT + "_" + MODEL +   "_ids.pt"))
        train_attention_mask = torch.load(os.path.join(DATA_PATH, DATASET + "_train_" + VARIANT + "_" + MODEL +   "_attention_mask.pt"))
        train_tokens = {
            'input_ids': train_tokens,
            'attention_mask': train_attention_mask
        }
    else:
        logger.warning("Tokenizing train dataset...")
        train_tokens = tokenize_df(tokenizer, train_texts)
        torch.save(train_tokens['input_ids'], os.path.join(DATA_PATH, DATASET + "_train_" + VARIANT + "_" + MODEL +   "_ids.pt"))
        torch.save(train_tokens['attention_mask'], os.path.join(DATA_PATH, DATASET + "_train_" + VARIANT + "_" + MODEL +   "_attention_mask.pt"))

    test_ids = test_df['id']
    test_text = test_df['text']
    if os.path.exists(os.path.join(DATA_PATH, DATASET + "_test_" + VARIANT + "_" + MODEL +   "_ids.pt")) and os.path.exists(os.path.join(DATA_PATH, DATASET + "_test_" + VARIANT + "_attention_mask.pt")):
        logger.warning("Loading pre-tokenized test dataset...")
        test_tokens = torch.load(os.path.join(DATA_PATH, DATASET + "_test_" + VARIANT + "_" + MODEL +   "_ids.pt"))
        test_attention_mask = torch.load(os.path.join(DATA_PATH, DATASET + "_test_" + VARIANT + "_" + MODEL +   "_attention_mask.pt"))
        test_tokens = {
            'input_ids': test_tokens,
            'attention_mask': test_attention_mask
        }
    else:
        logger.warning("Tokenizing test dataset...")
        test_tokens = tokenize_df(tokenizer, test_text)
        torch.save(test_tokens['input_ids'], os.path.join(DATA_PATH, DATASET + "_test_" + VARIANT + "_" + MODEL +   "_ids.pt"))
        torch.save(test_tokens['attention_mask'], os.path.join(DATA_PATH, DATASET + "_test_" + VARIANT + "_" + MODEL +   "_attention_mask.pt"))

    # Load Stylometric, ignore text and id columns
    train_stylometric = train_df.drop(columns=['id', 'text'])
    test_stylometric = test_df.drop(columns=['id', 'text'])
    train_stylometric = train_stylometric.to_numpy()
    test_stylometric = test_stylometric.to_numpy()

    # Map all ids to integers
    ids = pd.concat([train_ids, test_ids]).unique()
    id_to_int = {id_: i for i, id_ in enumerate(ids)}
    train_ids = train_ids.map(id_to_int)
    test_ids = test_ids.map(id_to_int)

    train_dataset = ClassificationDataset(
        id=train_ids.to_numpy(),
        tokens=train_tokens,
        stylometric_features=train_stylometric,
    )

    test_dataset = ClassificationDataset(
        id=test_ids.to_numpy(),
        tokens=test_tokens,
        stylometric_features=test_stylometric,
    )

    # Create DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True, 
        pin_memory=True,
        drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=32, 
        shuffle=False, 
        pin_memory=True,
    )

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    num_epochs = 10

    for epoch in range(num_epochs):

        model.train()

        for step, batch in enumerate(train_loader):
            
            labels, input_ids, attention_mask, stylometric_features = batch

            labels = labels.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            stylometric_features = stylometric_features.to(device)

            batch_input = {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }

            logits = model(batch_input, stylometric_features)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if (step+1) % 10 == 0:
                preds = torch.argmax(logits, dim=1)
                accuracy = (preds == labels).float().mean().item()
                logger.warning(f"Epoch {epoch+1}/{num_epochs}, Step {step+1}/{len(train_loader)}, Accuracy: {accuracy:.4f}, Loss: {loss.item()}")

            if (step+1) % 100 == 0:
                model.eval()
                accs = []
                with torch.no_grad():
                    for j, batch in enumerate(test_loader):
                        if j >= 10:
                            break
                        labels, input_ids, attention_mask, stylometric_features = batch

                        labels = labels.to(device)
                        input_ids = input_ids.to(device)
                        attention_mask = attention_mask.to(device)
                        stylometric_features = stylometric_features.to(device)

                        batch_input = {
                            'input_ids': input_ids,
                            'attention_mask': attention_mask
                        }

                        logits = model(batch_input, stylometric_features)
                        preds = torch.argmax(logits, dim=1)
                        accs.append((preds == labels).float().mean().item())
                avg_acc = np.mean(accs)
                logger.warning(f"Epoch {epoch+1}/{num_epochs}, Step {step+1}/{len(train_loader)}, Validation Accuracy: {avg_acc:.4f}")

        # Validation
        model.eval()
        accs = []
        with torch.no_grad():
            for batch in test_loader:
                labels, input_ids, attention_mask, stylometric_features = batch

                labels = labels.to(device)
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                stylometric_features = stylometric_features.to(device)

                batch_input = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask
                }

                logits = model(batch_input, stylometric_features)
                preds = torch.argmax(logits, dim=1)
                accs.append((preds == labels).float().mean().item())

        avg_acc = np.mean(accs)
        logger.warning(f"Epoch {epoch+1}/{num_epochs}, Validation Accuracy: {avg_acc:.4f}")

        if (epoch+1) % 1 == 0:
            model_save_path = f"models/classification/{DATASET}_{VARIANT}_epoch_{epoch+1}.pt"
            torch.save(model.state_dict(), model_save_path)
            logger.warning(f"Model saved to {model_save_path}")

        scheduler.step()

if __name__ == "__main__":
    main()