import torch
import torch.nn as nn
import torch.nn.functional as F
from torchpq.index import IVFPQIndex
import pandas as pd
import os
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from models.SiameseModel import SiameseModel
from utils.data import tokenize_df

from transformers.utils import logging

def main():

    # Logger
    logger = logging.get_logger("transformers")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load top-k dataset
    DATA_PATH = 'dataset'
    DATASET = "blogtext"
    VARIANT = "10"
    MODEL = "bert"
    print(f"Using dataset: {DATASET}, variant: {VARIANT}")
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
    model = SiameseModel(encoder)

    # Load model weights
    model_path = f"models/{MODEL}.pt"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)

    logger.warning("Tokenizing Dataset...")
    train_ids = train_df['id']
    train_texts = train_df['text']
    if os.path.exists(os.path.join(DATA_PATH, DATASET + "_train_" + VARIANT + "_" + MODEL +   "_ids.pt")) and os.path.exists(os.path.join(DATA_PATH, DATASET + "_train_" + VARIANT + "_attention_mask.pt")):
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

    train_stylometric = torch.tensor(train_stylometric).float()
    test_stylometric = torch.tensor(test_stylometric).float()

    # Calculate embeddings for train set
    logger.warning("Calculating train embeddings...")

    BATCH_SIZE = 32
    EMBEDDING_SIZE = 768

    if os.path.exists(f"{DATA_PATH}/{DATASET}_train_{VARIANT}_{MODEL}_embeddings.pt"):
        logger.warning("Loading pre-computed train embeddings...")
        train_embeddings = torch.load(f"{DATA_PATH}/{DATASET}_train_{VARIANT}_{MODEL}_embeddings.pt")
    else:
        train_embeddings = torch.zeros((len(train_tokens['input_ids']), EMBEDDING_SIZE))
        with torch.no_grad():
            for i in tqdm(range(0, len(train_tokens['input_ids']), BATCH_SIZE)):
                batch_ids = train_tokens['input_ids'][i:i+BATCH_SIZE].to(device)
                batch_attention_mask = train_tokens['attention_mask'][i:i+BATCH_SIZE].to(device)
                batch_stylometric = train_stylometric[i:i+BATCH_SIZE].to(device)
                batch_input = {
                    'input_ids': batch_ids,
                    'attention_mask': batch_attention_mask
                }

                batch_embeddings = model(batch_input, batch_stylometric)
                train_embeddings[i:i+BATCH_SIZE] = batch_embeddings.cpu()

        # Save
        if not os.path.exists("data"):
            os.makedirs("data")
        torch.save(train_embeddings, f"{DATA_PATH}/{DATASET}_train_{VARIANT}_{MODEL}_embeddings.pt")

    # Get test embeedings
    logger.warning("Calculating test embeddings...")
    if os.path.exists(f"{DATA_PATH}/{DATASET}_test_{VARIANT}_{MODEL}_embeddings.pt"):
        logger.warning("Loading pre-computed test embeddings...")
        test_embeddings = torch.load(f"{DATA_PATH}/{DATASET}_test_{VARIANT}_{MODEL}_embeddings.pt")
    else:
        test_embeddings = torch.zeros((len(test_tokens['input_ids']), EMBEDDING_SIZE))
        with torch.no_grad():
            for i in tqdm(range(0, len(test_tokens['input_ids']), BATCH_SIZE)):
                batch_ids = test_tokens['input_ids'][i:i+BATCH_SIZE].to(device)
                batch_attention_mask = test_tokens['attention_mask'][i:i+BATCH_SIZE].to(device)
                batch_stylometric = test_stylometric[i:i+BATCH_SIZE].to(device)
                batch_input = {
                    'input_ids': batch_ids,
                    'attention_mask': batch_attention_mask
                }

                batch_embeddings = model(batch_input, batch_stylometric)
                test_embeddings[i:i+BATCH_SIZE] = batch_embeddings.cpu()

        # Save
        if not os.path.exists("data"):
            os.makedirs("data")
        torch.save(test_embeddings, f"{DATA_PATH}/{DATASET}_test_{VARIANT}_{MODEL}_embeddings.pt")


    index = IVFPQIndex(
        d_vector=EMBEDDING_SIZE,
        n_subvectors=32,
        n_cells=1024,
        initial_size=2048,
        distance="cosine",
    )

    train_embeddings = train_embeddings.T.contiguous()
    train_embeddings = train_embeddings.to(device)
    train_embeddings = F.normalize(train_embeddings, p=2, dim=1)
    index.train(train_embeddings)
    index.add(train_embeddings)

    test_embeddings = test_embeddings.T.contiguous()
    test_embeddings = test_embeddings.to(device)
    test_embeddings = F.normalize(test_embeddings, p=2, dim=1)

    # First Consider Classification

    # Query the index for each test embedding
    logger.warning("Querying the index for test embeddings...")
    k = 20
    index.n_probe = 32
    topk_values, topk_ids = index.search(test_embeddings, k=k)

    # Convert results to author ids
    train_ids = torch.tensor(train_ids.to_numpy(), dtype=torch.int64, device=device)
    author_ids = train_ids[topk_ids]

    topk_values += 1

    topk_values = topk_values / topk_values.sum(dim=1, keepdim=True)

    classification = torch.zeros(topk_values.shape[0], dtype=torch.int64)
    for i in range(topk_values.shape[0]):
        unique_authors = author_ids[i].unique(sorted=True)
        mask = author_ids[i] == unique_authors.unsqueeze(1)
        confidence = (topk_values[i] * mask).sum(dim=1)
        classification[i] = unique_authors[confidence.argmax()]

    test_ids = torch.tensor(test_ids.to_numpy(), dtype=torch.int64)

    # Calculate accuracy
    acc = (classification == test_ids).sum().item() / len(test_ids)
    logger.warning(f"Classification Accuracy: {acc:.4f}")

    # Retrival Problem
    # Get count of each author in the train set
    author_counts = train_df['id'].value_counts().to_dict()

    f1_scores = []
    precisions = []
    recalls = []

    for i in range(len(test_ids)):
        author_id = test_ids[i]
        test_embed = test_embeddings[:, i]
        if author_id.item() not in author_counts:
            continue
        author_count = author_counts[author_id.item()]

        # Get the top-k results for the test embedding
        topk_values, topk_ids = index.search(test_embed.unsqueeze(1), k=author_count)
        topk_ids = topk_ids.squeeze(0)

        retrieved_authors = train_ids[topk_ids]

        relevant = (retrieved_authors == author_id).sum().item()
        precision = relevant / len(retrieved_authors)
        recall = relevant / author_count
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        f1_scores.append(f1_score)
        precisions.append(precision)
        recalls.append(recall)

    avg_f1 = sum(f1_scores) / len(f1_scores)
    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)
    logger.warning(f"Retrieval F1 Score: {avg_f1:.4f}")
    logger.warning(f"Retrieval Precision: {avg_precision:.4f}")
    logger.warning(f"Retrieval Recall: {avg_recall:.4f}")

if __name__ == "__main__":
    main()
