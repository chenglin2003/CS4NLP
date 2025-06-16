# CS4NLP

## Installation
```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Running
To pre-process all datasets, please run
```bash
python preprocess_data.py
```

To perform training for the model embeddings, please run
```bash
python train_nce.py
```

To evaluate on Information Retrival, please run
```bash
python ir.py
```

To train and evaluate on classification, please run
```bash
python classification.py
```
