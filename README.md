# BERT Sentiment Analysis (Transformer-based)

Sentiment classification using a transformer encoder with multi-head self-attention and custom word-level tokenization.

## Overview

Word-level tokenization + custom BERT-inspired transformer model with:
- Single-head and multi-head attention implementations
- Positional encoding
- Encoder blocks with residual connections
- Early stopping and gradient clipping

## Data Preprocessing

- Load CSV and remove null values
- Lowercase text
- Remove URLs, mentions (@), hashtags
- Keep only letters and spaces
- Strip extra whitespace

## Tokenization

**Word-Level (from scratch)**
- Build vocabulary from unique words in corpus
- Create `stoi` (string-to-index) and `itos` (index-to-string) mappings
- Add special tokens: `<PAD>` (index 0) and `<UNK>` (unknown words)
- Encode sentences: split into words → map to indices
- Pad/truncate to fixed length (BLOCK_SIZE = 24)

```python
stoi = {word: idx for idx, word in enumerate(unique_words)}
encoded = [stoi.get(w, stoi['<UNK>']) for w in text.split()]
```

## Model Architecture

### Embeddings
- **InputEmbedding**: Word embedding (64-dim) + positional encoding
- **PositionalEncoding**: Sinusoidal positional information added to each token

### Single-Head Attention (v1)
- Query, Key, Value projections
- Scaled dot-product attention
- Attention mask for padding tokens
- Dropout (0.1)

### Multi-Head Attention (v2)
- Multiple attention heads (num_heads = 2)
- Split embeddings across heads
- Concatenate and project outputs
- Same masking and dropout

### Encoder Block
- Multi-head self-attention + residual connection + layer norm
- Feed-forward network (embedding_dim → 4*embedding_dim → embedding_dim) + residual + layer norm
- ReLU activation, dropout (0.1)

### Classification Head
- Mean pooling over sequence (ignore padding)
- Linear layer to num_classes

## Training

**Setup**
- 90/10 train/val split
- Batch size: 32
- Optimizer: AdamW (lr=1e-3, weight_decay=0.01)
- Loss: CrossEntropyLoss
- Device: GPU if available, else CPU

**Features**
- Early stopping (patience=2)
- Gradient clipping (max_norm=1.0)
- Model checkpointing (saves best validation loss)

**Epochs**: 15 (or until early stopping)

## Inference

```python
def predict_multiple_sentiments(sentences):
    # Preprocess & encode each sentence
    # Batch predict with model
    # Return: [(sentence, sentiment_label, confidence), ...]
```

## Key Parameters

| Parameter | Value |
|-----------|-------|
| BLOCK_SIZE | 24 |
| embedding_dim | 64 (v1), 128 (v2) |
| num_heads | 2 |
| num_layers | 2 |
| BATCH_SIZE | 32 |
| vocab_size | Unique words in dataset |
| num_classes | Number of sentiment classes |

## Dependencies

- torch
- pandas
- scikit-learn (LabelEncoder)
- math
- re
