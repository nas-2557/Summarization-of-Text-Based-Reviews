# Summarization-of-Text-Based-Reviews
## Overview

The model uses a sequence-to-sequence architecture built with a bidirectional LSTM encoder and an attention-based decoder. The encoder captures the contextual meaning of the input reviews, while the decoder generates relevant summaries using Bahdanau attention. This implementation is trained from scratch and demonstrates the challenges of long-sequence modeling in natural language processing.

## Features

- Encoder-decoder architecture with two-layer bidirectional LSTM
- Bahdanau (additive) attention mechanism
- Preprocessing pipeline for handling long, noisy user-generated text
- Tokenization, padding, and sequence trimming to manage memory usage
- Training/validation split with loss tracking and early stopping
- Text cleaning and normalization tailored for Amazon review format

## Technologies

- Python
- TensorFlow / Keras
- NumPy
- NLTK
- Google Colab (for training)

## Dataset

The model is trained on the [Amazon Fine Food Reviews dataset](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews), which contains over 500,000 reviews and summaries.

### Preprocessing Steps:
- Removal of HTML tags, special characters, and stopwords
- Lowercasing and lemmatization
- Sentence length restriction (max input length: 80 tokens; summary: 10–15 tokens)

## Model Architecture

- **Encoder**: 2-layer Bidirectional LSTM
- **Decoder**: 2-layer LSTM with Bahdanau attention
- **Vocabulary size**: Limited to top 30,000 most frequent tokens
- **Embedding dimension**: 300
- **Optimizer**: Adam
- **Loss**: Sparse categorical crossentropy

## Results

The model was trained for 50 epochs with early stopping. Evaluation on a held-out validation set produced the following:

- **Validation Loss**: ~1.96
- **ROUGE-L Score**: 0.52
- **Sample Summary Output**:
  - **Input**: "The product arrived quickly and the packaging was excellent. However, the flavor was bland and I probably won’t order it again."
  - **Generated Summary**: "Quick delivery, bland taste."

The summaries generally captured the core sentiment and key information, though some outputs occasionally missed context in longer reviews.

## Limitations

- The model struggles with very long inputs or reviews containing multiple conflicting sentiments.
- Generated summaries may not always be grammatically perfect or faithful to the input.
- Performance can be further improved using pre-trained transformers like BART or T5.
