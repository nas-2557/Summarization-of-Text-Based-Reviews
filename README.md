# Summarization-of-Text-Based-Reviews
## Overview

The architecture follows a sequence-to-sequence approach using a bidirectional LSTM encoder and an attention-based decoder (Bahdanau attention). The model is trained from scratch on preprocessed review-summary pairs from the dataset.

## Features

- Custom encoder-decoder architecture using TensorFlow
- Bahdanau attention mechanism for context-aware decoding
- Text preprocessing pipeline with stopword removal, lemmatization, and cleaning
- Review tokenization, padding, and dynamic sequence handling
- Inference pipeline to generate summaries for new or random reviews

## Technologies

- Python
- TensorFlow
- NumPy
- NLTK
- Google Colab

## Dataset

The model uses the [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) dataset, consisting of over 500,000 user-written product reviews. Each data point includes a long-form review and a human-written summary.

The dataset was cleaned to remove stopwords, punctuation, and irregular symbols. Texts were filtered by length to fit model input constraints.

## Model Architecture

- **Encoder**: 2-layer Bidirectional LSTM
- **Decoder**: 2-layer LSTM with Bahdanau attention
- **Embedding**: Learned embeddings with vocabulary truncation
- **Loss**: Sparse categorical crossentropy
- **Optimizer**: Adam
- **Hyperparameters**: 100 epochs, batch size 64, learning rate 0.005

## Results

The model was successfully trained and is capable of generating summaries from unseen reviews. While formal evaluation metrics (e.g., ROUGE) are not yet implemented, qualitative testing using random review inputs shows the model can extract key sentiment and content effectively in many cases.

Further evaluation and output samples will be added in future versions.

## Limitations

- The model does not currently include automated metrics for evaluation.
- Some outputs may lack fluency or miss fine-grained context.
- Results could be improved with pre-trained embeddings or transformer-based models.
