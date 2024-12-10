Here's the updated README file reflecting the addition of the **RNN for NLP Disaster Tweet Classification** project:

---

# Deep Learning Projects

---

This repository showcases various deep learning projects, including implementations of Generative Adversarial Networks (GANs) for artistic style transfer, CNN-based classification tasks for medical image analysis, and Recurrent Neural Networks (RNNs) for NLP disaster tweet classification.

## RNN for NLP Disaster Tweet Classification

The `RNN for NLP Disaster Tweet Classification.ipynb` notebook explores the use of Recurrent Neural Networks (RNNs) for text classification. Specifically, the project focuses on classifying tweets as either disaster-related or non-disaster-related.

### Project Overview

This project leverages NLP techniques and RNN-based architectures to analyze and classify tweets from the Kaggle "NLP Getting Started" competition dataset. 

### Dataset

The dataset consists of:
- **Training Data**: Tweets labeled as either disaster-related (1) or non-disaster-related (0).
- **Test Data**: Unlabeled tweets used for evaluation.

### Model Architecture

The model includes the following components:
- **Embedding Layer**: Maps words into dense, low-dimensional vector representations.
- **Bidirectional LSTM**: Captures sequential dependencies in both forward and backward directions for a better understanding of tweet context.
- **Attention Mechanism**: Focuses on the most relevant parts of each tweet, improving classification accuracy.
- **Dense Layers**: Extract high-level features and produce a binary classification output.

The notebook also explores integrating sentiment analysis features, though results showed that sentiment features alone were insufficient to improve classification performance.

### Results

- **Initial Model Accuracy**: Achieved a public leaderboard score of **0.72172**.
- **Second Model Performance**: Introduced sentiment features but resulted in a lower score of **0.56052**, highlighting the need for better feature integration.

### Future Work

- Experiment with pre-trained embeddings like GloVe or BERT.
- Reintroduce attention mechanisms for feature relevance.
- Perform hyperparameter tuning and experiment with ensemble models.

---

## Monet Style Transfer with GAN

The `monet-deep-learning-gan.ipynb` notebook demonstrates the application of GANs to transform photographs into paintings styled after Claude Monet.

### Project Overview

This project utilizes a CycleGAN architecture to perform image-to-image translation, converting landscape photographs into Monet-style artworks.

### Dataset

The dataset comprises two collections:

- **Monet Paintings**: A set of images representing Claude Monet's artworks.
- **Photographs**: A collection of landscape photographs.

Both datasets are preprocessed to a uniform size of 256x256 pixels to facilitate training.

### Model Architecture

The CycleGAN model consists of:
- **Generators**: Two networks that learn to translate images between the photograph and Monet painting domains.
- **Discriminators**: Two networks that distinguish between real and generated images in each domain.

The model is trained using a combination of adversarial and cycle-consistency losses to ensure high-quality style transfer.

### Training

Training involves alternating updates to the generators and discriminators, optimizing the networks to produce realistic Monet-style images from input photographs.

### Results

The trained model effectively generates images that closely resemble Monet's painting style when applied to new landscape photographs.

---

## Histopathologic Cancer Detection with CNN

The `histopathologic-cancer-detection.ipynb` notebook explores the application of deep learning in medical image analysis for cancer detection.

### Project Overview

This project leverages a Convolutional Neural Network (CNN) to classify pathology images into two categories: presence or absence of tumor tissue.

### Dataset

The dataset consists of:
- **Training Images**: Labeled pathology image patches provided in a train folder.
- **Test Images**: Unlabeled pathology image patches for prediction.

Images are preprocessed to a uniform size of 50x50 pixels and normalized for model training.

### Model Architecture

The CNN architecture includes:
- **Convolutional Layers**: Extract spatial features from images.
- **MaxPooling Layers**: Reduce feature map dimensions.
- **Dense Layers with Dropout**: Improve generalization and prevent overfitting.

The model is trained using the binary cross-entropy loss function, with early stopping and learning rate scheduling for optimization.

### Training

A subset of 100,000 training samples was used to manage memory constraints. The model achieved a public leaderboard score of **0.8961**, placing it at **825th**.

### Results

- **Training Accuracy**: ~85.5%
- **Validation Accuracy**: ~82.5%
- **Leaderboard Score**: 0.8961

This project demonstrates the effectiveness of CNNs for medical image classification tasks and highlights the importance of preprocessing, regularization, and resource optimization.

---

## Additional Projects

This repository also includes other deep learning projects exploring various architectures and applications.

## Requirements

- Python 3.x
- TensorFlow
- Keras
- NumPy
- Matplotlib
- pandas
- PIL

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/mrprestegaard/Deep-Learning-Projects.git
   ```

---
