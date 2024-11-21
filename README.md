# Deep Learning Projects

---

This repository showcases various deep learning projects, including implementations of Generative Adversarial Networks (GANs) for artistic style transfer and CNN-based classification tasks for medical image analysis.

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
