# Deep Learning Projects

---

This repository showcases various deep learning projects, including implementations of Generative Adversarial Networks (GANs) for artistic style transfer.

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

## Additional Projects

This repository also includes other deep learning projects exploring various architectures and applications.

## Requirements

- Python 3.x
- TensorFlow
- Keras
- NumPy
- Matplotlib

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/mrprestegaard/Deep-Learning-Projects.git
   ```

2. Navigate to the project directory:

   ```bash
   cd Deep-Learning-Projects
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Open the Jupyter notebook:

   ```bash
   jupyter notebook monet-deep-learning-gan.ipynb
   ```

5. Follow the instructions within the notebook to train the model and generate Monet-style images.

## References

- [CycleGAN Paper](https://arxiv.org/abs/1703.10593)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras Documentation](https://keras.io/)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details. 
