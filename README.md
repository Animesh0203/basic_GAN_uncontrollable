# Generative Adversarial Networks (GANs)

This repository contains an implementation of Generative Adversarial Networks (GANs) using PyTorch. GANs are deep learning models used for generative modeling, capable of generating new data that resembles a given training dataset.

## Getting Started

To get started with this project, follow the instructions below.

### Prerequisites

- Python 3.x
- PyTorch
- torchvision
- tqdm
- matplotlib

### Installation
 ```python:
git clone https://github.com/your-username/gans.git

pip install -r requirements.txt

python main.py
```
The training progress and generated images will be displayed in the console output.

Dataset
The CIFAR-100 dataset is used for training the GAN model. The dataset consists of 100 classes, with each class containing 600 images of size 32x32 pixels.

Project Structure
The project structure is organized as follows:

#### main.py: The main script to train the GAN model.
##### dataset.py: Contains the code for creating the dataset and dataloader.
##### models.py: Defines the generator and discriminator models.
##### utils.py: Contains utility functions for visualization.
##### loss.py: Defines the loss functions for the generator and discriminator.

During the training process, the GAN model generates images that gradually improve in quality, resembling the images from the CIFAR-100 dataset. The progress and generated images are displayed in the console output.

License
This project is licensed under the MIT License. See the LICENSE file for more information.

### Acknowledgements:
This implementation is based on the works of Goodfellow et al. (2014): Generative Adversarial Networks.

### References:

#### PyTorch
#### CIFAR-100 Dataset
#### Generative Adversarial Networks (GANs)
