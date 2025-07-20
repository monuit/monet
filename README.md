# Monet Style Transfer with GANs

A deep learning project implementing a **Deep Convolutional Generative Adversarial Network (DCGAN)** to generate Monet-style paintings. This project uses a sophisticated adversarial training approach to learn Claude Monet's distinctive artistic style and create new images that authentically capture the impressionist master's brushwork, color palette, and composition techniques.

## 🎨 Project Overview

This project is based on a **Peer-graded Assignment: Week 5: GANs** and implements a **Kaggle competition solution** for generating Monet-style paintings. The challenge is to produce 7,000–10,000 images that convincingly mimic Claude Monet's style—so convincingly that a trained classifier cannot distinguish them from genuine Monet paintings.

### Dataset
- **300 Monet paintings** (256×256 pixels, JPEG)
- **7,028 photographs** (256×256 pixels, JPEG)
- Data is provided in **TFRecord format** for efficient processing in TensorFlow

### Approach
This implementation employs a **Deep Convolutional Generative Adversarial Network (DCGAN)** composed of two complementary networks:

1. **Generator**: Creates Monet-style images from random noise, with the goal of fooling the discriminator by producing images indistinguishable from real Monet paintings
2. **Discriminator**: Accurately distinguishes genuine Monet paintings from the generator's outputs

The training alternates between:
- **Discriminator Update**: Training on batches of real paintings and generated images, minimizing classification error
- **Generator Update**: Using discriminator feedback to adjust weights, maximizing the discriminator's error on generated images

## 🚀 Features

- **DCGAN Implementation**: Deep Convolutional GAN architecture specifically designed for image generation
- **Monet Style Generation**: Creates authentic Monet-style paintings from scratch using learned artistic patterns
- **Data Preprocessing**: Comprehensive exploratory data analysis (EDA) with data inspection, visualization, and cleaning
- **Efficient Data Loading**: Uses TFRecord format for optimized TensorFlow data handling
- **Advanced Training Pipeline**: Adversarial training with alternating generator and discriminator updates
- **Visualization Tools**: Display functions for image grids and training progress monitoring
- **Kaggle Competition Ready**: Implementation optimized for competitive machine learning standards

## 📋 Dependencies

```python
# Standard library
import os
import shutil
import time
from pathlib import Path
from typing import List, Tuple, Union

# Third-party libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers
```

### Requirements
- **TensorFlow** 2.x
- **NumPy**
- **Pandas**
- **Matplotlib**
- **PIL (Pillow)**
- **Python** 3.7+

## 🔧 Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd monet
```

2. Install dependencies:
```bash
pip install tensorflow numpy pandas matplotlib pillow
```

3. Ensure you have the dataset in TFRecord format in the appropriate directory structure.

## 📊 Project Structure

```
monet/
├── README.md
├── monet_gans.ipynb          # Main Jupyter notebook
├── Input/                    # Dataset directory (if using local data)
│   └── monet_tfrec/         # TFRecord files
│       ├── monet00-60.tfrec
│       ├── monet04-60.tfrec
│       ├── monet08-60.tfrec
│       ├── monet12-60.tfrec
│       └── monet16-60.tfrec
└── .git/                    # Git repository
```

## 🎯 Usage

### Running the Notebook

1. Launch Jupyter Notebook:
```bash
jupyter notebook monet_gans.ipynb
```

2. The notebook includes the following main sections:
   - **Data Loading and Preprocessing**
   - **Exploratory Data Analysis (EDA)**
   - **GAN Model Architecture**
   - **Training Pipeline**
   - **Results Visualization**
   - **Image Generation**

### Key Functions

The notebook implements several key functions:

- `count_images_in_folder()`: Counts image files in directories with support for various image formats
- `display_images()`: Displays up to the first nine JPEG images in a directory on a 3×3 grid
- TFRecord data loading and preprocessing functions
- DCGAN generator and discriminator architecture definitions
- Adversarial training loop with alternating updates
- Visualization utilities for monitoring training progress

## 🧠 Technical Implementation

### Data Processing
- **Image Format**: 256×256 pixel JPEG images
- **Data Format**: TFRecord for efficient TensorFlow processing
- **Preprocessing**: Normalization, augmentation, and batching

### Model Architecture
The project implements a **DCGAN architecture** with:
- **Generator Network**: Deep convolutional network that transforms random noise into authentic Monet-style images
- **Discriminator Network**: Convolutional classifier that distinguishes between real Monet paintings and generated images
- **Adversarial Loss**: Minimax game formulation where generator maximizes discriminator error while discriminator minimizes classification error
- **Convolutional Layers**: Deep convolutional architecture optimized for high-quality image generation

### Training Process
- **Adversarial Training**: Generator and discriminator trained alternately
- **Monitoring**: Loss tracking and periodic image generation for evaluation
- **Optimization**: Adam optimizer with appropriate learning rates

## 📈 Results

The model generates high-quality Monet-style paintings that capture:
- **Color Palette**: Monet's characteristic color choices and combinations
- **Brushwork Style**: Impressionistic painting techniques
- **Composition**: Artistic composition and style elements
- **Visual Quality**: Images that are visually convincing as Monet-style paintings

## 🔬 Evaluation

The success of the model is evaluated based on:
- **Visual Quality**: Human assessment of generated images
- **Style Consistency**: Consistency with Monet's artistic style
- **Diversity**: Variety in generated outputs
- **Classifier Performance**: Ability to fool trained classifiers

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add improvement'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## 📝 License

This project is available for educational and research purposes. Please ensure appropriate attribution when using or modifying the code.

## 🔍 Technical Notes

- The implementation uses TensorFlow's high-level Keras API
- Data processing leverages TFRecord format for optimal performance
- The notebook includes comprehensive visualization and analysis tools
- Training was optimized for GPU acceleration when available

## 📚 References

- **Original Assignment**: Peer-graded Assignment: Week 5: GANs
- **Kaggle Competition**: "I'm Something of a Painter Myself" - Monet Style Transfer
- **TensorFlow Documentation**: https://www.tensorflow.org/
- **DCGAN Paper**: Radford, A., et al. "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks" (2015)
- **GAN Fundamentals**: Goodfellow, I., et al. "Generative Adversarial Networks" (2014)

---

**Note**: This project demonstrates the application of **Deep Convolutional GANs** for artistic style generation, specifically focused on Claude Monet's impressionist painting style. The implementation serves as both a comprehensive learning tool and a competitive-level solution for machine learning challenges in computer vision and generative modeling. The DCGAN architecture enables high-quality image generation through adversarial training between convolutional generator and discriminator networks.