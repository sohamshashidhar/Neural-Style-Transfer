# Neural Style Transfer Project

## Overview
This project implements Neural Style Transfer using a pre-trained VGG model. It applies the artistic style of one image (style image) to another (content image) to create a visually appealing artwork.

## Features
- Load and preprocess content and style images.
- Extract deep features using a VGG-based model.
- Compute content and style losses using feature maps.
- Optimize an initial image to blend content and style.
- Save and visualize the generated image.

## Technologies Used
- Python
- PyTorch
- torchvision
- PIL (Pillow)
- NumPy

## Installation
### Prerequisites
Ensure you have Python installed (recommended version: 3.8 or later).

### Clone the Repository
```sh
git clone https://github.com/yourusername/style-transfer-project.git
cd style-transfer-project
```

### Create a Virtual Environment
```sh
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### Install Dependencies
```sh
pip install -r requirements.txt
```

## Usage
### Train the Model
```sh
python train.py
```

### Input Images
Ensure you have `content.jpg` and `style.jpg` in the project folder.

### Output
The generated image is saved as `output.jpg`.

## Folder Structure
```
style_transfer_project/
│── model.py            # VGG feature extractor and gram matrix computation
│── train.py            # Training script for style transfer
│── utils.py            # Utility functions for image loading and saving
│── content.jpg         # Content image
│── style.jpg           # Style image
│── output.jpg          # Generated output image
│── README.md           # Project documentation
│── requirements.txt    # Dependencies
```

## Contributing
Feel free to open issues and submit pull requests to improve the project.

## License
This project is open-source and available under the MIT License.

## Contact
For questions or suggestions, reach out to sohamshashidhar(GITHUB).

