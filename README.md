# cat-dog-cnn
CNN model for classifying cats vs dogs using TensorFlow
# Cat vs Dog Image Classification using CNN

This repository contains a Convolutional Neural Network (CNN) implementation for binary image classification of cats and dogs. The model is built using TensorFlow and Keras, trained from scratch on a labeled dataset from Kaggle. It achieves over 85% accuracy on unseen test data.

## Dataset

- **Source**: [Kaggle - Cat and Dog Dataset](https://www.kaggle.com/datasets/tongpython/cat-and-dog)
- **Format**:
data/
├── training_set/
│ ├── cats/
│ └── dogs/
└── test_set/
├── cats/
└── dogs/
- The dataset contains 8005 training images and 2023 test images across both classes.

## Model Overview

The CNN model consists of the following architecture:

Input Layer: 150 x 150 x 3 RGB image
→ Conv2D (32 filters, 3x3) + ReLU
→ MaxPooling2D (2x2)
→ Conv2D (64 filters, 3x3) + ReLU
→ MaxPooling2D (2x2)
→ Flatten
→ Dense (128 units) + ReLU
→ Dense (1 unit) + Sigmoid


- **Loss Function**: Binary Crossentropy  
- **Optimizer**: Adam  
- **Final Activation**: Sigmoid (for binary output: 0 = Cat, 1 = Dog)

## Training Details

- **Epochs**: 20  
- **Batch Size**: 32  
- **Input Image Size**: 150 x 150  
- **Framework**: TensorFlow 2.x, Keras API

### Performance

- **Final Test Accuracy**: 85.71%  
- **Final Test Loss**: 0.3494

### Training Graphs

Add your training graphs below after uploading them to GitHub.

**Accuracy over epochs**

![Training Accuracy](insert-your-accuracy-graph.png)

**Loss over epochs**

![Training Loss](insert-your-loss-graph.png)

## Prediction on New Images

To classify new images, upload an image file and run the following Python code:

```python
from tensorflow.keras.preprocessing import image
import numpy as np

img = image.load_img("path_to_image.jpg", target_size=(150, 150))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
label = "Dog" if prediction[0][0] > 0.5 else "Cat"
print(f"Prediction: {label}")
```

Project Structure
cat-dog-cnn/
├── cat_dog_cnn.ipynb        # Main Jupyter notebook with training, evaluation, and prediction
├── requirements.txt         # Dependencies
├── README.md                # Project documentation
├── saved_model/             # Trained model (e.g., cat_dog_model.keras)
└── data/                    # Dataset directory (excluded from GitHub)


How to Run
1.Clone the repository:
git clone https://github.com/your-username/cat-dog-cnn.git
cd cat-dog-cnn

2.Install dependencies:
pip install -r requirements.txt

3.Start Jupyter Notebook and open:
jupyter notebook

4.Run the notebook cat_dog_cnn.ipynb to train, evaluate, and test predictions.


Future Improvements
Introduce data augmentation using ImageDataGenerator

Add dropout layers to prevent overfitting

Use transfer learning with pre-trained models like MobileNet or VGG16

Deploy the model with a web interface using Flask or Streamlit


