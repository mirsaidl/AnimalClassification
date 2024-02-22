import streamlit as st
import gdown
import pickle
import pathlib
import plotly.express as px
from fastai.vision.all import *

plt = platform.system()
if plt == 'Linux': 
    pathlib.WindowsPath = pathlib.PosixPath

# Download the model file from Google Drive
url = 'https://drive.google.com/uc?id=1E_nwpxAHAru84TkCkSIVztT3WI2ThoBw'
output = "animals_cls.pkl"
gdown.download(url, output, quiet=False)

# Load the model
with open(output, 'rb') as f:
    model = pickle.load(f)

# title 
st.title('Animal Classification Model')

# Get user's name
name = st.text_input("Enter your name: ")

# Check if name is provided
if name:
    st.write(f"Hi {name}, Welcome to Our Streamlit App!")
else:
    st.write("Please enter your name above.")

# uploading
file = st.file_uploader("Upload picture", type=['png', 'jpeg', 'gif', 'svg'])

if file:
    # Display the uploaded image
    st.image(file)
    
    # Convert the uploaded image to PILImage
    img = PILImage.create(file)
    
    # Perform prediction with the model
    pred, pred_id, probs = model.predict(img)
    
    # Display prediction and probability
    st.success(f"Prediction: {pred}")
    st.info(f"Probability: {probs[pred_id]*100:.1f}%")

    # Plotting
    fig = px.bar(y=probs*100, x=model.dls.vocab)
    fig.update_layout(
        yaxis_title="Probability(%)",
        xaxis_title="Animals"
    )
    st.plotly_chart(fig)

# Model Description
st.markdown("""
### Animal Classification Model Description

#### Objective:
The objective of this animal classification model is to accurately classify images of four different animals: dogs, chickens, cows, and lions. The model will be trained on a dataset containing labeled images of these animals to learn patterns and features that distinguish one animal from another.

#### Dataset:
The dataset used for training, validation, and testing consists of a collection of images, each labeled with one of the four classes: dog, chicken, cow, or lion. The dataset is preprocessed to ensure uniformity in size, color, and orientation to facilitate effective learning by the model.

#### Model Architecture:
The model architecture for this animal classification task will be a Convolutional Neural Network (CNN), a type of neural network well-suited for image classification tasks. The CNN will consist of the following layers:

1. **Input Layer**:
   - Receives input images of fixed size (e.g., 224x224 pixels).
   
2. **Convolutional Layers**:
   - Comprise multiple convolutional and pooling layers to extract features from the input images.
   - Each convolutional layer detects patterns like edges, textures, and shapes.
   - ReLU activation functions are applied to introduce non-linearity.

3. **Pooling Layers**:
   - Follow convolutional layers to reduce spatial dimensions while retaining important information.
   - Max pooling is commonly used to down-sample the feature maps.

4. **Flattening Layer**:
   - Flattens the 2D feature maps into a 1D vector for input into the fully connected layers.

5. **Fully Connected Layers** (Dense Layers):
   - Process the flattened features to learn high-level patterns.
   - Dropout layers may be added for regularization to prevent overfitting.
   - ReLU activation functions are used.

6. **Output Layer**:
   - Final layer with four nodes, one for each class (dog, chicken, cow, lion).
   - Utilizes softmax activation to output probabilities of each class.
   - The class with the highest probability is the predicted class.

#### Training:
- The model is trained using a batch size suitable for the available memory.
- Training is performed over multiple epochs, where each epoch processes the entire dataset.
- The loss function used is categorical cross-entropy, suitable for multi-class classification.
- An optimizer such as Adam is employed to update the model's weights.
- The model's performance is monitored on a separate validation set to prevent overfitting.
- Techniques like early stopping can be used to halt training if validation loss stops improving.

#### Evaluation:
- The trained model's performance is evaluated on a separate test dataset, not seen during training.
- Evaluation metrics include accuracy, precision, recall, and F1-score for each class.
- Confusion matrix visualization can provide insights into the model's performance on each class.

#### Deployment:
- Once trained and evaluated, the model can be deployed in various applications:
  - Web or mobile applications for real-time animal classification.
  - Embedded systems for on-device inference.
  - Integration into larger systems for wildlife monitoring, farm animal recognition, or pet applications.

#### Model Improvement:
- The model's performance can be further improved through:
  - Data augmentation to increase the diversity of training examples.
  - Transfer learning using pre-trained models like VGG, ResNet, or EfficientNet.
  - Hyperparameter tuning to optimize learning rate, batch size, etc.
  - Exploring more complex architectures or ensembling multiple models.

#### Summary:
This animal classification model, based on a Convolutional Neural Network, aims to accurately classify images of dogs, chickens, cows, and lions. Through training on a labeled dataset, the model learns features and patterns specific to each animal class. The model's performance is evaluated rigorously, and once deployed, it can assist in various applications requiring automated animal classification.

Model is made by Mirsaid, Instagram: mirsaid_kr

Link for Github repository and project: [GitHub Repository](https://github.com/mirsaidl/DeepLearning_models/tree/main/Animals_classifier)
"""
)
