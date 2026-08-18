This project demonstrates the process of generating captions for images using deep learning techniques. The code is implemented in Python and utilizes TensorFlow and Keras libraries for building and training the neural network models. The primary objective is to train a model that can predict descriptive captions for given images.

Features:

Image Feature Extraction: Utilizes the VGG16 pre-trained model to extract features from images.
Text Preprocessing: Cleans and tokenizes text data to prepare it for training.
Caption Generation: Trains an LSTM model combined with image features to generate image captions.
Evaluation: Uses BLEU scores to evaluate the quality of the generated captions.
Visualization: Includes functionality to visualize the generated captions alongside the actual images.

Dependencies:

TensorFlow
Keras
NumPy
tqdm
NLTK
PIL
Matplotlib
