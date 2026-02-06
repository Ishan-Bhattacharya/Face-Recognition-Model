# Face Recognition Model using CNN

This project implements a Convolutional Neural Network (CNN)-based face recognition system using Python and deep learning techniques. The model is trained to recognize and classify human faces from image datasets after preprocessing and data partitioning.

## Project Overview

Face recognition is a core computer vision task with applications in security, authentication, and surveillance. This project focuses on building a simple yet effective CNN-based pipeline for face recognition, covering:

- Dataset preprocessing and partitioning
- Model training using convolutional layers
- Performance evaluation on unseen data

## Features

- Custom data generator for efficient image loading
- Dataset split into training, validation, and testing sets
- CNN architecture for feature extraction and classification
- Modular Jupyter notebooks for experimentation and visualization

## Technologies Used

- Python  
- TensorFlow / Keras  
- NumPy  
- OpenCV  
- Jupyter Notebook  

## Project Structure

Face-Recognition-Model/
│
├── Data_partition.ipynb # Dataset preprocessing and splitting
├── Training.ipynb # CNN model definition and training
├── data_generator.py # Custom data loading and augmentation logic
├── README.md


## Workflow

1. **Data Preparation**
   - Face images are organized and partitioned into training, validation, and test sets.
   - Preprocessing includes resizing and normalization.

2. **Model Training**
   - A CNN is used to automatically extract facial features.
   - The network is trained using supervised learning.

3. **Evaluation**
   - Model performance is evaluated on unseen test data.

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Ishan-Bhattacharya/Face-Recognition-Model.git
   cd Face-Recognition-Model
   
2. Install dependencies:
pip install tensorflow numpy opencv-python

3. Run the notebooks in order:

Data_partition.ipynb

Training.ipynb


