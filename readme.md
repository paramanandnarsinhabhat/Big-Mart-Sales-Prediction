
# Big Mart Sales Prediction

## Overview

This project aims to predict the sales of products in various Big Mart outlets. It involves data preprocessing, implementing a neural network model using TensorFlow and Keras, and predicting the sales based on test data. The project is structured to handle data processing, model training, and prediction generation seamlessly.

## File Structure

- `data/`
  - `sample_submission_hP4II7x.csv` - A sample submission file in the correct format.
  - `test_FewQE9B.csv` - The test dataset containing features without sales figures.
  - `train_XnW6LSF.csv` - The training dataset with both features and sales figures.
- `notebook/`
  - `preprocess_retail.ipynb` - Jupyter notebook containing data preprocessing steps and model training.
- `source/`
  - `preprocess.py` - Python script for data preprocessing.
- `submission_file/`
  - `submission.csv` - The output file containing the predicted sales.
- `.gitignore` - Specifies intentionally untracked files to ignore.
- `requirements.txt` - Contains all the necessary packages to be installed.

## Requirements

- pandas==1.4.2
- scikit-learn==1.1.1
- numpy==1.22.3
- tensorflow==2.9.1

## Setup

Ensure that you have Python installed on your system. You can install all dependencies by running:


pip install -r requirements.txt
```

## Running the Code

1. First, ensure that the datasets are placed in the `data/` directory.
2. Run the preprocessing script to clean and transform the data:


   python source/preprocess.py


3. Open the `preprocess_retail.ipynb` notebook in a Jupyter environment to train the model:


   jupyter notebook notebook/preprocess_retail.ipynb


   Execute all cells in the notebook to train the model and generate predictions.

4. The predicted sales are saved in `submission.csv` in the `submission_file/` directory. 

## Model Details

The model is a simple feedforward neural network with ReLU activation in the hidden layers and the output layer. The architecture is as follows:

- Input Layer: Matches the number of features
- Hidden Layer 1: 64 neurons with ReLU activation
- Hidden Layer 2: 32 neurons with ReLU activation
- Output Layer: 1 neuron with ReLU activation (to ensure non-negative predictions)

The model is compiled with the Adam optimizer and mean squared error loss function. It is trained for 50 epochs with a validation split of 20%.

