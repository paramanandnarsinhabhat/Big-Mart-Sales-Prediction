# Big Mart Sales Prediction Project Documentation

## Executive Summary

The Big Mart Sales Prediction initiative is a sophisticated analytical model designed to forecast product sales across various Big Mart retail outlets. The project encompasses rigorous data preprocessing techniques, the deployment of an advanced neural network utilizing TensorFlow and Keras frameworks, and the subsequent execution of predictive modeling. This comprehensive strategy ensures a streamlined process from data ingestion to prediction output.

## Directory Structure

- `data/`: This directory hosts the datasets essential for the analysis and modeling.
  - `sample_submission_hP4II7x.csv`: A template showcasing the expected format for submissions.
  - `test_FewQE9B.csv`: The test dataset, comprising features devoid of the target variable (sales).
  - `train_XnW6LSF.csv`: The training dataset, inclusive of both the predictive features and target variable.
- `notebook/`: Contains the Jupyter notebooks that detail the preprocessing and model training.
  - `preprocess_retail.ipynb`: A Jupyter notebook that outlines the data preprocessing and model training pipeline.
- `source/`: Includes scripts that perform data processing.
  - `preprocess.py`: A Python script designated for the preprocessing of data.
- `submission_file/`: Contains the results of the predictive model.
  - `submission.csv`: The file that records the predicted sales figures.
- `.gitignore`: A file that lists the entities to be omitted from version control.
- `requirements.txt`: A manifest file listing all dependencies required by the project.

## Prerequisites

- pandas==1.4.2
- scikit-learn==1.1.1
- numpy==1.22.3
- tensorflow==2.9.1

## Installation Instructions

Prior to execution, ensure Python is correctly installed on your local machine. Install the necessary dependencies via the following command:

```bash
pip install -r requirements.txt
```

## Operational Guide

1. Position the datasets within the `data/` directory prior to initiating the model.
2. Execute the data preprocessing script to sanitize and format the data appropriately:

   ```bash
   python source/preprocess.py
   ```

3. Navigate to the `preprocess_retail.ipynb` notebook within a Jupyter environment to commence model training:

   ```bash
   jupyter notebook notebook/preprocess_retail.ipynb
   ```

   Progress through the notebook by executing all cells to train the model and generate the predictive outcomes.

4. Post-analysis, the model's sales predictions are systematically stored in the `submission.csv` file within the `submission_file/` directory.

## Model Architecture

The constructed model is a multilayer perceptron (MLP), a type of feedforward neural network, with the following configuration:

- **Input Layer**: Corresponds to the input feature count.
- **Hidden Layer 1**: Comprises 64 neurons, activated by the ReLU function.
- **Hidden Layer 2**: Contains 32 neurons, also utilizing ReLU activation.
- **Output Layer**: Consists of a single neuron with ReLU activation to ensure the generation of non-negative sales values.

The model's compilation leverages the Adam optimization algorithm and is quantified using the mean squared error loss function. It undergoes a training regime spanning 50 epochs with a validation split of 20% to ensure robust performance.

