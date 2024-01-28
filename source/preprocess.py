import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Load the training, test, and sample submission data
train_data_path = '/Users/paramanandbhat/Downloads/train_XnW6LSF.csv'
test_data_path = '/Users/paramanandbhat/Downloads/test_FewQE9B.csv'
sample_submission_path = '/Users/paramanandbhat/Downloads/sample_submission_hP4II7x.csv'

train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)
sample_submission = pd.read_csv(sample_submission_path)

# Impute missing values
train_data['Item_Weight'].fillna(train_data['Item_Weight'].mean(), inplace=True)
test_data['Item_Weight'].fillna(test_data['Item_Weight'].mean(), inplace=True)
train_data['Outlet_Size'].fillna(train_data['Outlet_Size'].mode()[0], inplace=True)
test_data['Outlet_Size'].fillna(test_data['Outlet_Size'].mode()[0], inplace=True)

# Prepare for preprocessing
categorical_cols = train_data.select_dtypes(include=['object']).columns
numerical_cols = train_data.select_dtypes(include=['float64', 'int64']).columns.drop('Item_Outlet_Sales')

# Preprocessor for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Transform training and test data
train_data_processed = preprocessor.fit_transform(train_data.drop('Item_Outlet_Sales', axis=1))
test_data_processed = preprocessor.transform(test_data)

# Define the neural network architecture with ReLU activation in the output layer
model = keras.Sequential([
    keras.layers.Input(shape=(train_data_processed.shape[1],)),
    keras.layers.Dense(units=64, activation='relu'),
    keras.layers.Dense(units=32, activation='relu'),
    keras.layers.Dense(units=1, activation='relu')  # ReLU activation in the output layer
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Splitting the data into features (X) and target (y)
X_train = train_data_processed.toarray()  # Convert to NumPy array
y_train = train_data['Item_Outlet_Sales']

# Train the model
history = model.fit(X_train, y_train, epochs=50, validation_split=0.2, verbose=2)

# Make predictions on the test data
X_test = test_data_processed.toarray()  # Convert to NumPy array
predictions = model.predict(X_test)

# Ensure all predictions are non-negative
predictions = np.maximum(predictions, 0)

# Save the submission file with adjusted predictions
sample_submission['Item_Outlet_Sales'] = predictions
sample_submission.to_csv('/Users/paramanandbhat/Downloads/submission.csv', index=False)
