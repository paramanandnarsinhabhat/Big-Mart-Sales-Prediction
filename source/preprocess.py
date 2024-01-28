import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler



# Load the training, test, and sample submission data
train_data_path = '/Users/paramanandbhat/Downloads/train_XnW6LSF.csv'
test_data_path = '/Users/paramanandbhat/Downloads/test_FewQE9B.csv'
sample_submission_path = '/Users/paramanandbhat/Downloads/sample_submission_hP4II7x.csv'


train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)
sample_submission = pd.read_csv(sample_submission_path)

# Display the first few rows of each dataset to understand their structure
train_data_head = train_data.head()
test_data_head = test_data.head()
sample_submission_head = sample_submission.head()



print(train_data_head)
print(test_data_head)
print(sample_submission_head)

# Checking for missing values in both training and test datasets
missing_values_train = train_data.isnull().sum()
missing_values_test = test_data.isnull().sum()

print(missing_values_train)
print(missing_values_test)


# Imputing missing values
#Replace 'Item weight' with mean value
# For numerical attribute 'Item_Weight', we use the mean to impute missing values
train_data['Item_Weight'] = train_data['Item_Weight'].fillna(train_data['Item_Weight'].mean())
test_data['Item_Weight'] = test_data['Item_Weight'].fillna(test_data['Item_Weight'].mean())

# For categorical attribute 'Outlet_Size', we use the mode (most frequent value) to impute missing values
mode_outlet_size_train = train_data['Outlet_Size'].mode()[0]
mode_outlet_size_test = test_data['Outlet_Size'].mode()[0]

train_data['Outlet_Size'].fillna(mode_outlet_size_train,inplace=True)
test_data['Outlet_Size'].fillna(mode_outlet_size_test,inplace=True)

# Check if the missing values are imputed for both train and test data
imputed_missing_values_train = train_data.isnull().sum()
imputed_missing_values_test = test_data.isnull().sum()

print(imputed_missing_values_train)
print(imputed_missing_values_test)

print('Missing values have been replaced by mean and mode respectively')

#Categorical columns include object
categorical_cols = train_data.select_dtypes(include=['object']).columns

print('Categorical columns',categorical_cols)

print(train_data.dtypes)

numerical_cols = train_data.select_dtypes(include=['float64', 'int64']).columns

print('Numerical columns',numerical_cols)

#Removing the target variable from the list of numerical columns:
numerical_cols = numerical_cols.drop('Item_Outlet_Sales')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

