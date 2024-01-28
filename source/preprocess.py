import pandas as pd


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





