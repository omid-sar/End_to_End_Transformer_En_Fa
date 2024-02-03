import os
from datasets import load_dataset

# Load the dataset
dataset_name = "tep_en_fa_para"
dataset = load_dataset(dataset_name)

# Specify your custom save path
save_path = "my_dataset_directory"

# Ensure the directory exists
os.makedirs(save_path, exist_ok=True)

# Example: Save the 'train' split of the dataset as a CSV file
train_dataset = dataset["train"]
train_dataset.to_csv(os.path.join(save_path, "train.csv"))

# Similarly, you can save other splits (e.g., 'validation', 'test') as needed
