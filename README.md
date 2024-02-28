# Project Overview

This project showcases an advanced machine learning pipeline for translating English to Farsi, demonstrating cutting-edge MLOps practices, including model development from scratch in PyTorch, automated workflows, and cloud deployment.

## Special Features

- **Transformer Model Implementation**: Built from the ground up based on the influential "Attention is All You Need" paper, this project utilizes PyTorch to create a powerful and efficient translation model.

- **Modular Pipeline Design**: The project adopts a modular approach, with the pipeline segmented into independent stages (Data Ingestion, Validation, Transformation, Model Training, Evaluation, and Inference). This design facilitates easier debugging, iterative development, and scalability.

- **MLOps Integration**: Demonstrates comprehensive MLOps practices by integrating continuous integration and continuous delivery (CI/CD) pipelines using GitHub Actions. This ensures that any changes to the codebase automatically trigger workflows for testing, building, and deploying the application, maintaining the project in a release-ready state.

- **Cloud Deployment and Containerization**: The model is containerized using Docker, making it platform-independent and easily deployable on cloud services like Amazon EC2. This approach underscores the project's readiness for real-world applications and ease of use across different environments.

- **Interactive Model Access**: Utilizing Gradio for creating a user-friendly web interface, the project allows easy access to the translation model through a simple interface, enabling users to experience the model's capabilities directly.



## Getting Started

### Prerequisites
- Conda (Miniconda or Anaconda)
- Python 3.8

### Installation Steps
1. **Clone the Repository**: 
   ```bash
   git clone https://github.com/omid-sar/End_to_End_Transformer_En_Fa.git
   ```
    ```bash
    cd End_to_End_Transformer_En_Fa
    ```
2. **Create and Activate a Conda Environment**: 

    ```bash
    conda create -n transformer_pytorch python=3.8 -y
    ```
   ```bash
   conda activate transformer_pytorch
   ```

3. **Install the Requirements**: 
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**: 
   ```bash
   python app.py
   ```

   After running, access the application via your local host and specified port.

## Contact
- **Author**: Omid Sar
- **Email**: [mr.omid.sardari@gmail.com](mailto:mr.omid.sardari@gmail.com)

---

# AWS CI/CD Deployment with Github Actions

## Overview
This guide provides a comprehensive walkthrough for deploying a Dockerized application on AWS EC2 using Github Actions for continuous integration and continuous deployment (CI/CD).

## Prerequisites
- AWS Account
- Github Account

## Steps

### 1. AWS Console Preparation
   - **Login**: Ensure you are logged into your AWS console.
   - **Create IAM User**: Ensure the user has the following policies:
     - `AmazonEC2ContainerRegistryFullAccess`
     - `AmazonEC2FullAccess`
   - **Create ECR Repository**: Note down the URI.

### 2. EC2 Instance Setup
   - **Create an EC2 Instance**: Preferably Ubuntu.
   - **Install Docker on EC2**: 
     - Optional: Update and upgrade the system.
       ```bash
       sudo apt-get update -y
       ```
       ```bash
       sudo apt-get upgrade
       ```

     - Required: Install Docker.
       ```bash
       curl -fsSL https://get.docker.com -o get-docker.sh
       ```
       ```bash
       sudo sh get-docker.sh
       ```
       ```bash
       sudo usermod -aG docker ubuntu
       ```
       ```bash
       sudo usermod -aG docker ubuntu
       ```
       ```bash
       newgrp docker
       ```

### 3. Configure Self-hosted Runner on Github
   - Navigate to your repository's settings.
   - Go to Actions > Runners.
   - Click "New self-hosted runner" and follow the instructions.

### 4. Set Up Github Secrets
   - Navigate to your repository's settings.
   - Go to Secrets and add the following:
     - `AWS_ACCESS_KEY_ID`
     - `AWS_SECRET_ACCESS_KEY`
     - `AWS_REGION`
     - `AWS_ECR_LOGIN_URI`
     - `ECR_REPOSITORY_NAME`

## Deployment Flow
1. **Build Docker Image**: Locally or in CI/CD pipeline.
2. **Push Docker Image to ECR**: Use AWS CLI or Github Actions.
3. **Launch EC2 Instance**: Ensure it has Docker installed.
4. **Pull Docker Image on EC2**: Use AWS CLI.
5. **Run Docker Container on EC2**: Start your application.
```


















### Model Components

1. **Transformer Model and Related Components**:
   - **`src/{project_name}/models/transformer.py`**:
     - `Transformer` class
     - `ProjectionLayer` class
   - **`src/{project_name}/models/components.py`**:
     - `LayerNormalization` class
     - `FeedForwardBlock` class
     - `InputEmbeddings` class
     - `PositionalEncoding` class
     - `ResidualConnection` class
   - **`src/{project_name}/models/multi_head_attention.py`**:
     - `MultiHeadAttentionBlock` class
   - **`src/{project_name}/models/encoder_decoder.py`**:
     - `EncoderBlock` class
     - `DecoderBlock` class
     - `Encoder` class
     - `Decoder` class
   - **`src/{project_name}/models/utils.py`**:
     - `build_transformer` function

### Dataset and Data Preprocessing

2. **Data Preprocessing and Dataset Handling**:
   - **`src/{project_name}/components/data_preprocessing.py`**:
     - `get_all_sentences` function
     - `get_or_build_tokenizer` function
     - `get_ds` function
   - **`src/{project_name}/data/datasets.py`**:
     - `BilingualDataset` class

### Training and Evaluation Functions

3. **Model Training and Evaluation**:
   - **`src/{project_name}/components/model_training.py`**:
     - `train_model` function
     - `greedy_decode` function
     - `run_validation` function

### Utility and Helper Functions

4. **Utility Functions**:
   - **`src/{project_name}/utils/model_utils.py`**:
     - Any additional model-related utility functions
   - **`src/{project_name}/utils/data_utils.py`**:
     - Data-related utility functions not directly part of preprocessing or dataset handling

### Main Script

5. **Main Script**:
   - **`main.py`**:
     - This file typically contains the main execution logic, such as initializing the training process or orchestrating different components of the pipeline.

This organization ensures that each part of your Transformer model and related functionalities is logically placed within your project's structure, facilitating maintainability, scalability, and clear separation of concerns.






.
├── artifacts
│   ├── data_ingestion
│   │   └── tep_en_fa_para
│   ├── data_transformation
│   ├── data_validation
│   └── model_trainer
│       └── transformer
│           └── [model checkpoints and configurations]
├── config
│   └── config.yaml
├── logs
├── research
│   ├── [Jupyter notebooks for exploratory analysis and testing]
│   └── logs
├── src
│   ├── transformer_project
│   │   ├── components
│   │   │   ├── __init__.py
│   │   │   ├── data_ingestion.py
│   │   │   ├── data_preprocessing.py
│   │   │   │   ├── BilingualDataset
│   │   │   │   ├── get_all_sentences
│   │   │   │   ├── get_or_build_tokenizer
│   │   │   │   └── get_ds
│   │   │   ├── model_evaluation.py
│   │   │   │   └── run_validation
│   │   │   ├── model_trainer.py
│   │   │   │   ├── train_model
│   │   │   │   └── greedy_decode
│   │   │   └── data_validation.py
│   │   ├── constants
│   │   │   └── __init__.py
│   │   ├── config
│   │   │   ├── __init__.py
│   │   │   └── configuration.py
│   │   │       ├── ConfigurationManager     
│   │   │       
│   │   ├── entity
│   │   │   └── __init__.py
│   │   ├── logging
│   │   │   └── __init__.py
│   │   ├── models
│   │   │   ├── __init__.py
│   │   │   ├── components.py
│   │   │   │   ├── LayerNormalization
│   │   │   │   ├── FeedForwardBlock
│   │   │   │   ├── InputEmbeddings
│   │   │   │   ├── PositionalEncoding
│   │   │   │   └── ResidualConnection
│   │   │   ├── encoder_decoder.py
│   │   │   │   ├── EncoderBlock
│   │   │   │   ├── DecoderBlock
│   │   │   │   ├── Encoder
│   │   │   │   └── Decoder
│   │   │   ├── multi_head_attention.py
│   │   │   │   └── MultiHeadAttentionBlock
│   │   │   ├── transformer.py
│   │   │   │   ├── Transformer
│   │   │   │   └── ProjectionLayer
│   │   │   └── utils.py
│   │   │   |     └── build_transformer
|   |   │   └── common.py
|   |   │   │       |── get_weights_file_path
|   |   │   │       └── latest_weights_file_path
│   │   ├── pipeline
│   │   │   ├── __init__.py
│   │   │   ├── stage_01_data_ingestion.py
│   │   │   ├── stage_02_data_validation.py
│   │   │   ├── stage_03_data_transformation.py
│   │   │   ├── stage_04_model_trainer.py
│   │   │   └── stage_05_model_evaluation.py
│   │   └── utils
│   │       ├── __init__.py
│   │       ├── common.py
│   │       └── data_utils.py
│   └── transformer_project__init__.py
├── app.py
│   └── translate
├── main.py
├── .gitignore
├── Dockerfile
├── LICENSE
├── output.txt
├── params.yaml
├── README.md
├── requirements.txt
└── setup.py


## Project Structure

- `artifacts`
  - `data_ingestion`
    - `tep_en_fa_para`
  - `data_transformation`
  - `data_validation`
  - `model_trainer`
    - `transformer`
      - `[model checkpoints and configurations]`
- `config`
  - `config.yaml`
- `logs`
- `research`
  - `[Jupyter notebooks for exploratory analysis and testing]`
  - `logs`
- `src`
  - `transformer_project`
    - `components`
      - `__init__.py`
      - `data_ingestion.py`
      - `data_preprocessing.py`
        - `BilingualDataset` class
        - `get_all_sentences`
        - `get_or_build_tokenizer`
        - `get_ds`
      - `model_evaluation.py`
        - `run_validation`
      - `model_trainer.py`
        - `train_model`
        - `greedy_decode`
      - `data_validation.py`
    - `constants`
      - `__init__.py`
    - `config`
      - `__init__.py`
      - `configuration.py`
        - `get_config`
        - `get_weights_file_path`
        - `latest_weights_file_path`
    - `entity`
      - `__init__.py`
    - `logging`
      - `__init__.py`
    - `models`
      - `__init__.py`
      - `components.py`
        - `LayerNormalization`
        - `FeedForwardBlock`
        - `InputEmbeddings`
        - `PositionalEncoding`
        - `ResidualConnection`
      - `encoder_decoder.py`
        - `EncoderBlock`
        - `DecoderBlock`
        - `Encoder`
        - `Decoder`
      - `multi_head_attention.py`
        - `MultiHeadAttentionBlock`
      - `transformer.py`
        - `Transformer`
        - `ProjectionLayer`
        - `build_transformer`
    - `pipeline`
      - `__init__.py`
      - `stage_01_data_ingestion.py`
      - `stage_02_data_validation.py`
      - `stage_03_data_transformation.py`
      - `stage_04_model_trainer.py`
      - `stage_05_model_evaluation.py`
    - `utils`
      - `__init__.py`
      - `common.py`
      - `data_utils.py`
  - `transformer_project__init__.py`
- `app.py`
  - `translate`
- `main.py`
- `.gitignore`
- `Dockerfile`
- `LICENSE`
- `output.txt`
- `params.yaml`
- `README.md`
- `requirements.txt`
- `setup.py`
