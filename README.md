# 🫁 Kidney Disease Classification

<<<<<<< HEAD
A deep learning application that classifies kidney CT scan images to detect the presence of **cancer** (tumor) or determine the kidney is **normal**. Built with a transfer-learned **VGG16** CNN, served via a **Flask** web API, and tracked end-to-end with **MLflow** and **DVC**.

> ⚠️ **Disclaimer:** This tool is for educational purposes only and is **not** intended for clinical diagnosis or medical advice.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Pipeline Stages](#pipeline-stages)
- [Model Architecture](#model-architecture)
- [Scores](#scores)
- [Setup & Installation](#setup--installation)
- [Running the Application](#running-the-application)
- [API Reference](#api-reference)
- [Docker Deployment](#docker-deployment)
- [MLflow & DagsHub Tracking](#mlflow--dagshub-tracking)
- [Configuration](#configuration)
- [License](#license)

---

## Overview

This project uses **transfer learning on VGG16** (pre-trained on ImageNet) to classify kidney CT scan images into two categories:

- **Tumor** — cancerous tissue detected
- **Normal** — healthy kidney

The model's convolutional base is frozen and a custom classification head is trained on top. The full pipeline — from data download through evaluation — is orchestrated with **DVC** for reproducibility.

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.9 |
| Deep Learning | TensorFlow 2.20, Keras (VGG16) |
| Data | Kidney CT Scan Dataset (Google Drive via `gdown`) |
| MLOps | MLflow 2.2.2, DagsHub, DVC |
| Evaluation | Keras metrics (Loss, Accuracy) |
| Web Framework | Flask, Flask-CORS |
| Deployment | Docker, Hugging Face Spaces |
| Config | YAML (`config.yaml`, `params.yaml`) |
| Project Structure | Cookiecutter Data Science (CCDS) |

---

## Project Structure

```
kidney-classification/
├── src/
│   └── cnnClassifier/
│       ├── components/         # Data ingestion, base model, training, evaluation
│       ├── pipeline/           # Stage pipelines + prediction pipeline
│       ├── config/             # ConfigurationManager
│       ├── entity/             # Dataclass configs
│       ├── utils/              # common.py (read_yaml, decodeImage, save_json, etc.)
│       └── constants/          # File path constants
├── artifacts/                  # Auto-generated during pipeline runs
│   ├── data_ingestion/         # Raw + unzipped CT scan images
│   ├── prepare_base_model/     # base_model.h5, base_model_updated.h5
│   └── training/               # model.h5 (trained)
├── research/                   # Jupyter notebooks (experimentation)
│   ├── 01_data_ingestion.ipynb
│   ├── 02_prepare_base_model.ipynb
│   ├── 03_training_model.ipynb
│   └── 04_model_evaluation.ipynb
├── templates/                  # Flask HTML templates
├── config/
│   └── config.yaml             # Artifact paths & directory config
├── params.yaml                 # Model hyperparameters
├── dvc.yaml                    # DVC pipeline stage definitions
├── main.py                     # Runs all pipeline stages sequentially
├── app.py                      # Flask web application
├── Dockerfile                  # Docker image for HF Spaces deployment
├── requirements.txt            # Python dependencies
├── scores.json                 # Latest evaluation scores
└── setup.py                    # Package installer
```

---

## Pipeline Stages

The DVC pipeline has 4 stages, each tracked for reproducibility:

```
Data Ingestion → Prepare Base Model → Training → Evaluation
```

1. **Data Ingestion** — Downloads the kidney CT scan dataset as a ZIP from Google Drive using `gdown`, then extracts it into `artifacts/data_ingestion/`
2. **Prepare Base Model** — Loads VGG16 (ImageNet weights, no top), freezes all convolutional layers, appends a Flatten + Dense(2, softmax) head, and saves both the base model and updated model
3. **Training** — Loads the updated model, applies image augmentation (rotation, flip, shear, zoom), trains with an 80/20 train/validation split, saves the final `model.h5`
4. **Evaluation** — Evaluates on a 30% validation split, logs loss and accuracy to MLflow/DagsHub, saves scores to `scores.json`

---

## Model Architecture

**VGG16 + Custom Head** (Transfer Learning):

```
Input (224 × 224 × 3)
    → VGG16 Convolutional Base (frozen, ImageNet weights)
        → 5 blocks of Conv2D + MaxPooling
    → Flatten
    → Dense(2, activation='softmax')

Optimizer: SGD (base prep) / Adam (fine-tuning)
Loss:      CategoricalCrossentropy
Metric:    Accuracy
```

| Hyperparameter | Value |
|---|---|
| `IMAGE_SIZE` | [224, 224, 3] |
| `CLASSES` | 2 (Tumor / Normal) |
| `EPOCHS` | 5 |
| `BATCH_SIZE` | 16 |
| `LEARNING_RATE` | 0.01 |
| `WEIGHTS` | imagenet |
| `INCLUDE_TOP` | False |
| `AUGMENTATION` | True |
| Total Params | ~14.76M |
| Trainable Params | ~50K (head only) |

---

## Scores

Latest evaluation results (from `scores.json`):

| Metric | Score |
|---|---|
| Loss | 0.0 |
| Accuracy | **1.0 (100%)** |

---

## Setup & Installation

### Prerequisites

- Python 3.9
- `conda` or `venv`
- `git`

### 1. Clone the Repository

```bash
git clone https://github.com/Nilansh-garg/Kidney_Disease_classification_MLFlow-DVC.git
cd Kidney_Disease_classification_MLFlow-DVC
```

### 2. Create a Virtual Environment

**Using conda:**
```bash
conda create -n kidney-classification python=3.9 -y
conda activate kidney-classification
```

**Using venv:**
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 3. Install Dependencies

=======
"Disclaimer: This tool is for educational purposes only and is not intended for clinical diagnosis or medical advice."


# Kidney-Disease-Classification-MLflow-DVC


## Workflows

1. Update config.yaml
2. Update secrets.yaml [Optional]
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline 
8. Update the main.py
9. Update the dvc.yaml
10. app.py

# How to run?
### STEPS:

Clone the repository

```bash
https://github.com/krishnaik06/Kidney-Disease-Classification-Deep-Learning-Project
```
### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n cnncls python=3.8 -y
```

```bash
conda activate cnncls
```


### STEP 02- install the requirements
>>>>>>> 790b6fb7128b016c679ce9b519bd183fe05b91e2
```bash
pip install -r requirements.txt
```

<<<<<<< HEAD
This installs all dependencies including TensorFlow, Flask, MLflow, DVC, and also registers the local `cnnClassifier` package via the `-e .` entry in `requirements.txt`.

---

## Running the Application

### Option A — Run All Pipeline Stages at Once

```bash
python main.py
```

Sequentially runs all 4 stages: data ingestion → prepare base model → training → evaluation.

### Option B — Run with DVC (Recommended)

```bash
dvc repro
```

DVC only re-runs stages whose dependencies have changed, making iteration fast.

### Option C — Run Individual Stages

```bash
python src/cnnClassifier/pipeline/stage_01_data_ingestion.py
python src/cnnClassifier/pipeline/stage_02_prepare_model.py
python src/cnnClassifier/pipeline/stage_03_training_model.py
python src/cnnClassifier/pipeline/stage_04_model_evaluation.py
```

### Launch the Flask Web App

```bash
python app.py
```

The app will be available at `http://localhost:7860`.

---

## API Reference

### `POST /predict`

Accepts a base64-encoded CT scan image and returns the classification result.

**Request:**
```json
{
  "image": "<base64-encoded-image-string>"
}
```

**Response:**
```json
{ "result": "Normal" }
```
or
```json
{ "result": "Tumor" }
```

### `GET /train`

Triggers model retraining by running `dvc repro` on the server.

**Response:**
```json
{ "status": "success", "message": "Training successful" }
```

### `GET /health`

```json
{ "status": "healthy", "message": "API is running" }
```

**All Routes:**

| Route | Method | Description |
|---|---|---|
| `/` | GET | Home page |
| `/classify` | GET | Classification UI |
| `/about` | GET | About page |
| `/contact` | GET | Contact page |
| `/train` | GET/POST | Trigger pipeline retraining |
| `/predict` | POST | Classify a CT scan image |
| `/health` | GET | API health check |

---

## Docker Deployment

The app is containerized for deployment on **Hugging Face Spaces**.

```bash
# Build the image
docker build -t kidney-classification .

# Run locally
docker run -p 7860:7860 kidney-classification
```

The Dockerfile uses `python:3.9-slim-buster` with system packages `libgl1-mesa-glx` and `libglib2.0-0` for OpenCV/TensorFlow compatibility.

---

## MLflow & DagsHub Tracking

Metrics and parameters are logged to **DagsHub** via **MLflow**.

To initialize tracking:
```python
import dagshub
dagshub.init(repo_owner='Nilansh-garg', repo_name='kidney-classification', mlflow=True)
```

Metrics logged per run: `loss`, `accuracy`

Params logged: all values from `params.yaml`

View experiments: [https://dagshub.com/Nilansh-garg/kidney-classification.mlflow](https://dagshub.com/Nilansh-garg/kidney-classification.mlflow)

---

## Configuration

### `params.yaml` — Model Hyperparameters

```yaml
AUGMENTATION: True
BATCH_SIZE: 16
EPOCHS: 5
IMAGE_SIZE: [224, 224, 3]
LEARNING_RATE: 0.01
INCLUDE_TOP: False
WEIGHTS: imagenet
MODEL_NAME: vgg16
CLASSES: 2
```

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

*Built by [Nilansh Garg](mailto:nilanshgarg13@gmail.com)*
=======
```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up you local host and port
```






## MLflow

- [Documentation](https://mlflow.org/docs/latest/index.html)

- [MLflow tutorial](https://youtu.be/qdcHHrsXA48?si=bD5vDS60akNphkem)

##### cmd
- mlflow ui

### dagshub
[dagshub](https://dagshub.com/)

MLFLOW_TRACKING_URI=https://dagshub.com/entbappy/Kidney-Disease-Classification-MLflow-DVC.mlflow \
MLFLOW_TRACKING_USERNAME=entbappy \
MLFLOW_TRACKING_PASSWORD=6824692c47a369aa6f9eac5b10041d5c8edbcef0 \
python script.py

Run this to export as env variables:

```bash

export MLFLOW_TRACKING_URI=https://dagshub.com/entbappy/Kidney-Disease-Classification-MLflow-DVC.mlflow

export MLFLOW_TRACKING_USERNAME=entbappy 

export MLFLOW_TRACKING_PASSWORD=6824692c47a369aa6f9eac5b10041d5c8edbcef0

```


### DVC cmd

1. dvc init
2. dvc repro
3. dvc dag


## About MLflow & DVC

MLflow

 - Its Production Grade
 - Trace all of your expriements
 - Logging & taging your model


DVC 

 - Its very lite weight for POC only
 - lite weight expriements tracker
 - It can perform Orchestration (Creating Pipelines)



# AWS-CICD-Deployment-with-Github-Actions

## 1. Login to AWS console.

## 2. Create IAM user for deployment

	#with specific access

	1. EC2 access : It is virtual machine

	2. ECR: Elastic Container registry to save your docker image in aws


	#Description: About the deployment

	1. Build docker image of the source code

	2. Push your docker image to ECR

	3. Launch Your EC2 

	4. Pull Your image from ECR in EC2

	5. Lauch your docker image in EC2

	#Policy:

	1. AmazonEC2ContainerRegistryFullAccess

	2. AmazonEC2FullAccess

	
## 3. Create ECR repo to store/save docker image
    - Save the URI: 566373416292.dkr.ecr.us-east-1.amazonaws.com/chicken

	
## 4. Create EC2 machine (Ubuntu) 

## 5. Open EC2 and Install docker in EC2 Machine:
	
	
	#optinal

	sudo apt-get update -y

	sudo apt-get upgrade
	
	#required

	curl -fsSL https://get.docker.com -o get-docker.sh

	sudo sh get-docker.sh

	sudo usermod -aG docker ubuntu

	newgrp docker
	
# 6. Configure EC2 as self-hosted runner:
    setting>actions>runner>new self hosted runner> choose os> then run command one by one


# 7. Setup github secrets:

    AWS_ACCESS_KEY_ID=

    AWS_SECRET_ACCESS_KEY=

    AWS_REGION = us-east-1

    AWS_ECR_LOGIN_URI = demo>>  566373416292.dkr.ecr.ap-south-1.amazonaws.com

    ECR_REPOSITORY_NAME = simple-app

>>>>>>> 790b6fb7128b016c679ce9b519bd183fe05b91e2
