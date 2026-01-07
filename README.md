# MLOps Project - Vehicle Insurance Data Pipeline

End-to-end production-style MLOps pipeline for vehicle insurance prediction, built to demonstrate real-world machine learning system design, pipeline orchestration, data reliability, and deployment readiness.

---

## Project Overview

This project focuses on the **importance of MLOps** rather than just model building.  
It demonstrates how machine learning workflows are structured, validated, versioned, and served in a production-like environment.

The system predicts whether a customer is likely to opt for vehicle insurance based on demographic and policy-related features.

---
<p align="center">
  <img src="banner.png" alt="Local Deployment" width="700">
</p>


## Why This Is an MLOps Project

This is not a notebook-only ML project.  
It showcases how real ML systems are built with:

- Modular pipeline architecture
- Data ingestion from external sources
- Schema-based data validation
- Feature engineering and preprocessing pipelines
- Model training with evaluation
- Versioned artifacts
- Centralized logging and exception handling
- API-based prediction service
- Dockerized local deployment

The emphasis is on **reproducibility, reliability, and maintainability**.

---

## End-to-End Workflow

MongoDB  
→ Data Ingestion  
→ Data Validation (Schema-based)  
→ Data Transformation & Feature Engineering  
→ Model Training & Evaluation  
→ Prediction Pipeline (FastAPI)  
→ Dockerized Local Deployment  

---

## Data Source

- MongoDB Atlas is used as the data source
- Data is fetched dynamically using environment variables
- No hardcoded credentials are used

Environment variable setup (example):

PowerShell:



---

## Pipeline Components

### Data Ingestion
- Fetches data from MongoDB
- Performs train-test split
- Stores datasets as versioned artifacts

- ![Data Ingestion](data_ingestion.png)

### Data Validation
- Schema validation using a predefined schema file
- Checks required columns and data types
- Generates validation reports as artifacts

- ![Data Validation](data_validation.png)

### Data Transformation
- Feature engineering
- Encoding categorical variables
- Scaling numerical features
- Handling class imbalance using SMOTEENN
- Saves preprocessing pipeline for reuse

- ![Data Transformation](data_transformation.png)

### Model Training
- Model trained using RandomForestClassifier
- Evaluated using precision, recall, and F1-score
- Best-performing model saved as a versioned artifact


![Model Training](model_training.png)


---

## Prediction Service

- Built using FastAPI
- HTML-based form for user input
- Accepts customer details
- Returns prediction result:
  - Response-Yes
  - Response-No
 
  - 

The prediction pipeline reuses the trained model and preprocessing object to ensure consistency.

---

<p align="center">
  <img src="banner1.png" alt="Local Deployment" width="400">
</p>



---

## Docker Usage (Local Only)

Docker is used **only for local containerized execution** to demonstrate deployment readiness.


