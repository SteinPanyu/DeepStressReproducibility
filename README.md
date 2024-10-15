# Stress Prediction Pipeline Using DeepStress Dataset

This repository provides resources for independent reproducibility experiments focused on stress detection using the DeepStress dataset. It includes a complete machine learning pipeline from data exploration to model evaluation and hyperparameter tuning. The environment is configured using Conda, with dependencies outlined in `environment.yaml`.

This project replicates the results of the original experiments conducted on the K-EmoPhone dataset. For more details on the K-EmoPhone dataset, please refer to the original [IndependentReproducibility](https://github.com/SteinPanyu/IndependentReproducibility) GitHub repository.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline Stages](#pipeline-stages)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

This project leverages the DeepStress dataset to develop and optimize models for detecting stress. The pipeline includes stages for data preprocessing, feature extraction, model training, and evaluation, enabling reproducibility and facilitating research in stress detection.

## Dataset

- **DeepStress Dataset**: The DeepStress dataset used in this repository can be found at the [following link](https://github.com/Kaist-ICLab/DeepStress_Dataset/tree/main).

## Project Structure

```bash
.
├── Funcs/
│   ├── Utility.py                           # Utility functions for data processing
├── AppCategorization.ipynb                  # Categorzing the apps
├── Feature_Extraction.ipynb                 # Feature Extraction notebook
├── HyperParameterTuning.ipynb               # Hyperparameter tuning notebook (this is only to test model complexity related hyperparameters)
├── Model Evaluation & Deeper Analysis.ipynb # Evaluation and deeper analysis                        
├── Model Training.ipynb                     # Model training procedures
├── Preprocessing.ipynb                      # Data preprocessing steps
├── Preprocessing_raw.ipynb                  # Preprocess the raw data so that it is similar to the public K-EmoPhone dataset
│                                
└── environment.yaml                         # Conda environment configuration
```

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/SteinPanyu/IndependentReproducibility.git
   cd Stress_Detection_D-1
   ```

2. Set up the Conda environment:

   ```bash
   conda env create -f environment.yaml
   conda activate sci-data
   ```

## Usage

The project is organized into Jupyter notebooks, each representing a stage of the machine learning pipeline. Follow the sequence below to reproduce the entire process:

## Pipeline Stages

1. **App Catergorization**
   - Notebook: `AppCategorization.ipynb`
   - Purpose: Categorzie the apps first like in the EDA part for K-EmoPhone dataset.
   - Includes: Data crawling for app categories based on app package name and also manually search for undefined categories.
  
2. **Preprocessing Raw Data**
   - Notebook: `Preprocessing_raw.ipynb`
   - Purpose: Since the DeepStress dataset is quite different from the K-EmoPhone dataset, in order to replicate the experiments, we did some additional preprocessing
   - Includes: Remove irrelevant columns in the dataframe, remove duplicate records, and check for data quality

3. **Preprocessing**
   - Notebook: `Preprocessing.ipynb`
   - Purpose: Clean and transform the data for subsequent stages.
   - Includes: Handling missing data, normalization, and encoding.

4. **Feature Extraction**
   - Notebook: `Feature_Extraction.ipynb`
   - Purpose: Extract relevant features from the dataset.
   - Includes: Extraction of pitch, intensity, and text-based markers.

5. **Model Training**
   - Notebook: `Model Training.ipynb`
   - Purpose: Train machine learning models to classify stress levels.
   - Includes: Training procedures, model selection, and validation.

6. **Model Evaluation & Deeper Analysis**
   - Notebook: `Model Evaluation & Deeper Analysis.ipynb`
   - Purpose: Evaluate the model's performance with detailed analysis.
   - Includes: Metrics, interpretability, and areas for improvement.

7. **Hyperparameter Tuning**
   - Notebook: `HyperParameterTuning.ipynb`
   - Purpose: Test if model complexity related parameters can affect the overfitting issue. And see how much improvement could be via handling these parameters.
   - Includes: Hyperopt for model complexity parameters




## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- DeepStress Dataset for providing the data used in this project.
- scikit-learn and pandas libraries for data processing and model building.
