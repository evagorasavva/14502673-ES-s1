# Academic Dropout Prediction Analysis

**Coventry University | 6006CEM: Machine Learning and Related Applications**

## Overview

This project focuses on predicting student outcomes in higher education programs to address the critical issue of student dropout and failure rates. By leveraging machine learning techniques, the goal is to identify at-risk students early in their enrollment, enabling timely interventions.

### Key Objectives

- **Multinomial Classification Task:**  
  Predict the final status of a student (Dropout, Enrolled but not graduated on time, Graduate) based on data available at the time of enrollment:
  - **Dropout:** Student leaves the program before completion.
  - **Enrolled:** Student remains enrolled but does not complete within the expected duration.
  - **Graduate:** Student completes the program within the expected duration.

- **Feature Representation:**  
  Each instance represents a single student. Features include academic, demographic, and socio-economic attributes recorded at enrollment.

- **Outcome Variable:**  
  The target variable is the final status (Dropout/Enrolled/Graduate) at the end of the normal course duration.

## About the Dataset

The dataset originates from the UCI Machine Learning Repository and is aimed at reducing academic dropout rates. It includes student-level data—demographic, socio-economic, and academic history—collected at enrollment from the Polytechnic Institute of Portalegre, Portugal.

- **Data Scope:**  
  Undergraduate students from various disciplines (e.g., agronomy, design, education, nursing, journalism, management, social services, technology) spanning academic years 2008/09 to 2018/19.

- **Source & Funding:**  
  - [UCI ML Repository: Predict students dropout and academic success Dataset](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)  
  - Curated as part of a case study (Martins, 2021) and supported by SATDAP - Capacitação da Administração Pública, Portugal (POCI-05-5762-FSE-000191, Markelle Kelly, 2021).

## Project Structure

The main script is `academic_dropout_analysis.py`, which implements a machine learning pipeline for data preprocessing, model training, and evaluation.

### Steps Involved

1. **Data Loading & Preprocessing:**  
   - Load data from the specified file path.
   - Handle missing values, apply feature encoding, and normalization as needed.

2. **Model Training & Evaluation:**  
   - Split data into training and test sets (with a fixed random state for reproducibility).
   - Train a classification model (e.g., Logistic Regression, Random Forest, Gradient Boosting).
   - Evaluate the model’s performance (accuracy, F1-score, confusion matrix).

3. **Analysis of Results:**  
   - Interpret the performance metrics to understand the model’s effectiveness.
   - Discuss potential intervention strategies based on the predictions.

## Command-Line Arguments

The pipeline supports command-line parameters for customization:

- `--data_path` **(Required)**  
  Path to the input data file.  
  *Example:*  
  ```bash
  --data_path data.csv

- `--delimiter` **(optional)**  
  Delimiter used in the data file. Default is ;. We should use "\t" for this dataset.
  *Example:*  
  ```bash
  --delimiter "\t"

## Example Usage
  ```bash
  python academic_dropout_analysis.py --data_path data.csv --delimiter "\t" 
```


## Alternative Execution Method

Alternatively, you can run all the cells of the `academic_dropout_analysis_pipeline.ipynb` Jupyter Notebook to execute the analysis in an interactive environment.

### Steps to Run the Notebook

1. **Install Jupyter Notebook** (if not already installed):

    ```bash
    pip install notebook
    ```

2. **Navigate to the Project Directory**:

    ```bash
    cd path/to/your/project
    ```

3. **Launch Jupyter Notebook**:

    ```bash
    jupyter notebook academic_dropout_analysis_pipeline.ipynb
    ```

4. **Execute All Cells**:
    - Once the notebook is open in your browser, you can run all cells by selecting `Kernel` > `Restart & Run All` from the menu bar.

This method provides an interactive way to explore the data, visualize results, and understand the workflow step-by-step.

---





