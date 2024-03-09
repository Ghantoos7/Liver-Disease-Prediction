# Liver Disease Prediction Project

## Overview
This project aims to analyze and predict liver disease using a dataset from OpenML. We focus on preprocessing the data, exploring its characteristics, and selecting relevant features for model development.

## Data
The dataset (ID: 1480) contains biochemical and demographic parameters potentially indicative of liver disease.

## Repository Structure
- `exploration.ipynb`: Jupyter notebook containing the exploratory data analysis and preprocessing steps.
- `src/`:
  - `load_data.py`: Module to download the dataset from OpenML.
  - `preprocess.py`: Module containing functions for data cleaning and feature selection.
- `data/`:
  - `liver_disease_preprocessed.csv`: The cleaned and preprocessed dataset ready for modeling.

## Setup
To run this project, you will need Python 3.x and the following libraries:

Install the necessary packages using pip:
```bash
pip install -r requirements.txt