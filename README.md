# SAM AI Task 2 - Fake News Detection

A Streamlit NLP + Machine Learning demo that classifies a news statement as Real or Fake.

## Features
- NLP preprocessing with TF-IDF
- Logistic Regression classifier
- Train/test split
- Accuracy and classification report
- Confusion matrix visualization
- Prediction confidence
- Interactive news-text prediction

## Dataset

This project uses the **Fake and Real News Dataset** from Kaggle.

**Source:** https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

### Setup

1. Download the dataset from Kaggle and extract the ZIP file.
2. Create a folder named `data` in the project directory.
3. Place `Fake.csv` and `True.csv` inside the `data` folder.
4. Install the required dependencies and run the application using the commands below.

**Note:** The dataset files are not included in this GitHub repository because of their large size.

## Run
```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

> Download the Fake and Real News Dataset from Kaggle and place Fake.csv and True.csv inside the data folder before running the application.
