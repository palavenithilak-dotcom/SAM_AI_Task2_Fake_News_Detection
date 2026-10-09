import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide"
)

st.title("📰 Fake News Detection System")
st.write(
    "An NLP and Machine Learning based system that classifies "
    "news articles as **Real** or **Fake**."
)


# --------------------------------------------------
# DATASET PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

FAKE_FILE = DATA_DIR / "Fake.csv"
TRUE_FILE = DATA_DIR / "True.csv"


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

@st.cache_data
def load_dataset():

    fake_df = pd.read_csv(FAKE_FILE)
    true_df = pd.read_csv(TRUE_FILE)

    fake_df["label"] = "Fake"
    true_df["label"] = "Real"

    fake_df["text"] = fake_df["text"].fillna("")
    true_df["text"] = true_df["text"].fillna("")

    fake_df["title"] = fake_df["title"].fillna("")
    true_df["title"] = true_df["title"].fillna("")

    # Combine title + article text
    fake_df["content"] = fake_df["title"] + " " + fake_df["text"]
    true_df["content"] = true_df["title"] + " " + true_df["text"]

    # Keep only required columns
    fake_df = fake_df[["content", "label"]]
    true_df = true_df[["content", "label"]]

    # Combine both datasets
    df = pd.concat([fake_df, true_df], ignore_index=True)

    # Remove empty articles
    df = df[df["content"].str.strip() != ""]

    # Remove duplicate articles
    df = df.drop_duplicates(subset="content")

    # Shuffle dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return df


# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------

@st.cache_resource
def train_model(df):

    X = df["content"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                stop_words="english",
                max_features=50000,
                ngram_range=(1, 2),
                sublinear_tf=True
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    report = classification_report(
        y_test,
        predictions,
        output_dict=True
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=["Real", "Fake"]
    )

    return (
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        predictions,
        accuracy,
        report,
        matrix
    )


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

try:

    df = load_dataset()

except FileNotFoundError:

    st.error(
        "Dataset files were not found. "
        "Please make sure Fake.csv and True.csv are inside the data folder."
    )

    st.stop()


# --------------------------------------------------
# TRAIN
# --------------------------------------------------

with st.spinner("Training the NLP model on the dataset..."):

    (
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        predictions,
        accuracy,
        report,
        matrix
    ) = train_model(df)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("📊 Dataset Information")

st.sidebar.metric(
    "Total Articles",
    len(df)
)

st.sidebar.metric(
    "Training Articles",
    len(X_train)
)

st.sidebar.metric(
    "Testing Articles",
    len(X_test)
)

st.sidebar.metric(
    "Model Accuracy",
    f"{accuracy * 100:.2f}%"
)

st.sidebar.markdown("---")

st.sidebar.write("### Dataset Classes")

real_count = len(df[df["label"] == "Real"])
fake_count = len(df[df["label"] == "Fake"])

st.sidebar.write(f"🟢 Real News: **{real_count:,}**")
st.sidebar.write(f"🔴 Fake News: **{fake_count:,}**")


# --------------------------------------------------
# TOP METRICS
# --------------------------------------------------

st.subheader("📈 Model Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Accuracy",
        f"{accuracy * 100:.2f}%"
    )

with col2:
    st.metric(
        "Training Samples",
        f"{len(X_train):,}"
    )

with col3:
    st.metric(
        "Testing Samples",
        f"{len(X_test):,}"
    )

with col4:
    st.metric(
        "Features",
        "TF-IDF"
    )


# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------

st.subheader("🔍 Confusion Matrix")

fig, ax = plt.subplots()

ax.imshow(matrix)

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])

ax.set_xticklabels(["Real", "Fake"])
ax.set_yticklabels(["Real", "Fake"])

ax.set_xlabel("Predicted Label")
ax.set_ylabel("Actual Label")

ax.set_title("Fake News Classification Results")

for i in range(2):
    for j in range(2):
        ax.text(
            j,
            i,
            matrix[i, j],
            ha="center",
            va="center"
        )

st.pyplot(fig)


# --------------------------------------------------
# CLASSIFICATION REPORT
# --------------------------------------------------

with st.expander("📋 View Detailed Classification Report"):

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df.round(3),
        use_container_width=True
    )


# --------------------------------------------------
# NEWS PREDICTION
# --------------------------------------------------

st.subheader("🔎 Test Your Own News")

news_text = st.text_area(
    "Enter a news headline or article:",
    height=180,
    placeholder="Example: The government announced a new education policy..."
)


if st.button("🚀 Analyze News", use_container_width=True):

    if not news_text.strip():

        st.warning("Please enter some news text first.")

    else:

        prediction = model.predict([news_text])[0]

        probabilities = model.predict_proba([news_text])[0]

        classes = model.classes_

        probability_dict = dict(
            zip(classes, probabilities)
        )

        confidence = probability_dict[prediction]

        st.markdown("---")

        if prediction == "Real":

            st.success(
                f"🟢 Prediction: **REAL NEWS**"
            )

        else:

            st.error(
                f"🔴 Prediction: **FAKE NEWS**"
            )

        st.metric(
            "Model Confidence",
            f"{confidence * 100:.2f}%"
        )

        st.subheader("Prediction Probabilities")

        real_probability = probability_dict.get("Real", 0)
        fake_probability = probability_dict.get("Fake", 0)

        st.write(
            f"🟢 Real: **{real_probability * 100:.2f}%**"
        )

        st.progress(real_probability)

        st.write(
            f"🔴 Fake: **{fake_probability * 100:.2f}%**"
        )

        st.progress(fake_probability)


# --------------------------------------------------
# ABOUT
# --------------------------------------------------

st.markdown("---")

with st.expander("ℹ️ About This Project"):

    st.write(
        """
        This project uses Natural Language Processing (NLP)
        and Machine Learning to classify news articles.

        **Technologies used:**

        • Python  
        • Pandas  
        • Scikit-learn  
        • TF-IDF Vectorization  
        • Logistic Regression  
        • Streamlit  

        The model learns patterns from labelled Real and Fake
        news articles and predicts the class of new text.

        ⚠️ This system is an educational machine-learning
        classifier and should not be treated as a definitive
        fact-checking system.
        """
    )