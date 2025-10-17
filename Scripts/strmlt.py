import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

MODEL1_PATH = ROOT_DIR / "Models" / "eng_log_reg_model.pkl"
MODEL2_PATH = ROOT_DIR / "Models" / "eng_n_b_model.pkl"
MODEL3_PATH = ROOT_DIR / "Models" / "eng_tfidf_vectorizer.pkl"
MODEL4_PATH = ROOT_DIR / "Models" / "logistic_model_arabic.pkl"
MODEL5_PATH = ROOT_DIR / "Models" / "naive_bayes_model_arabic.pkl"
MODEL6_PATH = ROOT_DIR / "Models" / "tfidf_vectorizer_arabic.pkl"

# Load models and vectorizers
logreg_en = joblib.load(MODEL1_PATH)
nb_en = joblib.load(MODEL2_PATH)
vectorizer_en = joblib.load(MODEL3_PATH)

logreg_ar = joblib.load(MODEL4_PATH)
nb_ar = joblib.load(MODEL5_PATH)
vectorizer_ar = joblib.load(MODEL6_PATH)

# Streamlit UI setup
st.set_page_config(page_title="Bilingual Sentiment Analyzer", page_icon="🗣️", layout="wide")

st.title("🗣️ Bilingual Sentiment Analyzer")
st.markdown("Supports both **English** and **Arabic** text with *Logistic Regression* and *Naive Bayes* comparison")

language = st.selectbox("🌍 Choose Language", ["English", "Arabic"])
user_input = st.text_area("✏️ Enter your text here:", height=150)

show_eval = st.checkbox("📊 Show Model Evaluation", value=False)

if st.button("🔍 Analyze Sentiment"):
    if not user_input.strip():
        st.warning("🚨 Please enter some text to analyze.")
    else:
        if language == "English":
            vect = vectorizer_en.transform([user_input])
            pred_logreg = logreg_en.predict(vect)[0]
            pred_nb = nb_en.predict(vect)[0]
        else:
            vect = vectorizer_ar.transform([user_input])
            pred_logreg = logreg_ar.predict(vect)[0]
            pred_nb = nb_ar.predict(vect)[0]

        label_map = {"Positive": "😊 Positive", "Negative": "😠 Negative", 4: "😊 Positive", 0: "😠 Negative"}

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🤖 Logistic Regression Prediction")
            st.success(f"**{label_map.get(pred_logreg, pred_logreg)}**")

        with col2:
            st.markdown("#### 📈 Naive Bayes Prediction")
            st.info(f"**{label_map.get(pred_nb, pred_nb)}**")

        if show_eval:
            st.markdown("---")
            st.markdown("## 🔍 Model Evaluation")

            if language == "English":
                X_test, y_test = joblib.load(r"D:\Projects_25\Sentiment_analysis\senti\Datasets\Split_Datasets\english_test.pkl")
                X_test_vect = vectorizer_en.transform(X_test)
                y_pred_logreg = logreg_en.predict(X_test_vect)
                y_pred_nb = nb_en.predict(X_test_vect)
            else:
                X_test, y_test = joblib.load(r"D:\Projects_25\Sentiment_analysis\senti\Datasets\Split_Datasets\arabic_test.pkl")
                X_test_vect = vectorizer_ar.transform(X_test)
                y_pred_logreg = logreg_ar.predict(X_test_vect)
                y_pred_nb = nb_ar.predict(X_test_vect)

            col3, col4 = st.columns(2)

            with col3:
                st.subheader("Logistic Regression Evaluation")
                st.text(classification_report(y_test, y_pred_logreg))
                st.text(f"Accuracy (Logistic Regression): {accuracy_score(y_test, y_pred_logreg) * 100:.2f}%")
                cm_lr = confusion_matrix(y_test, y_pred_logreg)
                fig_lr, ax1 = plt.subplots()
                sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=ax1)
                ax1.set_title("Confusion Matrix - Logistic Regression")
                st.pyplot(fig_lr)

            with col4:
                st.subheader("Naive Bayes Evaluation")
                st.text(classification_report(y_test, y_pred_nb))
                st.text(f"Accuracy (Naive Bayes): {accuracy_score(y_test, y_pred_nb) * 100:.2f}%")
                cm_nb = confusion_matrix(y_test, y_pred_nb)
                fig_nb, ax2 = plt.subplots()
                sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Greens', ax=ax2)
                ax2.set_title("Confusion Matrix - Naive Bayes")
                st.pyplot(fig_nb)

st.markdown("---")
st.markdown("<center><sub>Made with ❤️ using Streamlit</sub></center>", unsafe_allow_html=True)
