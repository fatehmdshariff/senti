
# 🗣️ Bilingual Sentiment Analyzer

A **Streamlit** web application for real-time sentiment analysis on **English** and **Arabic** texts using **Logistic Regression** and **Naive Bayes** models. The project showcases text preprocessing, model training, evaluation, and deployment in a simple and scalable way.

---

## 📁 Project Structure

```
SENTI/
├── Datasets/
│   └── Split_Datasets/               # Contains test sets used for evaluation
├── Models/
│   ├── eng_log_reg_model.pkl
│   ├── eng_n_b_model.pkl
│   ├── eng_tfidf_vectorizer.pkl
│   ├── logistic_model_arabic.pkl
│   ├── naive_bayes_model_arabic.pkl
│   └── tfidf_vectorizer_arabic.pkl
├── Scripts/
│   ├── Evaluation/
│   │   ├── arabic_eval.ipynb
│   │   └── english_eval.ipynb
│   └── Model_training/
│       ├── arabic_model.ipynb
│       └── english_model.ipynb
├── strmlt.py                         # Streamlit UI for live predictions and evaluation
├── Requirements.txt
├── LICENSE
└── .gitignore
```

---

## 🚀 Features

- Supports **both English and Arabic** sentiment analysis.
- Dual-model comparison: **Logistic Regression** and **Naive Bayes**.
- Real-time prediction and optional model evaluation on test data.
- Intuitive web UI using **Streamlit**.
- Integrated visualization for **confusion matrix** and classification metrics.

---

## 📦 Datasets Used

- **English Dataset**  
  👉 [Sentiment Analysis Dataset by abhi8923shriv](https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset)

- **Arabic Dataset**  
  👉 [Arabic 100K Reviews Dataset by abedkhooli](https://www.kaggle.com/datasets/abedkhooli/arabic-100k-reviews)

---

## ⚙️ Setup Instructions

1. **Clone the repo**
   ```bash
   git clone https://github.com/your-username/senti.git
   cd senti
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv_senti_2
   .\venv_senti_2\Scripts\activate  # On Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r Requirements.txt
   ```

4. **Run the Streamlit app**
   ```bash
   streamlit run strmlt.py
   ```

---

## 📊 Models Used

Each language has its own pipeline:
- **Vectorizer**: TF-IDF
- **Models**: Logistic Regression & Naive Bayes
- **Evaluation**: Accuracy, Precision, Recall, F1-score, and Confusion Matrix

---

## 📈 Results (approximate)

| Language | Model             | Accuracy |
|----------|------------------|----------|
| English  | Logistic Regression | ~77%     |
| English  | Naive Bayes         | ~77%     |
| Arabic   | Logistic Regression | ~85%     |
| Arabic   | Naive Bayes         | ~85%     |

---

## 🧠 Future Improvements

- Add support for **neutral sentiment** classification.
- Explore **transformer-based models** like BERT for improved performance.
- Integrate **language detection** for automatic language switching.
- Deploy using **Streamlit Cloud** or **Docker** for easier sharing.

---

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## ❤️ Acknowledgements

- Developed using **scikit-learn**, **Streamlit**, **matplotlib**, and **seaborn**.
- Datasets sourced from **Kaggle**.
