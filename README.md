# 🧠 Emotion Detection from Text

A Natural Language Processing system for classifying text into six emotion categories using **TF-IDF**, classical machine learning models, and a fine-tuned **DistilBERT Transformer**.

---

# 📌 Overview

**Emotion Detection from Text** is an NLP classification project that explores the progression from traditional machine learning approaches to modern Transformer-based architectures.

The project evaluates **Multinomial Naive Bayes, Logistic Regression, Linear SVM, and DistilBERT** on the same emotion classification task.

After experimentation, the best-performing system was **DistilBERT**, achieving **93% test accuracy and 0.90 macro F1-score** across all six emotion classes.

The project includes exploratory data analysis, text preprocessing, TF-IDF feature extraction, classical ML baselines, Transformer fine-tuning, model comparison, and final evaluation.

---

# ✨ Features

- Exploratory Data Analysis
- Text Cleaning and Preprocessing
- Class Distribution Analysis
- Emotion-Specific Word Frequency Analysis
- TF-IDF Feature Extraction
- Unigram and Bigram Features
- Multinomial Naive Bayes
- Logistic Regression
- Linear Support Vector Machine
- DistilBERT Fine-Tuning
- Stratified Train/Validation/Test Splitting
- Dynamic Token Padding
- Class Imbalance Analysis
- Precision, Recall, and F1 Evaluation
- Macro F1 and Weighted F1
- Classification Reports
- Model Comparison

---

# 📂 Dataset

The project uses the **Emotion Detection Text Dataset**.

Each record follows the format:

```text
text;emotion
```

The dataset contains **16,000 text samples** belonging to six emotion categories.

| Emotion | Samples |
|---------|--------:|
| joy | 5,362 |
| sadness | 4,666 |
| anger | 2,159 |
| fear | 1,937 |
| love | 1,304 |
| surprise | 572 |

### Dataset Summary

| Property | Value |
|----------|------:|
| Total samples | 16,000 |
| Duplicate rows | 1 |
| Missing values | 0 |
| Unique samples after cleaning | 15,999 |
| Number of classes | 6 |

---

# 🔬 Exploratory Data Analysis

The dataset was analyzed before model training to understand its structure, text characteristics, and class imbalance.

### Text Statistics

| Statistic | Value |
|-----------|------:|
| Mean length | 19.17 words |
| Median length | 17 words |
| Minimum length | 2 words |
| Maximum length | 66 words |

### Most Frequent Words

```text
feel
feeling
like
im
really
know
time
get
little
people
want
think
```

Emotion-specific vocabulary was also analyzed.

**Anger**

```text
angry, irritable, offended, resentful, selfish, bothered
```

**Fear**

```text
strange, nervous, terrified, anxious, afraid, scared, uncertain
```

**Love**

```text
love, sweet, loving, passionate, caring, sympathetic, lovely
```

**Surprise**

```text
amazed, impressed, overwhelmed, surprised, curious, shocked
```

**Joy**

```text
good, happy, well, pretty, love
```

**Sadness**

```text
alone, dont, could, back, today, always
```

---

# 🧹 Data Preprocessing

The preprocessing pipeline included:

- Removed duplicate rows
- Removed missing values
- Removed empty text samples
- Converted text to lowercase
- Removed unnecessary punctuation
- Reset dataframe indices

Stopword removal, stemming, and aggressive lemmatization were avoided because negation words such as `not`, `dont`, and `cant` can carry important emotional information.

For example:

```text
"I am happy"
```

and:

```text
"I am not happy"
```

have very different meanings.

---

# ✂️ Train / Validation / Test Split

For the Transformer experiment:

```text
80% → Training
10% → Validation
10% → Testing
```

Stratified splitting was used to preserve emotion proportions.

```python
train_test_split(
    ...,
    stratify=y,
    random_state=42
)
```

The test set remained untouched until final evaluation.

---

# 🧮 Machine Learning Pipeline

The classical NLP pipeline follows:

```text
                    Raw Text
                       │
                       ▼
                Text Preprocessing
                       │
                       ▼
                    TF-IDF
                 (1-2 grams)
                       │
          ┌────────────┼────────────┐
          │            │            │
          ▼            ▼            ▼
    Naive Bayes   Logistic Reg.    Linear SVM
          │            │            │
          └────────────┼────────────┘
                       │
                       ▼
                Emotion Prediction
```

---

# 🔢 TF-IDF Feature Extraction

Text was converted into numerical representations using **TF-IDF (Term Frequency-Inverse Document Frequency)**.

Configuration:

```python
TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2)
)
```

Both unigrams and bigrams were used.

### Unigrams

```text
happy
sad
angry
scared
```

### Bigrams

```text
very happy
feel sad
really angry
```

---

# 🧠 Classical Models

## Model 1 — Multinomial Naive Bayes

### Performance

```text
Accuracy:     66.66%
Macro F1:      0.43
Weighted F1:   0.60
```

Naive Bayes struggled particularly with minority classes.

---

## Model 2 — Logistic Regression

```python
LogisticRegression(
    max_iter=1000
)
```

### Performance

```text
Accuracy:     82.22%
Macro F1:      0.74
Weighted F1:   0.81
```

---

## Model 3 — Linear SVM

```python
LinearSVC()
```

### Performance

```text
Accuracy:     88.13%
Macro F1:      0.84
Weighted F1:   0.88
```

### Classification Performance

| Emotion | Precision | Recall | F1 |
|---------|----------:|-------:|---:|
| anger | 0.89 | 0.84 | 0.86 |
| fear | 0.86 | 0.85 | 0.85 |
| joy | 0.88 | 0.93 | 0.90 |
| love | 0.81 | 0.77 | 0.79 |
| sadness | 0.92 | 0.91 | 0.91 |
| surprise | 0.81 | 0.75 | 0.78 |

Linear SVM became the strongest classical model.

---

# 🤖 Transformer Model — DistilBERT

After establishing classical baselines, a pretrained **DistilBERT** model was fine-tuned for the six-class emotion classification task.

Model:

```text
distilbert-base-uncased
```

DistilBERT was selected because it provides a strong balance between model size, computational efficiency, and classification performance.

---

# 🏗️ DistilBERT Pipeline

```text
                         Input Text
                             │
                             ▼
                    DistilBERT Tokenizer
                             │
                             ▼
                   Token IDs + Attention Mask
                             │
                             ▼
                       DistilBERT
                             │
                             ▼
                    Classification Head
                             │
                             ▼
                    6 Emotion Classes
```

---

# ⚙️ Fine-Tuning Configuration

| Parameter | Value |
|-----------|-------|
| Model | DistilBERT |
| Epochs | 3 |
| Learning Rate | 2e-5 |
| Training Batch Size | 16 |
| Evaluation Batch Size | 16 |
| Weight Decay | 0.01 |
| Maximum Sequence Length | 128 |
| Best Model Metric | Macro F1 |
| GPU | NVIDIA T4 |

Dynamic padding was implemented using:

```python
DataCollatorWithPadding
```

---

# 📈 Validation Results

| Epoch | Validation Loss | Accuracy | Macro F1 | Weighted F1 |
|------:|----------------:|---------:|---------:|------------:|
| 1 | 0.3766 | 92.56% | 0.8849 | 0.9252 |
| 2 | 0.3706 | 92.50% | 0.8824 | 0.9236 |
| 3 | 0.3733 | **92.75%** | **0.8894** | **0.9280** |

---

# 🏆 Final Test Results

The final DistilBERT model was evaluated on the completely unseen test set.

```text
Test Accuracy: 93.00%
Macro F1:      0.90
Weighted F1:   0.93
```

Approximately:

```text
1488 / 1600
```

test samples were classified correctly.

### Final Classification Report

| Emotion | Precision | Recall | F1-Score | Support |
|---------|----------:|-------:|---------:|--------:|
| anger | 0.93 | 0.93 | 0.93 | 216 |
| fear | 0.90 | 0.90 | 0.90 | 194 |
| joy | 0.95 | 0.93 | 0.94 | 536 |
| love | 0.80 | 0.85 | 0.83 | 131 |
| sadness | 0.97 | 0.97 | 0.97 | 466 |
| surprise | 0.82 | 0.86 | 0.84 | 57 |
| **Overall** | **0.93** | **0.93** | **0.93** | **1600** |

---

# 📊 Model Comparison

| Model | Accuracy | Macro F1 | Weighted F1 |
|-------|---------:|---------:|------------:|
| Multinomial Naive Bayes | 66.66% | 0.43 | 0.60 |
| Logistic Regression | 82.22% | 0.74 | 0.81 |
| Linear SVM | 88.13% | 0.84 | 0.88 |
| **DistilBERT** | **93.00%** | **0.90** | **0.93** |

### Best Classical Model vs Transformer

```text
Linear SVM
Accuracy: 88.13%
Macro F1: 0.84

          ↓

DistilBERT
Accuracy: 93.00%
Macro F1: 0.90
```

DistilBERT improved accuracy by **4.87 percentage points** over Linear SVM.

---

# 🧠 Key Findings

- TF-IDF + Linear SVM provided a strong **88.13% accuracy** baseline.
- Naive Bayes struggled with minority classes.
- The dataset is significantly imbalanced, making Macro F1 important.
- DistilBERT achieved the strongest overall performance.
- Contextual Transformer representations improved performance over TF-IDF features.
- DistilBERT achieved **93.00% test accuracy and 0.90 macro F1**.

---

# 📌 Limitations

- The dataset is class-imbalanced.
- Some sentences can express multiple emotions.
- Emotion classification can be subjective.
- A single sentence may not provide enough context.
- Performance may not generalize to social media, conversations, or other domains.
- The model predicts dataset-defined emotion categories and should not be interpreted as a psychological assessment.

---

# 🔮 Future Improvements

- Fine-tune BERT or RoBERTa
- Experiment with DeBERTa
- Hyperparameter optimization
- Class-weighted loss
- Data augmentation
- Confusion matrix-based error analysis
- Misclassification analysis
- Confidence calibration
- FastAPI deployment
- Streamlit/Gradio interface
- Real-time emotion prediction

---

# 📁 Project Structure

```text
emotion-detection/
│
├── data/
│   └── emotions.txt
│
├── notebooks/
│   └── emotion_detection.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── train_classical.py
│   ├── train_distilbert.py
│   └── predict.py
│
├── models/
│   └── README.md
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# ⚙️ Installation

```bash
git clone https://github.com/yourusername/emotion-detection.git
cd emotion-detection
```

Create a virtual environment:

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 📦 Requirements

```text
pandas
numpy
scikit-learn
matplotlib
seaborn
torch
transformers
datasets
accelerate
```

---

# 🧪 Evaluation Metrics

### Accuracy

Percentage of predictions that are correct.

### Precision

Measures how many predictions for a particular class were actually correct.

### Recall

Measures how many actual samples of a class were correctly identified.

### F1-Score

The harmonic mean of precision and recall.

### Macro F1

Calculates F1 independently for each class and gives every class equal importance.

### Weighted F1

Calculates F1 while weighting each class according to its number of samples.

---

# 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- PyTorch
- Hugging Face Transformers
- DistilBERT
- Matplotlib
- Seaborn
- Kaggle
- NVIDIA T4 GPU

---

# 📚 Concepts Demonstrated

- Exploratory Data Analysis
- Text preprocessing
- Feature engineering
- TF-IDF
- N-grams
- Stratified splitting
- Naive Bayes
- Logistic Regression
- Support Vector Machines
- Precision and Recall
- F1-score
- Macro F1
- Weighted F1
- Class imbalance
- Tokenization
- Transformers
- Transfer learning
- Fine-tuning
- DistilBERT
- Model evaluation

---

# 🏁 Conclusion

This project demonstrates the progression of NLP emotion classification from traditional machine learning to modern Transformer architectures.

The classical **TF-IDF + Linear SVM** approach achieved:

```text
88.13% accuracy
```

Fine-tuning **DistilBERT** improved performance to:

```text
93.00% test accuracy
0.90 macro F1
0.93 weighted F1
```

The results demonstrate that while traditional NLP approaches remain competitive, pretrained Transformer models can provide stronger performance by leveraging contextual representations of language.

---

# 👨‍💻 Author

**Darshan**

AI/ML Student | Computer Vision & NLP Enthusiast

Interested in:

- Artificial Intelligence
- Machine Learning
- Computer Vision
- NLP
- LLMs
- RAG Systems
- Deep Learning

---

## 📜 License

This project is intended for educational and research purposes.
