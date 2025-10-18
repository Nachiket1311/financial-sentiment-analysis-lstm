# Financial News Sentiment Analysis using LSTM

**Author**: Nachiket Palekar  
**Email**: nachiket.palekar.nmims@gmail.com  
**Institution**: NMIMS University

A deep learning project that analyzes financial news headlines to predict sentiment and provide trading recommendations using Long Short-Term Memory (LSTM) neural networks.

## 🚀 Project Overview

This project implements a sentiment analysis system specifically designed for financial news headlines. It uses advanced NLP techniques and deep learning to:

- Classify news sentiment as **Positive**, **Neutral**, or **Negative**
- Provide trading recommendations: **Buy**, **Hold**, or **Sell**
- Extract named entities from financial news
- Achieve high accuracy in sentiment prediction

## 📊 Model Architecture

```python
model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=100),
    LSTM(128, return_sequences=True),
    Dropout(0.5),
    LSTM(64, return_sequences=False),
    Dense(3, activation='softmax')
])
```

## 🛠️ Technologies Used

- **Python 3.12**
- **TensorFlow/Keras** - Deep learning framework
- **NLTK & spaCy** - Natural language processing
- **Pandas & NumPy** - Data manipulation
- **Scikit-learn** - Model evaluation
- **Matplotlib & Seaborn** - Data visualization

## ⚡ Quick Start

### Prerequisites
```bash
pip install pandas numpy tensorflow scikit-learn matplotlib seaborn nltk spacy
python -m spacy download en_core_web_sm
```

### Usage
```python
# Load the model and predict
result = predict_action("Apple reports record quarterly earnings", tokenizer)
print(f"Sentiment: {result['sentiment']}")
print(f"Action: {result['action']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## 📈 Sample Results

**Example 1:**
```
Input: "Oil prices surge after OPEC+ announces production cuts"
Sentiment: Positive
Action: Buy
Confidence: 87.3%
```

**Example 2:**
```
Input: "Market crashes amid inflation concerns"
Sentiment: Negative
Action: Sell
Confidence: 91.2%
```

## 🎯 Key Features

- **Sentiment Classification**: 3-class sentiment analysis (Positive/Neutral/Negative)
- **Trading Recommendations**: Automated buy/hold/sell suggestions
- **Named Entity Recognition**: Extracts companies, people, dates, monetary values
- **Text Preprocessing**: Advanced NLP preprocessing pipeline
- **Model Evaluation**: Comprehensive metrics including F1-score, confusion matrix

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Training Accuracy | 85.2% |
| Validation Accuracy | 82.1% |
| Test Accuracy | 81.8% |
| F1-Score (Macro) | 0.79 |

## 🔧 Project Structure

```
NLP project/
│
├── Project.ipynb                 # Main Jupyter notebook
├── financial_news_events.csv     # Dataset
├── sentiment_lstm_model.h5       # Trained model
├── README.md                     # This file
└── requirements.txt              # Dependencies
```

## 🚀 Future Improvements

- [ ] Implement BERT for better contextual understanding
- [ ] Add real-time news feed integration
- [ ] Deploy as a web application
- [ ] Include stock price correlation analysis

## 👨‍💻 Author

**Nachiket Palekar**
- GitHub: [@Nachiket1311](https://github.com/Nachiket1311)
- Email: nachiket.palekar.nmims@gmail.com
- University: NMIMS University

## 📄 Academic Context

This project was developed as part of the NLP coursework in Semester 7 at NMIMS University. It demonstrates practical application of deep learning techniques in financial domain.

---

⭐ **Star this repository if you found it helpful!** ⭐