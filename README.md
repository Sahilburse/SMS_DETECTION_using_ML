# SMS Spam Detection  

This project is an implementation of an **SMS Spam Classifier** using Natural Language Processing (NLP) and Machine Learning. The model can classify incoming text messages as **Spam** or **Ham (Not Spam)**.  

## 📌 Project Overview  
- Preprocess SMS text messages (lowercasing, removing stopwords, punctuation, stemming/lemmatization).  
- Extract features using **Bag of Words (BoW)** or **TF-IDF Vectorizer**.  
- Train machine learning models such as **Naive Bayes, Logistic Regression, SVM, Random Forest, XGBoost**.  
- Evaluate model performance with accuracy, precision, recall, and F1-score.  

## 📂 Repository Structure  
```
├── sms-spam-detection.ipynb   # Jupyter Notebook with code & experiments
├── README.md                  # Project documentation
├── requirements.txt           # Dependencies
└── data/                      # (Optional) Dataset storage folder
```

## ⚙️ Installation  

Clone the repository:  
```bash
git clone https://github.com/your-username/sms-spam-detection.git
cd sms-spam-detection
```

Create a virtual environment (recommended):  
```bash
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
```

Install dependencies:  
```bash
pip install -r requirements.txt
```

## 📊 Dataset  
The project uses the **SMS Spam Collection Dataset** from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection).  
It contains 5,574 messages labeled as `spam` or `ham`.  

Place the dataset (e.g., `spam.csv`) inside the `data/` folder.  

## 🚀 Usage  

Run the Jupyter Notebook:  
```bash
jupyter notebook sms-spam-detection.ipynb
```

Or, if you package the model later as a Python script:  
```bash
python predict.py "Free entry in 2 a wkly comp to win FA Cup..."
```

## 📈 Results  
- Achieved high accuracy (>95%) with Naive Bayes and Logistic Regression.  
- TF-IDF features worked slightly better than simple Bag of Words.  

## 🔮 Future Improvements  
- Deploy the model as a **Flask/FastAPI web app**.  
- Build a simple **Streamlit/Gradio UI**.  
- Experiment with **Deep Learning (LSTM, BERT)**.  

## 🛠️ Tech Stack  
- **Python 3.x**  
- **NLTK** for text preprocessing  
- **scikit-learn, XGBoost** for ML models  
- **pandas, numpy, matplotlib, seaborn, wordcloud** for data handling & visualization  

## 👨‍💻 Author  
Your Name – [your-username](https://github.com/your-username)  
