# 📱 Text SMS Categorizer

This project is focused on **classifying SMS messages** into categories such as **Spam** or **Not Spam (Ham)** using **Natural Language Processing (NLP)** and **Machine Learning techniques**.  
It demonstrates text preprocessing, feature extraction, model building, and evaluation to automate SMS classification.

---

## 📂 Project Structure
```
Text_SMS_CATEGORIZER.ipynb   # Main Jupyter Notebook
dataset/                      # Folder containing SMS dataset (if applicable)
README.md                      # Project documentation
```

---

## 🚀 Features
- Preprocessing of raw SMS text data.
- **Exploratory Data Analysis (EDA)** to understand dataset trends.
- Text vectorization using **TF-IDF** or **Bag of Words** techniques.
- Implementation of multiple **Machine Learning algorithms** for classification.
- Evaluation of models using metrics like accuracy, precision, recall, and F1-score.
- Predict whether a message is **Spam** or **Not Spam**.

---

## 🛠 Tech Stack
- **Programming Language:** Python 🐍
- **Libraries Used:**
  - `pandas` - Data manipulation and analysis
  - `numpy` - Numerical computations
  - `matplotlib` & `seaborn` - Data visualization
  - `scikit-learn` - Machine learning and evaluation
  - `nltk` - Natural Language Processing
  - `jupyter` - Notebook environment

---

## 📊 Workflow
1. **Import Libraries** – Load required Python libraries.
2. **Load Dataset** – Import the SMS dataset.
3. **Data Preprocessing**
   - Remove duplicates and null values.
   - Clean text (remove stopwords, punctuation, etc.).
   - Tokenization and lemmatization.
4. **Feature Extraction**
   - Convert text into numerical features using:
     - Bag of Words
     - TF-IDF Vectorizer
5. **Model Building**
   - Train multiple ML models such as:
     - Naive Bayes
     - Logistic Regression
     - Support Vector Machines (SVM)
     - Random Forest
6. **Model Evaluation**
   - Evaluate models using:
     - Accuracy
     - Precision
     - Recall
     - F1 Score
     - Confusion Matrix
7. **Prediction**
   - Predict if a new SMS is **Spam** or **Not Spam**.

---

## 📥 Installation
### **1. Clone the Repository**
```bash
git clone https://github.com/yourusername/text-sms-categorizer.git
cd text-sms-categorizer
```

### **2. Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate      # For Windows
```

### **3. Install Dependencies**
```bash
pip install -r requirements.txt
```

---

## 🧪 Usage
To run the project:
```bash
jupyter notebook
```
Then open `Text_SMS_CATEGORIZER.ipynb` and execute the cells step-by-step.

---

## 📈 Results
- The model accurately categorizes SMS messages as Spam or Not Spam.
- Performance metrics demonstrate the efficiency of different algorithms.

Example:
| **Message**                       | **Predicted Label** |
|----------------------------------|----------------------|
| "Congratulations! You won a prize!" | Spam 🚫 |
| "Hey, are we still meeting tomorrow?" | Not Spam ✅ |

---

## 📜 License
This project is licensed under the MIT License.  
Feel free to use and modify as needed.

---

## 👤 Author
- **Krishna Karbhari**
- GitHub: [kishu01karb](https://github.com/kishu01karb)

---

## 🌟 Acknowledgements
- [UCI SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- Inspiration from various spam detection research papers and NLP applications.
