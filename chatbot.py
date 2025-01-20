import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore")
nltk.download('punkt')
nltk.download('stopwords')

try:
    data = pd.DataFrame({
        "question": ["Hello", "How are you", "What is your name", "Goodbye"],
        "answer": ["Hi there!", "I'm good, thank you!", "I'm a chatbot!", "See you later!"]
    })
    print("Sample data loaded successfully!")
except Exception as e:
    print(f"An error occurred: {e}")
    exit()

def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation and not char.isdigit()])
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

data['question'] = data['question'].apply(preprocess_text)
data['answer'] = data['answer'].apply(preprocess_text)

X_train, X_test, y_train, y_test = train_test_split(data['question'], data['answer'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred)}")

def chatbot():
    print("Hello! I'm your mini-chatbot. Type 'exit' to stop.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break
        user_input_processed = preprocess_text(user_input)
        user_input_vec = vectorizer.transform([user_input_processed])
        response = model.predict(user_input_vec)
        print(f"Chatbot: {response[0]}")

chatbot()
