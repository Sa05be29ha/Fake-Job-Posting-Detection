import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


nltk.download('stopwords')

# Load the dataset
df = pd.read_csv("fake_job_postings.csv")

# Display basic information
print("Dataset loaded successfully!")
print("\nShape of Dataset:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nFirst 5 Rows:")
print(df.head())

print("\nFraudulent Job Distribution:")
print(df['fraudulent'].value_counts())

# Combine important text columns
df['text'] = (
    df['title'].fillna('') + ' ' +
    df['company_profile'].fillna('') + ' ' +
    df['description'].fillna('') + ' ' +
    df['requirements'].fillna('')
)

print("\nCombined Text Created Successfully!")

print("\nSample Combined Text:")
print(df['text'].iloc[0][:500])   # Display first 500 characters

# Define stopwords
stop_words = set(stopwords.words('english'))

# Text preprocessing function
def preprocess(text):
    text = text.lower()

    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z ]', '', text)

    # Split into words
    words = text.split()

    # Remove stopwords
    words = [word for word in words if word not in stop_words]

    return " ".join(words)


# Apply preprocessing
df['clean_text'] = df['text'].apply(preprocess)

print("\nText Preprocessing Completed!")

print("\nSample Cleaned Text:")
print(df['clean_text'].iloc[0][:500])

# Features and Target
X = df['clean_text']
y = df['fraudulent']

print("\nFeatures and Target Created!")
print("Number of samples:", len(X))

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain-Test Split Completed!")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# Convert text into numerical features using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("\nTF-IDF Vectorization Completed!")

print("Training Data Shape:", X_train_vec.shape)
print("Testing Data Shape:", X_test_vec.shape)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000)

model.fit(X_train_vec, y_train)

print("\nModel Training Completed!")
# Predict on test data
predictions = model.predict(X_test_vec)

print("\nPredictions Generated!")
# Evaluate the model
accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:")
print(f"{accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))