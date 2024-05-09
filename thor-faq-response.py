import os
import pickle
import nltk
import pandas as pd
import numpy as np
from nltk.stem.lancaster import LancasterStemmer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import LabelEncoder as LE
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
from flask import Flask, jsonify, request
from flask_cors import CORS


app = Flask(__name__)
cors = CORS(app, resources={r"/faq": {"origins": "*"}})

class TfidfVectorGenerator:
    def __init__(self, size=100):
        self.vec_size = size
        self.vectorizer = None

    def vectorize(self, clean_questions):
        self.vectorizer = TfidfVectorizer(min_df=1, stop_words='english')
        self.vectorizer.fit(clean_questions)
        transformed_X_csr = self.vectorizer.transform(clean_questions)
        transformed_X = transformed_X_csr.A  # csr_matrix to numpy matrix
        return transformed_X

    def query(self, clean_usr_msg):
        try:
            t_usr = self.vectorizer.transform([clean_usr_msg])
            t_usr_array = t_usr.toarray()
            return t_usr_array
        except Exception as e:
            print(e)
            return None

def get_vectoriser(type='tfidf'):
    if type == 'tfidf':
        return TfidfVectorGenerator()
    else:
        return None

class FaqEngine:
    def __init__(self, faqslist, type='tfidf'):
        self.faqslist = faqslist
        self.vector_store = None
        self.stemmer = LancasterStemmer()
        self.le = LE()
        self.classifier = None
        self.vectorizer = None
        self.build_model(type)

    def cleanup(self, sentence):
        word_tok = nltk.word_tokenize(sentence)
        stemmed_words = [self.stemmer.stem(w) for w in word_tok]
        return ' '.join(stemmed_words)

    def build_model(self, type):
        self.vectorizer = get_vectoriser(type)
        dataframeslist = [pd.read_csv(csvfile).dropna() for csvfile in self.faqslist]
        self.data = pd.concat(dataframeslist, ignore_index=True)
        self.data['Clean_Question'] = self.data['Question'].apply(lambda x: self.cleanup(x))
        self.data['Question_embeddings'] = list(self.vectorizer.vectorize(self.data['Clean_Question'].tolist()))
        self.questions = self.data['Question'].values
        X = self.data['Question_embeddings'].tolist()
        X = np.array(X)

        d = X.shape[1]
        index = faiss.IndexFlatL2(d)
        if index.is_trained:
            index.add(X)
        self.vector_store = index

        if 'Class' not in list(self.data.columns):
            return

        y = self.data['Class'].values.tolist()
        if len(set(y)) < 2:  # 0 or 1
            return

        y = self.le.fit_transform(y)

        trainx, testx, trainy, testy = tts(X, y, test_size=.25, random_state=42)

        self.classifier = SVC(kernel='poly', degree=3)
        self.classifier.fit(trainx, trainy)

    def query(self, usr):
        try:
            cleaned_usr = self.cleanup(usr)
            t_usr_array = self.vectorizer.query(cleaned_usr)
            if t_usr_array is None:
                return "Sorry, I couldn't understand your question."

            if self.classifier:
                prediction = self.classifier.predict(t_usr_array)[0]
                class_ = self.le.inverse_transform([prediction])[0]
                questionset = self.data[self.data['Class'] == class_]
            else:
                questionset = self.data

            top_k = 1
            D, I = self.vector_store.search(t_usr_array, top_k)
            question_index = int(I[0][0])
            response = self.data['Answer'][question_index]

            # Create response in the specified format
            response_data = {
                "Answer": response,
                "Context": "predict",
                "Confidence": 1.0
            }

            return response_data

        except Exception as e:
            print(e)
            return None

def train_model(faqslist, model_type='tfidf'):
    faqmodel = FaqEngine(faqslist, model_type)
    return faqmodel

faqmodel = None

@app.route('/faq', methods=['POST'])
def query_faq():
    global faqmodel
    if not faqmodel:
        base_path = ""  # Change this to your data directory
        faqslist = [os.path.join(base_path, "Greetings.csv"), os.path.join(base_path, "GST FAQs 2.csv"), os.path.join(base_path, "BankFAQs.csv")]
        model_type = 'tfidf'  # You can change this to 'doc2vec' or other supported types
        faqmodel = train_model(faqslist, model_type)

    try:
        data = request.get_json()
        user_input = data["Question"]
    except KeyError:
        return jsonify({"message": "Missing 'Question' field in request body."}), 400

    response_data = faqmodel.query(user_input)
    if response_data is None:
        response = "Sorry, I couldn't find an appropriate answer."
    else:
        response = {
            "Answer": response_data["Answer"],
            "Context": "predict",
            "Confidence": 1.0
        }
    return jsonify(response)

if __name__ == "__main__":
    app.run(host='172.17.0.1', port=3001)
