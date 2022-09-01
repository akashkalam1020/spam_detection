from flask import Flask, jsonify, request, render_template
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

app = Flask(__name__)
ps = PorterStemmer()

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/spamdetection', methods = ["POST"])
def spamdetection():
    input_sms = request.form.get('input_sms')

    # 1. preporocess
    transform_sms = tranform_text(input_sms)

    # 2. vectorize
    vector_input = tfidf.transform([transform_sms])

    # 3. predict
    result = model.predict(vector_input)[0]

    # 4. display
    if result == 1:
        result = 'Spam'
    else:
        result = 'Not Spam'

    return render_template('index.html',result=result)

def tranform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return ' '.join(y)

if __name__ =="__main__":
    app.run(debug=True)