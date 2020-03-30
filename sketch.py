import pickle

import pymorphy2
morph = pymorphy2.MorphAnalyzer()

from nltk.corpus import stopwords
stop_words = stopwords.words('russian')

from nltk.tokenize import WordPunctTokenizer
tokenizer = WordPunctTokenizer()

import gensim
import re
import os

from flask import Flask, jsonify, request, render_template


app = Flask(__name__)

@app.before_first_request
def load_model_to_app():
    with open(r'corpus.pickle', 'rb') as f:
        corpus = pickle.load(f)
    with open(r'static\model\id2word.pickle', 'rb') as f:
        app.id2word = pickle.load(f)
        
    os.environ['MALLET_HOME'] = 'C:\\mallet-2.0.8'
    mallet_path = 'C:\\mallet-2.0.8\\bin\\mallet.bat'

    app.ldamallet = gensim.models.wrappers.LdaMallet(mallet_path=mallet_path, corpus=corpus, num_topics=5, id2word=app.id2word)

    #app.predictor = load_model('./static/model/model.h5')

@app.route("/")
def index():
    return render_template('index.html', views=0, reactions=0)

@app.route('/predict', methods=['POST'])
def predict():
    #with open(r'static\model\ldamallet.pickle', 'rb') as f:
    #    ldamallet = pickle.load(f)
	
    with open(r'static\model\bigram_mod.pickle', 'rb') as f:
        bigram_mod = pickle.load(f)
	
    #with open(r'static\model\id2word.pickle', 'rb') as f:
    #    id2word = pickle.load(f)
    with open(r'static\model\stats.pickle', 'rb') as f:
        stats = pickle.load(f)
        
    upost = request.form['post_text']
    upost = upost.lower()
    tokens = [morph.parse(tok)[0].normal_form for tok in tokenizer.tokenize(upost) if tok.isalpha() and len(tok) > 2 and tok not in stop_words and bool(re.match(r'[а-я]', tok)) and morph.parse(tok)[0].tag.POS not in ['NPRO', "NUMR", "CONJ", "INTJ", "PRCL", "PREP"]]

    upost_t = bigram_mod[tokens]
    upost_t = app.id2word.doc2bow(upost_t)

    topics = app.ldamallet[upost_t]
    prim_topic = sorted(topics, key=lambda x: x[1], reverse=True)[0][0]
    result = stats[prim_topic]

    return render_template('index.html', views=round(result['views']), reactions=round(result['reactions']))

if __name__ == '__main__':
    app.run(debug=False)
  
