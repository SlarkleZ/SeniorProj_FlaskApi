# Flask
import os
from flask import jsonify
from flask import Flask
from flask import request
from json import JSONEncoder

# Web Scrape Library
from bs4 import BeautifulSoup
import urllib
import requests
from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver

# Setting for Chrome
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--window-size=1420,1080')
chrome_options.add_argument('--headless')
chrome_options.add_argument('--disable-gpu')
chrome_options.add_argument('--disable-dev-shm-usage')

# Model Preprocessing Library
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import numpy

nltk.download('wordnet')
nltk.download('stopwords')
wordnet_lemmatizer = WordNetLemmatizer()

# Model Library
import re
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import model_from_yaml
from keras_preprocessing.text import tokenizer_from_json

# Init Key
OMDBkey = '6be019fc'

class NumpyArrayEncoder(JSONEncoder):  # Use for DecoderArrayToList
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def loadModel(yamlPathName, h5PathName):
    with open(yamlPathName + '.yaml', 'r') as yaml_file:
        loaded_model_yaml = yaml_file.read()
        loaded_model = model_from_yaml(loaded_model_yaml)
        loaded_model.load_weights(h5PathName + '.h5')
    loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return loaded_model


def scrapeReviewIMDB(movieCode, isAllReview):
    url = 'http://www.imdb.com/title/' + movieCode + '/reviews?ref_=tt_urv'
    driver = webdriver.Chrome(ChromeDriverManager().install(), chrome_options=chrome_options)
    driver.get(url)
    if isAllReview:
        count = 0
        while True:
            try:
                loadmore = driver.find_element_by_class_name("load-more-data")
                loadmore.click()
                count += 1
                print(count)  # Count time scrape
                if count == 100: raise Exception("Reach!")
            except Exception as e:
                print("Finish!")
                break

    soup = BeautifulSoup(driver.page_source, 'html.parser')

    reviews = []
    for item in soup.select(".review-container"):
        title = item.select(".title")[0].text
        review = item.select(".text")[0].text
        # print("Title: {}\n\nReview: {}\n\n".format(title, review))
        reviews.append([title, review])
        # print(len(reviews))
    scoreUrl = "https://www.imdb.com/title/" + movieCode + "/?ref_=tt_urv"
    try:
        htmlScore = urllib.request.urlopen(scoreUrl).read().decode('utf8')
    except urllib.error.HTTPError:
        return []
    rawScore = BeautifulSoup(htmlScore, 'html.parser')
    scoreData = rawScore.find('script', type='application/ld+json')
    scoreDataJson = json.loads(scoreData.contents[0])
    movieScore = scoreDataJson['aggregateRating']['ratingValue']
    return reviews, movieScore


app = Flask(__name__)
g1 = tf.Graph()
with g1.as_default():
    session1 = tf.compat.v1.Session()
    with session1.as_default():
        model = loadModel('./model/imdb_GRU/Summary', './model/imdb_GRU/Weights')

g2 = tf.Graph()
with g2.as_default():
    session2 = tf.compat.v1.Session()
    with session2.as_default():
        model2 = loadModel('./model/rotten_GRU/Summary', './model/rotten_GRU/Weights')

with open('model/imdb_GRU/tokenizer.json') as f:
    data_imdb = json.load(f)
    tokenizer_imdb = tokenizer_from_json(data_imdb)

with open('model/imdb_GRU/tokenizer.json') as f2:
    data_rotten = json.load(f2)
    tokenizer_rotten = tokenizer_from_json(data_rotten)


@app.route('/', methods=['GET'])
def test():
    return "Hello World."


@app.route('/predict_review_imdb', methods=['POST', 'DELETE'])
def predictScoreIMDB():
    if request.method == 'POST':
        movieCode = request.form["moviecode"]
        reviews, movieScore = scrapeReviewIMDB(movieCode, False)
        positive = 0
        negative = 0
        reviewScore = []
        if not reviews:
            return jsonify(
                result='no review'
            )
        # Lowercase && remove htmltag
        for review in reviews:
            title = review[0]
            review_data = review[1]
            lower_sentence = review_data.lower()
            clean = re.compile('<.*?>')
            sentence_no_tag = re.sub(clean, '', lower_sentence)
            # Tokenization && clean word && Lemmatization
            cleaned = []
            regextokenizer = RegexpTokenizer(r'\w+')
            token_sentence = regextokenizer.tokenize(sentence_no_tag)
            for w in token_sentence:
                if not w in stopwords.words('english'):  # Delete stopwords
                    cleaned.append(wordnet_lemmatizer.lemmatize(w, pos="v"))
            review_cleaned = " ".join(cleaned)

            review_cleaned_token = tokenizer_imdb.texts_to_sequences([review_cleaned])
            review_feat = pad_sequences(review_cleaned_token, maxlen=853)
            # Declare which model we use
            graph_var = g1
            session_var = session1
            with graph_var.as_default():
                with session_var.as_default():
                    result = model.predict(review_feat, verbose=0)
            ###########
            if (result > 0.5):
                positive += 1
            elif (result < 0.5):
                negative += 1
            reviewScore.append([title, review_data, result])

        all_users = [{'reviewCount': str(len(reviewScore)),
                      'positiveReview': str(positive),
                      'negativeReview': str(negative),
                      'imdbScore': movieScore,
                      'allReview': [{
                          'title': each[0],
                          'review': each[1],
                          'score': json.dumps(each[2][0], cls=NumpyArrayEncoder)
                      } for each in reviewScore]}]
        return jsonify(
            all_users
        )


@app.route('/predict_review_rotten', methods=['POST', 'DELETE'])
def predictScoreRotten():
    if request.method == 'POST':
        movieCode = request.form["moviecode"]
        resOMDB = requests.post('http://www.omdbapi.com/?apikey=' + OMDBkey + '&tomatoes=true&i=' + movieCode)
        dataOMDB = resOMDB.json()

        tomatoURL = dataOMDB['tomatoURL']
        if not tomatoURL.endswith('/'):
            tomatoURL = tomatoURL + '/'
        tomatoURL = tomatoURL + 'reviews?type=user'
        movieScore = dataOMDB['Ratings']
        tomatoScore = ''
        for eachScore in movieScore:
            if eachScore['Source'] == 'Rotten Tomatoes':
                tomatoScore = eachScore['Value']

        reviews = []
        print(tomatoURL)
        driver = webdriver.Chrome(ChromeDriverManager().install(), chrome_options=chrome_options)
        driver.get(tomatoURL)
        # html = urllib.request.urlopen(tomatoURL).read().decode('utf8')
        # html[:400]

        soup = BeautifulSoup(driver.page_source, 'html.parser')
        # print(data)
        for item in soup.select(".audience-reviews__item"):
            review = item.select(".audience-reviews__review")[0].text
            reviews.append(review)

        if not reviews:
            return jsonify(
                result='no review'
            )

        positive = 0
        negative = 0
        reviewScore = []
        for review in reviews:
            review_data = review
            lower_sentence = review_data.lower()
            clean = re.compile('<.*?>')
            sentence_no_tag = re.sub(clean, '', lower_sentence)
            # Tokenization && clean word && Lemmatization
            cleaned = []
            regextokenizer = RegexpTokenizer(r'\w+')
            token_sentence = regextokenizer.tokenize(sentence_no_tag)
            for w in token_sentence:
                if not w in stopwords.words('english'):  # Delete stopwords
                    cleaned.append(wordnet_lemmatizer.lemmatize(w, pos="v"))
            review_cleaned = " ".join(cleaned)
            review_cleaned_token = tokenizer_rotten.texts_to_sequences([review_cleaned])
            review_feat = pad_sequences(review_cleaned_token, maxlen=28)
            # Declare which model we use
            graph_var = g2
            session_var = session2
            with graph_var.as_default():
                with session_var.as_default():
                    result = model2.predict(review_feat, verbose=0)
            ###########
            if (result > 0.5):
                positive += 1
            elif (result < 0.5):
                negative += 1
            reviewScore.append([review_data, result])

        all_users = [{'reviewCount': str(len(reviewScore)),
                      'positiveReview': str(positive),
                      'negativeReview': str(negative),
                      'rottenScore': tomatoScore,
                      'allReview': [{
                          'review': each[0],
                          'score': json.dumps(each[1][0], cls=NumpyArrayEncoder)
                      } for each in reviewScore]}]
        return jsonify(
            all_users
        )


@app.route('/predict_allreview_imdb', methods=['POST', 'DELETE'])
def predictScoreAllReviewIMDB():
    movieCode = request.form["moviecode"]
    reviews, movieScore = scrapeReviewIMDB(movieCode, True)
    positive = 0
    negative = 0
    reviewScore = []
    if not reviews:
        return jsonify(
            result='no review'
        )
    # Lowercase && remove htmltag
    for review in reviews:
        title = review[0]
        review_data = review[1]
        lower_sentence = review_data.lower()
        clean = re.compile('<.*?>')
        sentence_no_tag = re.sub(clean, '', lower_sentence)
        # Tokenization && clean word && Lemmatization
        cleaned = []
        regextokenizer = RegexpTokenizer(r'\w+')
        token_sentence = regextokenizer.tokenize(sentence_no_tag)
        for w in token_sentence:
            if not w in stopwords.words('english'):  # Delete stopwords
                cleaned.append(wordnet_lemmatizer.lemmatize(w, pos="v"))
        review_cleaned = " ".join(cleaned)

        review_cleaned_token = tokenizer_imdb.texts_to_sequences([review_cleaned])
        review_feat = pad_sequences(review_cleaned_token, maxlen=853)
        # Declare which model we use
        graph_var = g1
        session_var = session1
        with graph_var.as_default():
            with session_var.as_default():
                result = model.predict(review_feat, verbose=0)
        ###########
        if (result > 0.5):
            positive += 1
        elif (result < 0.5):
            negative += 1
        reviewScore.append([title, review_data, result])

    all_users = [{'reviewCount': str(len(reviewScore)),
                  'positiveReview': str(positive),
                  'negativeReview': str(negative),
                  'imdbScore': movieScore,
                  'allReview': [{
                      'title': each[0],
                      'review': each[1],
                      'score': json.dumps(each[2][0], cls=NumpyArrayEncoder)
                  } for each in reviewScore]}]
    return jsonify(
        all_users
    )


@app.route('/predict_allreview_rotten', methods=['POST', 'DELETE'])
def predictScoreAllReviewRotten():
    if request.method == 'POST':
        movieCode = request.form["moviecode"]
        resOMDB = requests.post('http://www.omdbapi.com/?apikey=6be019fc&tomatoes=true&i=' + movieCode)
        dataOMDB = resOMDB.json()

        tomatoURL = dataOMDB['tomatoURL']
        if not tomatoURL.endswith('/'):
            tomatoURL = tomatoURL + '/'
        tomatoURL = tomatoURL + 'reviews?type=user'
        movieScore = dataOMDB['Ratings']
        tomatoScore = ''
        for eachScore in movieScore:
            if eachScore['Source'] == 'Rotten Tomatoes':
                tomatoScore = eachScore['Value']

        reviews = []
        print(tomatoURL)
        driver = webdriver.Chrome(ChromeDriverManager().install(), chrome_options=chrome_options)
        driver.get(tomatoURL)

        # Page to scrape here
        for i in range(6):
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            for item in soup.select(".audience-reviews__item"):
                review = item.select(".audience-reviews__review")[0].text
                reviews.append(review)
            next = driver.find_element_by_class_name("js-prev-next-paging-next")
            next.click()
        if not reviews:
            return jsonify(
                result='no review'
            )
        positive = 0
        negative = 0
        reviewScore = []
        for review in reviews:
            review_data = review
            lower_sentence = review_data.lower()
            clean = re.compile('<.*?>')
            sentence_no_tag = re.sub(clean, '', lower_sentence)
            # Tokenization && clean word && Lemmatization
            cleaned = []
            regextokenizer = RegexpTokenizer(r'\w+')
            token_sentence = regextokenizer.tokenize(sentence_no_tag)
            for w in token_sentence:
                if not w in stopwords.words('english'):  # Delete stopwords
                    cleaned.append(wordnet_lemmatizer.lemmatize(w, pos="v"))
            review_cleaned = " ".join(cleaned)
            review_cleaned_token = tokenizer_rotten.texts_to_sequences([review_cleaned])
            review_feat = pad_sequences(review_cleaned_token, maxlen=28)
            # Declare which model we use
            graph_var = g2
            session_var = session2
            with graph_var.as_default():
                with session_var.as_default():
                    result = model2.predict(review_feat, verbose=0)
            ###########
            if (result > 0.5):
                positive += 1
            elif (result < 0.5):
                negative += 1
            reviewScore.append([review_data, result])

        all_users = [{'reviewCount': str(len(reviewScore)),
                      'positiveReview': str(positive),
                      'negativeReview': str(negative),
                      'rottenScore': tomatoScore,
                      'allReview': [{
                          'review': each[0],
                          'score': json.dumps(each[1][0], cls=NumpyArrayEncoder)
                      } for each in reviewScore]}]
        return jsonify(
            all_users
        )


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=os.getenv('PORT'))
