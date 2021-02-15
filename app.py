# Flask
from flask import jsonify
from flask import Flask
from flask import request
from json import JSONEncoder
from flask import Response

# Web Scrape Library
from bs4 import BeautifulSoup
from selenium import webdriver
import urllib
import requests
from webdriver_manager.microsoft import EdgeChromiumDriverManager
from msedge.selenium_tools import Edge, EdgeOptions
#Setting for Edge Chromium
options = EdgeOptions()
options.use_chromium = True


# Model Preprocessing Library
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import numpy
from selenium.common.exceptions import ElementNotInteractableException

nltk.download('wordnet')
wordnet_lemmatizer = WordNetLemmatizer()

# Model Library
import re
import json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import model_from_yaml
from keras_preprocessing.text import tokenizer_from_json


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


def scrapeReview(movieCode):
    url = 'http://www.imdb.com/title/' + movieCode + '/reviews?ref_=tt_urv'
    driver = Edge(EdgeChromiumDriverManager().install(), options=options)
    driver.get(url)

    count = 0
    while True:
        try:
            loadmore = driver.find_element_by_class_name("load-more-data")
            loadmore.click()
            count += 1
            print(count)  # Count time scrape
            if (count == 50): raise Exception("Reach!")
        except Exception as e:
            print("Finish!")
            break

    soup = BeautifulSoup(driver.page_source, 'html.parser')

    reviews = []
    for item in soup.select(".review-container"):
        title = item.select(".title")[0].text
        review = item.select(".text")[0].text
        print("Title: {}\n\nReview: {}\n\n".format(title, review))
        reviews.append([title,review])
    #print(len(reviews))
    scoreUrl = "https://www.imdb.com/title/" + movieCode + "/?ref_=tt_urv"
    try:
        htmlScore = urllib.request.urlopen(scoreUrl).read().decode('utf8')
        htmlScore[:400]
    except urllib.error.HTTPError as e:
        return [];
    rawScore = BeautifulSoup(htmlScore, 'html.parser')
    scoreData = rawScore.find('script', type='application/ld+json')
    scoreDataJson = json.loads(scoreData.contents[0])
    movieScore = scoreDataJson['aggregateRating']['ratingValue']
    return reviews, movieScore

app = Flask(__name__)
model = loadModel('./model/main_1_GRU/Summary', './model/main_1_GRU/Weights')
with open('./model/main_1_GRU/tokenizer.json') as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)


@app.route('/', methods=['GET'])
def test():
    return "Hello World."


@app.route('/predict_review', methods=['POST', 'DELETE'])
def predictScore():
    if request.method == 'POST':
        movieCode = request.form["moviecode"]
        reviews, movieScore = scrapeReview(movieCode)
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
                if not w in stopwords.words('English'):  # Delete stopwords
                    cleaned.append(wordnet_lemmatizer.lemmatize(w, pos="v"))
            review_cleaned = " ".join(cleaned)

            review_cleaned_token = tokenizer.texts_to_sequences([review_cleaned])
            review_feat = pad_sequences(review_cleaned_token, maxlen=853)
            result = model.predict(review_feat, verbose=0)
            if (result > 0.5):
                positive += 1
            elif (result < 0.5):
                negative += 1
            reviewScore.append([title, review_data, result])

        all_users = [{'reviewCount': str(len(reviewScore)),
                      'positiveReview': str(positive),
                      'negativeReview': str(negative),
                      'movieScore': movieScore,
                      'allReview': [{
                          'title': each[0],
                          'review': each[1],
                          'score': json.dumps(each[2][0], cls=NumpyArrayEncoder)
                      } for each in reviewScore]}]
        return jsonify(
            all_users
        )


if __name__ == '__main__':
    app.run()
