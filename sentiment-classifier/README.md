# Sentiment Classifier

A sentiment classifier using Yelp reviews [data set](https://www.yelp.com/dataset), with an interface that takes a string and returns a sentiment (positive, neutral, negative).

## How to execute

### Build
$ docker build  --tag sentiment-classifier    --file Dockerfile     --pull . 

### Run
$ docker run -it  --publish 5000:5000  sentiment-classifier <br>

- open http://0.0.0.0:5000/

