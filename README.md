# hatespeech

preprocessing / word vectors:

https://github.com/facebookresearch/fastText/blob/master/python/README.md

https://www.kaggle.com/mschumacher/using-fasttext-models-for-robust-embeddings/notebook

https://www.kaggle.com/sbongo/do-pretrained-embeddings-give-you-the-extra-edge/code

https://github.com/facebookresearch/fastText/issues/435

scikit-fasttext-estimator:

https://github.com/shaypal5/skift

http://scikit-learn.org/stable/developers/contributing.html#rolling-your-own-estimator

https://github.com/vishnumani2009/sklearn-fasttext

more ideas:

https://becominghuman.ai/my-solution-to-achieve-top-1-in-a-novel-data-science-nlp-competition-db8db2ee356a

https://www.kaggle.com/jagangupta/lessons-from-toxic-blending-is-the-new-sexy/code

https://www.kaggle.com/yekenot/pooled-gru-fasttext

the kaggle challenge:

https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge


# preprocessing! 

based on the [NLP Competition](https://becominghuman.ai/my-solution-to-achieve-top-1-in-a-novel-data-science-nlp-competition-db8db2ee356a) link suggested by Arndt.

## General Idea

Modify input data to improve the hate speech detection (F1 Score)
- Translate any language into English Using the Google API (improve quality of the data)
- Replace images with words
- Replace special symbols example “I’m” to I am
- Remove url links and IP addresses (have to be done)
- convert & to “and”, @ to “at” 
- ASCII representation using Unidecode


## to run it

first you need to get the google translate API, you get a year free trial and 300 Euros to spend. 
get the key and run: (use your own key)

GOOGLE_APPLICATION_CREDENTIALS=/Users/tata/Downloads/NormalizingData-8ba09ff29093. json python3 normalize.py (int)
