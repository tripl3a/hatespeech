# hatespeech

preprocessing / word vectors:

https://github.com/facebookresearch/fastText/blob/master/python/README.md

https://www.kaggle.com/mschumacher/using-fasttext-models-for-robust-embeddings/notebook

https://github.com/facebookresearch/fastText/issues/435

scikit-fasttext-estimator:

https://github.com/shaypal5/skift

http://scikit-learn.org/stable/developers/contributing.html#rolling-your-own-estimator

https://github.com/vishnumani2009/sklearn-fasttext

more ideas:

https://becominghuman.ai/my-solution-to-achieve-top-1-in-a-novel-data-science-nlp-competition-db8db2ee356a

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


###############

# Data Preprocessing and Normalization 
Improve the performance of the model applying some simple pre-processing
- Translation into English 
- only ASCII characters 
- remove special characters 
- Change Emojis to words

# Translation into English
```py 
def translate(text):
    # The target language
    target = 'en'

    # Translates some text into English
    translation = translate_client.translate(
            text,
            target_language=target)

    return translation['translatedText']
```

# Replace Characters
```py 
def replacetext(text):
    for key, value in REPLACE_TO.items():
        text = text.replace(key, value)
    return text
```

 ```py 
REPLACE_TO = {
    ':)': 'happy',':(': 'sad',':P': 'funny','@': 'at','&': 'and','i\'m': 'i am','don\'t': 'do not','can\'t': 'can not',
    '.': '',',': '',':': '', ';': '','!': '','\'': ' ','?': ' ', '(': ' ', ')': ' ', '[': ' ', ']': ' ', '-': ' ',
    '#': ' ', '=': ' ', '+': ' ', '/': ' ', '"': ' ', '0': ' zero ', '1': ' one ', '2': ' two ', '3': ' three ', '3': ' three ',
    '4': ' four ','5': ' five ', '6': ' six ', '7': ' seven ', '8': ' eight ', '9': ' nine ' }

```

# Unidecode and Google API
```py 
if __name__ == '__main__':
    try: 
        last_counter = int(sys.argv[1])
    except IndexError:
        last_counter = 0
with open('out_copy.csv', 'r') as f, open('data_unidecode.csv', 'a') as write_file:
    reader = csv.reader(f, delimiter = ",", quotechar = '"')
    writer = csv.writer(write_file, delimiter = ",", quotechar = '"')
        for counter, entry in enumerate(reader):
            if counter < last_counter:
                continue
            print(counter)
            converted_entry = unidecode(entry[1])
            converted_entry = normalize_commentary(entry[1])
            writer.writerow([entry[0], converted_entry, entry[2], entry[3], entry[4], entry[5], entry[6]])
```

# Google Dashboard 

![Dashboard](hatespeech/Screen Shot 2018-05-14 at 00.43.05.png)

