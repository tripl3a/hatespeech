'''
created on Sat 05.05.18
updated on thr 10.05.18
@Autor: Tata
'''
# pip install --upgrade google-cloud-translate

import re
import sys
import csv
from pprint import pprint 
from time import sleep
# Imports the Google Cloud client library
from google.cloud import translate

# Instantiates a client
translate_client = translate.Client()

REPLACE_TO = {
    ':)': 'happy',
    ':(': 'sad',
    ':P': 'funny',
    '@': 'at',
    '&': 'and',
    'i\'m': 'i am',
    'don\'t': 'do not',
    'can\'t': 'can not'
}

def normalize_commentary(cmt):

    if 'the' not in cmt.lower().split():
        cmt = translate(cmt)
        sleep(0.1)
    cmt = replacetext(cmt)
    return cmt

def translate(text):
    # The target language
    target = 'en'

    # Translates some text into English
    translation = translate_client.translate(
            text,
            target_language=target)

    return translation['translatedText']

def replacetext(text):
    for key, value in REPLACE_TO.items():
        text = text.replace(key, value)
    return text


if __name__ == '__main__':
    try: 
        last_counter = int(sys.argv[1])
    except IndexError:
        last_counter = 0
    with open('data/kaggle-data.csv', 'r') as f, open('out.csv', 'a') as write_file:
        reader = csv.reader(f, delimiter = ",", quotechar = '"')
        writer = csv.writer(write_file, delimiter = ",", quotechar = '"')
        for counter, entry in enumerate(reader):
            if counter < last_counter:
                continue
            print(counter)
            converted_entry = normalize_commentary(entry[1])
            writer.writerow([entry[0], converted_entry, entry[2], entry[3], entry[4], entry[5], entry[6]])
