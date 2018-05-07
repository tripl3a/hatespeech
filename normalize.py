'''
created on Sat 05.05.18
@Autor: Tata
'''
# pip install --upgrade google-cloud-translate

import re
import csv
from pprint import pprint 
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

def normalizefile(fileobj):
    lines = []
    for line in fileobj:
        line = translate(line)
        line = replacetext(line)
        lines.append(line)
    return lines 

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
    with open('preprocesingtest.txt', 'r') as f:
        lines  = normalizefile(f) 
        pprint(lines)
