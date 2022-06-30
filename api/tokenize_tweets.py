import re
import json
from utils.data import Text

import logging

TWITTER_USER_RE = re.compile(r'@([A-Za-z0-9_]+)')
TWITTER_URL_RE = re.compile(r'https?:\/\/\S+')
TWITTER_USER_PH = '@USER' + ('_' * 15)
TWITTER_URL_PH = 'HTTPURL' + ('_' * 93)

JSON_TWEETS = "api/data/raw_tweets.json"
OUT = "api/data/normalized_tweets.txt"


def normalize_user_url(text: str, lower: bool = False) -> str:
    text = TWITTER_USER_RE.sub(lambda m: m.group().replace(m.group(), TWITTER_USER_PH[:len(m.group())], 1), text)
    text = TWITTER_URL_RE.sub(lambda m: m.group().replace(m.group(), TWITTER_URL_PH[:len(m.group())], 1), text)
    text = text.lower() if lower else text
    text = text.replace("\"", " ")
    text = text.replace("\n", " ")
    return text


def load_json_texts(lower: bool = True, normalize: bool = True):
    texts = []

    data_tweets = []
    for line in open(JSON_TWEETS, 'r'):
        data_tweets.append(json.loads(line))

    for json_tweet in data_tweets:
        tid, text = json_tweet["id_str"], json_tweet["full_text"]
        text_obj = Text(tid, text)
        text_obj.text = text.lower() if lower else text
        if normalize:
            text_obj.text = normalize_user_url(text_obj.text)
        texts.append(text_obj)
    logging.info("Loaded %s texts", len(texts))
    write_text(texts)


def write_text(text_list):
    with open(OUT, 'w', encoding="utf-8") as f:
        for text_obj in text_list:
            line = f"{text_obj.tid}\t{text_obj.text}\n"
            f.write(line)


def main():
    # load json raw tweets and write them in a txt file normalized
    load_json_texts(JSON_TWEETS)


if __name__ == '__main__':
    main()
