import argparse
import logging
import sys
import os
from tweepy import API, OAuthHandler, Cursor
import json
from typing import List

logging.basicConfig(format='%(asctime)s: %(levelname)s:%(message)s',
                    level=logging.DEBUG)


def get_api(keys_file: str) -> API:
    if os.path.exists(keys_file):
        with open(keys_file) as f:
            key = json.load(f)
    else:
        logging.info("File %s does not exist in the specified path.", keys_file)
        sys.exit(0)
    # connect to twitter
    consumer_key, consumer_secret = key["CONSUMER_KEY"], key["CONSUMER_SECRET"]
    access_token, access_token_secret = key["ACCESS_TOKEN"], key["ACCESS_SECRET"]
    if all([consumer_key, consumer_key, access_token, access_token_secret]):
        auth = OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        api = API(auth, wait_on_rate_limit=True)
        return api
    else:
        logging.info("Please check twitter key entries and try again")
        sys.exit(0)


def download_tweets(api, keywords: List[str], max: int):
    tweets = []
    for keyword in keywords:
        logging.info("Searching '%s'", keyword)
        tweets += [status for status in
                   Cursor(api.search_tweets, q=keyword + " -RT -is:retweet lang:en", tweet_mode="extended").items(max)]
    return tweets


def write_json_line(data, path):
    with open(path, 'w') as f:
        for i in data:
            f.write("%s\n" % json.dumps(i._json))
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keywords", help="search keywords comma separated",
                        type=str, default='dexedrine,ritalin,adderall,concerta,vyvanse')
    parser.add_argument("--out", help="path to the output file",
                        type=str, default="api/data/raw_tweets.json")
    parser.add_argument("--keys", help="path to twitter keys file in json format",
                        type=str, default='api/resources/twitter_keys.json')
    parser.add_argument("--max", help="Max results per keyword",
                        type=int, default=1000)
    # Parse arguments
    try:
        args = parser.parse_args()
        logging.info("Input Arguments : %s", args)
    except:
        parser.print_help()
        sys.exit(0)
    # read key file and get api
    api = get_api(args.keys)
    # download tweets
    keywords = args.keywords.split(",")
    tweets = download_tweets(api, keywords, args.max)
    # write to file
    write_json_line(tweets, args.out)


if __name__ == '__main__':
    main()
