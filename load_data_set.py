import json
import os
from typing import Dict

file_dir = os.path.dirname(os.path.realpath('__file__'))

path_pos_reviews = os.path.join(file_dir, 'data/positive_reviews.json')
path_neg_reviews = os.path.join(file_dir, 'data/negative_reviews.json')


class Review:
    def __init__(self, index, title, content, star_rating):
        self.index = index
        self.title = title
        self.content = content
        self.star_rating = star_rating


def load_reviews_from_path(path):
    reviews_train: Dict[str, Review] = {}
    reviews_test: Dict[str, Review] = {}

    try:
        in_file = open(path, "r", encoding="utf-8")
        json_reviews = json.load(in_file)
        idx = 0
        for review in json_reviews['reviews']:
            # The original split from the paper.
            if idx < 6000:
                reviews_train[review['index']] = Review(review['index'],
                                                        review['title'],
                                                        review['content'],
                                                        review['starRating'])
            else:
                reviews_test[review['index']] = Review(review['index'],
                                                       review['title'],
                                                       review['content'],
                                                       review['starRating'])
            idx += 1

        in_file.close()

        return reviews_train, reviews_test

    except IOError as e:
        print(format(e.errno, e.strerror))
    except Exception as e:
        print(e)


def load_data_set():

    positive_reviews_train, positive_reviews_test = load_reviews_from_path(path=path_pos_reviews)
    print("Loaded %d training positive reviews..." % len(positive_reviews_train))
    print("Loaded %d test positive reviews..." % len(positive_reviews_test))

    negative_reviews_train, negative_reviews_test = load_reviews_from_path(path=path_neg_reviews)
    print("Loaded %d training negative reviews..." % len(negative_reviews_train))
    print("Loaded %d test negative reviews..." % len(negative_reviews_test))

    # The LaRoSeDa data set is now loaded in the memory.

    print("\nThe first positive review for training:")
    for review in positive_reviews_train.values():
        print("Index: ", review.index)
        print("Title: ", review.title)
        print("Content: ", review.content)
        print("Star Rating :", review.star_rating)
        print("\n")
        break

    print("\nThe first negative review for training:")
    for review in negative_reviews_train.values():
        print("Index: ", review.index)
        print("Title: ", review.title)
        print("Content: ", review.content)
        print("Star Rating :", review.star_rating)
        print("\n")
        break


load_data_set()
