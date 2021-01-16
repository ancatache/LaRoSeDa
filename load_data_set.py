import json
import os
from typing import Dict

file_dir = os.path.dirname(os.path.realpath('__file__'))

path_pos_reviews = os.path.join(file_dir, 'data/positive_reviews.json')
path_neg_reviews = os.path.join(file_dir, 'data/negative_reviews.json')

# It provides a class describing the review entity
class Review:
    def __init__(self, index, title, content, star_rating):
        self.index = index
        self.title = title
        self.content = content
        self.star_rating = star_rating

# Loads in the memory the reviews from a specific JSON file
# and splits them in two dictionaries, one for train and one for test,
# according to the number given as second parameter.
def load_reviews_from_path(path, train_samples_number):
    reviews_train: Dict[str, Review] = {}
    reviews_test: Dict[str, Review] = {}

    try:
        in_file = open(path, "r", encoding="utf-8")
        json_reviews = json.load(in_file)
        # first item from the reviews array must have the minimum index
        max_index_training = int(json_reviews['reviews'][0]['index']) + train_samples_number
        for review in json_reviews['reviews']:
            if int(review['index']) < max_index_training:
                reviews_train[review['index']] = Review(review['index'],
                                                        review['title'],
                                                        review['content'],
                                                        review['starRating'])
            else:
                reviews_test[review['index']] = Review(review['index'],
                                                       review['title'],
                                                       review['content'],
                                                       review['starRating'])

        in_file.close()

        return reviews_train, reviews_test

    except IOError as e:
        print(format(e.errno, e.strerror))
    except Exception as e:
        print(e)

# It provides an example for the usage of load_reviews_from_path function.
def load_data_set():
    # The original split from the paper.
    positive_reviews_train, positive_reviews_test = load_reviews_from_path(path=path_pos_reviews,
                                                                           train_samples_number=6000)
    print("Loaded %d training positive reviews..." % len(positive_reviews_train))
    print("Loaded %d test positive reviews..." % len(positive_reviews_test))

    negative_reviews_train, negative_reviews_test = load_reviews_from_path(path=path_neg_reviews,
                                                                           train_samples_number=6000)
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

# load_data_set()
