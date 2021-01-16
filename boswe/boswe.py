from boswe_utils import *

doc_index = 1 # this will be appended to the name of the output file
no_of_clusters = 500
size_word2vec = 300
epochs_sofm = 200
epochs_word2vec = 200
train_samples_number = 6000
test_samples_number = 1500

reviews_words_pos_train, reviews_words_neg_train, reviews_words_pos_test, reviews_words_neg_test, reviews, \
    all_sentences_pos, all_sentences_neg = load_reviews(train_samples_number)

reviews_words_train = concatenate_dicts([reviews_words_pos_train, reviews_words_neg_train])
shuffled_reviews_train = shuffle_dict(reviews_words_train, train_seed)

reviews_words_test = concatenate_dicts([reviews_words_pos_test, reviews_words_neg_test])
shuffled_reviews_test = shuffle_dict(reviews_words_test, test_seed)

reviews_words = concatenate_dicts([shuffled_reviews_train, shuffled_reviews_test])

all_sentences = all_sentences_pos + all_sentences_neg
all_sentences_shuffled = shuffle_all_sentences(all_sentences, sentences_seed)

data, vocabulary_ordered = compute_word_to_vec(all_sentences=all_sentences_shuffled,
                                               size_word2vec=size_word2vec,
                                               epochs_word2vec=epochs_word2vec)

# training
cluster_no_sofm = training(cluster_method='sofm',
                           n_clusters=no_of_clusters,
                           data=data,
                           vocabulary_ordered=vocabulary_ordered,
                           size_vec=size_word2vec,
                           epochs_sofm=epochs_sofm)

cluster_no_kmeans = training(cluster_method='kmeans',
                             n_clusters=no_of_clusters,
                             data=data,
                             vocabulary_ordered=vocabulary_ordered,
                             size_vec=size_word2vec,
                             epochs_sofm=epochs_sofm)

file_labels_train = open(os.path.join(file_dir, '../data/labels_train_' + str(doc_index) + '.txt'), 'w')
file_labels_test = open(os.path.join(file_dir, '../data/labels_test_' + str(doc_index) + '.txt'), 'w')
file_histo_sofm_train = open(os.path.join(file_dir, '../data/histo_boswe_sofm_train_' + str(doc_index) + '.txt'), 'w')
file_histo_kmeans_train = open(os.path.join(file_dir, '../data/histo_boswe_kmeans_train_' + str(doc_index) + '.txt'),
                               'w')
file_histo_sofm_test = open(os.path.join(file_dir, '../data/histo_boswe_sofm_test_' + str(doc_index) + '.txt'), 'w')
file_histo_kmeans_test = open(os.path.join(file_dir, '../data/histo_boswe_kmeans_test_' + str(doc_index) + '.txt'), 'w')

# create histograms
for index, (key, words_from_review) in enumerate(reviews_words.items()):
    histogram_sofm = create_histogram_one_voc(words_from_review=words_from_review,
                                              n_super_words=no_of_clusters,
                                              clusters_no=cluster_no_sofm)
    histogram_kmeans = create_histogram_one_voc(words_from_review=words_from_review,
                                                n_super_words=no_of_clusters,
                                                clusters_no=cluster_no_kmeans)

    if int(reviews[key].star_rating) == 4 or int(reviews[key].star_rating) == 5:
        max_index_training = train_samples_number
    else:
        max_index_training = train_samples_number * 2 + test_samples_number

    if int(reviews[key].index) <= max_index_training:
        write_histo(reviews_dict=reviews,
                    key=key,
                    histogram=histogram_sofm,
                    file_histo=file_histo_sofm_train)
        write_histo(reviews_dict=reviews,
                    key=key,
                    histogram=histogram_kmeans,
                    file_histo=file_histo_kmeans_train)
        # iterate through reviews_dict with key from review in order to obtain rating_star
        write_labels(reviews_dict=reviews,
                     key=key,
                     file_labels=file_labels_train)
    else:
        write_histo(reviews_dict=reviews,
                    key=key,
                    histogram=histogram_sofm,
                    file_histo=file_histo_sofm_test)
        write_histo(reviews_dict=reviews,
                    key=key,
                    histogram=histogram_kmeans,
                    file_histo=file_histo_kmeans_test)
        write_labels(reviews_dict=reviews,
                     key=key,
                     file_labels=file_labels_test)

file_labels_train.close()
file_histo_sofm_train.close()
file_histo_kmeans_train.close()
file_labels_test.close()
file_histo_sofm_test.close()
file_histo_kmeans_test.close()
