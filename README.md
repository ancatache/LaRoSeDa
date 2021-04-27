# LaRoSeDa - A Large Romanian Sentiment Data Set

## License Agreement

**Copyright (C) 2021 - Anca Maria Tache, Mihaela Gaman, Radu Tudor Ionescu**

This work is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/). 

You are free to **share** (copy and redistribute the material in any medium or format) and **adapt** (remix, transform, and build upon the material) under the following terms:
- **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
- **NonCommercial** — You may not use the material for commercial purposes.
- **ShareAlike** — If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original.
- **No additional restrictions** — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.

## Citation

Please cite the corresponding work (see citation.bib file to obtain the citation in [BibTex](citation.bib) format) if you use this data set and software (or a modified version of it) in any scientific work:

**[1] Anca Maria Tache, Mihaela Gaman, Radu Tudor Ionescu. Clustering Word Embeddings with Self-Organizing Maps. Application on
LaRoSeDa – A Large Romanian Sentiment Data Set. In Proceedings of EACL, pp. 949–956, 2021.  [(link to paper)](https://www.aclweb.org/anthology/2021.eacl-main.81.pdf).**

## Dataset

The dataset contains 15,000 reviews written in Romanian, of which 7500 are positive and 7500 negative. It is provided as two JSON files:
- data/positive_reviews.json
- data/negative_reviews.json
 
The data format is as follows:

```
{
    "reviews": [
        {
            "index": "Index_1",
            "title": "Title_1",
            "content": "Content_1",
            "starRating": "StarRating_1"
        },
        ...
        {
            "index": "Index_n",
            "title": "Title_n",
            "content": "Content_n",
            "starRating": "StarRating_n"
        }
    ]
}
  ```
  
Each review contains an index, the title, content and the associated star rating which can be 1 or 2 for the negative reviews and 4 or 5 for the positive reviews.

In the experiments presented in the paper we split the data in two subsets:
- training: 6000 positive samples, 6000 negative samples
- test: 1500 positive samples, 1500 negative samples

For convenience, we provide a Python loader for the dataset which does this split (_load_data_set.py_).

## BOSWE implementation

We provide two scripts for the BOSWE implementation:

- one that uses vectors generated using word2vec (_boswe.py_) - the word2vec model is computed from input data
- one that uses vectors generated using BERT (_boswe_bert.py_) - the BERT model used is from
https://huggingface.co/dumitrescustefan/bert-base-romanian-cased-v1

The scripts generate histograms for the input reviews using two clustering methods for word embeddings: SOFM and k-means.
The output for each is as follows:

- _histo_boswe_sofm_train.txt_, _histo_boswe_sofm_test.txt_, _histo_boswe_kmeans_train.txt_, _histo_boswe_kmeans_test.txt_ 

    ```
    Index_1. [Histogram_1]
    ...
    Index_n. [Histogram_n]
    ```
- _labels_train.txt_, _labels_test.txt_

    ```
    Index_1. Label_1
    ...
    Index_n. Label_n
    ```

The labels are _-1_ for the negative reviews and _+1_ for the positive ones.

## Open source software

In order to obtain pre-computed kernels for the learning stage, we applied [PQ kernel](http://pq-kernel.herokuapp.com/) <sup>[1]</sup> on the BOSWE histograms. 

For the fusion with the string kernels representation we used [String Kernels implementation](http://string-kernels.herokuapp.com/) <sup>[2]</sup> to generate the HISK kernel matrix.

**[1] Radu T. Ionescu, Marius Popescu. PQ kernel: a rank correlation kernel for visual word histograms. Pattern Recognition Letters, vol. 55, pp. 51-57, 2015. [(PQK 1.0 software)](http://pq-kernel.herokuapp.com/)**

**[2] Marius Popescu and Radu Tudor Ionescu. The Story of the Characters, the DNA and the Native Language. Proceedings of the Eighth Workshop on Innovative Use of NLP for Building Educational Applications, pp. 270–278, 2013. [(String Kernels 1.0 software)](http://string-kernels.herokuapp.com/)**