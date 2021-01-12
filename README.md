# LaRoSeDa - A Large Romanian Sentiment Data Set

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
  
Each review contains an index, the title, content and the associated star rating which can be 1 or 2 for the negative reviews and 3 or 4 for the positive reviews.

In the experiments presented in the paper we split the data in two subsets:
- training: 6000 positive samples, 6000 negative samples
- test: 1500 positive samples, 1500 negative samples

For convenience, we provide a Python loader for the dataset which does this split.





