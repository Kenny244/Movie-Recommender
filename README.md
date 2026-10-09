Project summary

This project is a movie recommender system built on the MovieLens 1M dataset. It explores user rating behavior, builds a simple popularity-based recommender, and then creates a personalized recommender using movie genres. The notebook splits the data chronologically into training and test sets, evaluates both models with ranking metrics such as Precision@10, Recall@10, and NDCG@10, and compares which approach performs better. The goal is to test whether user-specific genre preferences improve recommendation quality over a basic popularity baseline.



Simple explanation of what the notebook is doing
The notebook follows a standard recommender-system workflow:

It loads the ratings and movie metadata.
It merges them so each rating is connected to its movie title and genres.
It analyzes the data to understand rating distribution, popular movies, and user behavior.
It creates a baseline recommender that suggests the most popular unseen movies.
It builds a personalized recommender by looking at each user’s liked genres and scoring unseen movies that match those genres.
It splits the dataset into earlier ratings for training and later ratings for testing.
It removes already-seen movies from the recommendation candidates and evaluates recommendations against future relevant movies.
It compares the two models using metrics to see whether personalization actually helps.