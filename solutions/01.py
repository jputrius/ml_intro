#| output-location: slide
import numpy as np
import pandas as pd

ratings = pd.read_csv('../data/movielens/ratings.csv')
movies = pd.read_csv('../data/movielens/movies.csv')

popular_movie_ids = ratings.groupby("movieId").size().sort_values(ascending=False)[0:1000].index.sort_values()
popular_movie_names = movies.set_index("movieId").loc[popular_movie_ids]

popular_movie_ratings = ratings[ratings["movieId"].isin(popular_movie_ids)]
crosstable = popular_movie_ratings.pivot_table(values="rating", index="userId", columns="movieId").fillna(2.5)
crosstable = crosstable - crosstable.mean()

A = crosstable.to_numpy()
U, M, V = np.linalg.svd(A, full_matrices=False)
V1 = V[:40].T 

def find_recommendations(movie_name, data):
  movie_id = popular_movie_names[popular_movie_names["title"] == movie_name].index[0]
  movie_idx = np.argwhere(popular_movie_ids == movie_id)[0]
  movie_to_compare = data[movie_idx]

  distances = np.linalg.norm(data - movie_to_compare, axis=1) # Compute the Euclidean distance between the given movie and all others
  similar_ids = popular_movie_ids[distances.argsort()[:10]]
  similar_movies = popular_movie_names.loc[similar_ids]
  return similar_movies["title"].to_numpy() # Take ten movies with lowest distance from given movie

print(f"Movies similar to:")
print(f"Toy Story: {find_recommendations('Toy Story (1995)', V1)}")
print(f"Iron Man: {find_recommendations('Iron Man (2008)', V1)}")