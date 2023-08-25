---
layout: post
title: 'The Asiak Movie Engine!'
date: '2022-06-18'
categories: machine learning
---

### INTRODUCTION
<p>I love movies. Most of the movies I watch are influenced by their ratings and reviews online. 
I decided to build a simple recommender system for movies deployed on streamlit based on the ratings of other users gathered in the Movie lens dataset from the University of Minnesota.
Let me show you how to do this but before that, a little about the KNN algorithm.</p>

### K-NEAREST NEIGHBOR
<p>
  The K-Nearest Neighbor is a non-parametric learning algorithm used for both classification and regression tasks. 
  In simpler terms, the KNN does not really know anything about its data. 
  The learning strategy of the K-Nearest Neighbor algorithm is to find the most similar observations to the one you have to predict assuming that similar things are near each other.<br>
  **Let’s put this in context through an analogy;**

  Once a new example or input x from previously unseen data comes in, the KNN algorithm finds the closest k-training examples to x and returns the most       frequent class among them in the case of classification or the average class in the case of regression.
</p>

## HOW I BUILT THE RECOMMENDER SYSTEM

### PREPROCESSING
Preprocessing was the most important part of the building process. It was crucial to modify the data in order to ensure the classifier is able to give good classifications

1. LOADING AND EXPLORING DATA
  The first step in preprocessing is to load the data and perform some quick data exploration to understand how it should be preprocessed. This is true       for any machine learning or data science project.

    import pandas as pd 
    import numpy as np 
    import seaborn as sns
    from scipy.sparse import csr_matrix
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv('ratings.csv')

2. CONVERTING TO ZERO VALUE RATING
3. REMOVING USERS WITHOUT ENOUGH RATINGS
   Users who did not rate up to 50 movies were removed from the dataset. The logic is that, if you have not rated up to 50 movies, your ratings are 
   probably going to skew the data or you might just not be a legitimate movie rater. I considered only people who have rated more movies than 50.

4. REMOVING MOVIES WITHOUT ENOUGH RATINGS
   Movies with less than 10 ratings were removed. If a movie had less than 10 movies, it means a lot of people haven’t watched it. This system recommends 
   popular movies you might like.

5. REMOVING SPARSE DATA


