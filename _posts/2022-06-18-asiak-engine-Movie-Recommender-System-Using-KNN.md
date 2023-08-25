---
layout: post
title: 'The Asiak Movie Engine!'
date: '2022-06-18'
categories: machine learning
---

# INTRODUCTION
<p>I love movies. Most of the movies I watch are influenced by their ratings and reviews online. 
I decided to build a simple recommender system for movies deployed on streamlit based on the ratings of other users gathered in the Movie lens dataset from the University of Minnesota.
Let me show you how to do this but before that, a little about the KNN algorithm.</p>

# K-NEAREST NEIGHBOR
<p>
  The K-Nearest Neighbor is a non-parametric learning algorithm used for both classification and regression tasks. 
  In simpler terms, the KNN does not really know anything about its data. 
  The learning strategy of the K-Nearest Neighbor algorithm is to find the most similar observations to the one you have to predict assuming that similar things are near each other.<br>
  *Let’s put this in context through an analogy;*

  Once a new example or input x from previously unseen data comes in, the KNN algorithm finds the closest k-training examples to x and returns the most       frequent class among them in the case of classification or the average class in the case of regression.
</p>


