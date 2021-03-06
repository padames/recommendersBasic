# Basic Recommenders
This is publicly available development code used to learn and test concepts and basic ideas about statistical learning in recommender systems.

# Temporary repo to store more than just code
Storing freely available pdf books and some papers for personal study obtained through the UofC or freely available

# Code tool versions

Using github and git:
* git >= 2.17

When using python:
* python version >=3.6.9 from anaconda
* editor Wing Pro 7 >=7.2.2.8

When using R:
* r version >=3.6.3
* RStudio version >=1.2.5033


Policies: 
1. commit projects to facilitate communication and reduce maitenance

## Initial outlook

Start with hello world type of examples like the Movie Lens data set to formulate content-based recommendations since there are users with movie selection history that can be given recommendations based precisely on their previous choices. In this process define the utility matrix to compute similarity of previous choices with available ones. Test that you can easily update them when new choices become available. 

Follow with two cases: an item-based collaborative filter and a user-based collaborative filter.

Then test cold-start with a content-based recommender using association rules. 

Then build a hybrid recommender using both strategies.

### MovieLens examples
There are two sources for this basic _hello world_ type of example.

The first one can be seen in the **src** folder, a python Jupyter notebook called `ch9_recommendation_system.ipynb` from the book _Introduction to Data Science_ by Laura Igual and Santi Segui, Springer 2017.


