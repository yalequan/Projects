# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 16:34:30 2019

@author: Yale Quan
"""

#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
import numpy as np
import nltk
import os
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.tag import pos_tag
import matplotlib.pyplot as plt
import re
from nltk.stem.porter import PorterStemmer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from progressbar import *
pbar = ProgressBar()
from textblob import TextBlob
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
# Run only once
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('words')
#nltk.download('vader_lexicon')
#%%%
# Import the data usisng CSV

# yelp_user = pd.read_csv(r"C:/Users/Yale/Desktop/Data/yelp_user.csv")
yelp_business = pd.read_csv(r"C:/Users/Yale/Desktop/Data/yelp_business.csv")
# yelp_business_attributes = pd.read_csv(r"C:/Users/Yale/Desktop/Data/yelp_business_attributes.csv")
# yelp_business_hours = pd.read_csv(r"C:/Users/Yale/Desktop/Data/yelp_business_hours.csv")
yelp_checkin = pd.read_csv(r"C:/Users/Yale/Desktop/Data/yelp_checkin.csv")
yelp_review = pd.read_csv(r"C:/Users/Yale/Desktop/Data/yelp_review.csv")
# yelp_tip = pd.read_csv(r"C:/Users/Yale/Desktop/Data/yelp_tip.csv")

yelp_review.columns = ['review_id', 'user_id', 'business_id', 'stars', 'date', 'review_text','useful', 'funny', 'cool']
yelp_review['length'] = yelp_review['review_text'].apply(len)

#%%


# We begin the analysis by counting the missing values per dataset

# yelp_user.isnull().sum()
yelp_business.isnull().sum()
#yelp_business_attributes.isnull().sum()
# yelp_business_hours.isnull().sum()
# yelp_checkin.isnull().sum()
yelp_review.isnull().sum()
# yelp_tip.isnull().sum()


#We see that yelp_business.neighborhood and yelp_business.postal code have over 500 missing values
#Also, yelp_user.name is missing about 500 values.  

#We will drop these

del yelp_business['neighborhood']
del yelp_business['postal_code']


#%%%
#Now I want to subset by country.  I only want USA

state_dict = {"AL" : 'USA', "AK" : 'USA', "AZ" : 'USA', "AR" : 'USA', "CA" : 'USA', "CO" : 'USA', "CT" : 'USA', "DC" : 'USA', 
              "DE" : 'USA', "FL" : 'USA', "GA" : 'USA', "HI" : 'USA', "ID" : 'USA', "IL" : 'USA', "IN" : 'USA', "IA" : 'USA', 
              "KS" : 'USA', "KY" : 'USA', "LA" : 'USA', "ME" : 'USA', "MD" : 'USA', "MA" : 'USA', "MI" : 'USA', "MN" : 'USA', 
              "MS" : 'USA', "MO" : 'USA', "MT" : 'USA', "NE" : 'USA', "NV" : 'USA', "NH" : 'USA', "NJ" : 'USA', "NM" : 'USA', 
              "NY" : 'USA', "NC" : 'USA', "ND" : 'USA', "OH" : 'USA', "OK" : 'USA', "OR" : 'USA', "PA" : 'USA', "RI" : 'USA', 
              "SC" : 'USA', "SD" : 'USA', "TN" : 'USA', "TX" : 'USA', "UT" : 'USA', "VT" : 'USA', "VA" : 'USA', "WA" : 'USA', 
              "WV" : 'USA', "WI" : 'USA', "WY" : 'USA'}

yelp_business.dtypes

yelp_business['Country'] = yelp_business['state'].map(state_dict)

yelp_business['Country'].isnull().sum()

yelp_business_USA = yelp_business.dropna(subset = ['Country'])

yelp_business_USA.head(10)

# Export to Excel for Tableau

yelp_business_USA.to_csv (r'C:\Users\Yale\Desktop\Data\Yelp_USA.csv', index = None, header = True)

#cleanup

del yelp_business
del state_dict

#%%%
'''
Now we want to see the types of business in the yelp dataset
'''

# Create Business Categories

business_categories =','.join(yelp_business_USA['categories'])
categories = pd.DataFrame(business_categories.split(';'),columns=['category'])

#Count the categories
category_count = categories.category.value_counts()

df_category = pd.DataFrame(category_count)
df_category.reset_index(inplace=True)

# Plot the data and see the top 10 categories

plt.figure(figsize=(12,10))
f = sns.barplot( y= 'index',x = 'category' , data = df_category.iloc[0:10])
f.set_ylabel('Category')
f.set_xlabel('Number of businesses');

# Export to Excel for Tableau

categories.to_csv (r'C:\Users\Yale\Desktop\Data\Categories.csv', index = None, header = True)

del categories 
del business_categories
del category_count
del df_category

#%%%
'''Subset by resturaunt...'''

USA_Restaurants = yelp_business_USA[yelp_business_USA['categories'].str.contains("Restaurant")]

# Cleanup
del yelp_business_USA

#%%%

'''
Visualize the star ratings in the resturaunt dataset
'''
USA_Restaurants['stars'].nunique()
USA_Restaurants['stars'].unique()
df = pd.DataFrame(USA_Restaurants['stars'].value_counts())
df.sort_index()

sns.countplot(x = 'stars', data = USA_Restaurants)


'''
There are 9 unique star ratings
[4. , 3. , 1.5, 3.5, 5. , 4.5, 2. , 2.5, 1. ]

     stars
1.0   3788
1.5   4303
2.0   9320
2.5  16148
3.0  23142
3.5  32038
4.0  33492
4.5  24796
5.0  27540

From the plot and count we can see that 3.5 and 4.0 star ratings are the most common.  From the graph
I plan on binning the ratings as Low: 1.0 - 2.5  Mid: 3.0 - 4.0, and High: 4.5-5.0
'''

# Cleanup

df  = None


#%%%

# PLot bins

USA_Restaurants['rating_bin'] = pd.cut(USA_Restaurants['stars'], bins = [0, 2.5, 4.0, 5.0])

USA_Restaurants[['stars', 'rating_bin']].head(10)

USA_Restaurants['star_rating'] = pd.cut(USA_Restaurants['stars'], bins = [0, 2.5, 4.0, 5.0], labels = ['low', 'med', 'high'])

USA_Restaurants[['stars', 'rating_bin', 'star_rating']].head(10)

sns.countplot(x = 'star_rating', data = USA_Restaurants)

'''
Again we can see that the majority of our business recieve a medium review 3.0-4.0
'''

#%%%

# Stars of open and closed.  Find Mean and count

USA_Restaurants_Open = USA_Restaurants[USA_Restaurants['is_open'] == 1]
USA_Restaurants_Closed = USA_Restaurants[USA_Restaurants['is_open'] == 0]

USA_Restaurants_Open['stars'].mean()
USA_Restaurants_Closed['stars'].mean()

df_open = pd.DataFrame(USA_Restaurants_Open['stars'].value_counts())
df_closed = pd.DataFrame(USA_Restaurants_Closed['stars'].value_counts())

df_open.sort_index()
df_closed.sort_index()

#%%%

'''Look at the correaltion between open/closed and the star ratings.'''

df = USA_Restaurants[['stars', 'is_open']]

df.corr(method = 'pearson')


def spearmans_rank_correlation(xs, ys):
    
    # Calculate the rank of x's
    xranks = pd.Series(xs).rank()
    
    # Caclulate the ranking of the y's
    yranks = pd.Series(ys).rank()
    
    # Calculate Pearson's correlation coefficient on the ranked versions of the data
    return scipy.stats.pearsonr(xranks, yranks)

spearmans_rank_correlation(USA_Restaurants['stars'], USA_Restaurants['is_open'])

scipy.stats.pointbiserialr(USA_Restaurants['is_open'],USA_Restaurants['stars'] )


'''There is a very low correlation between stars and the review count'''

#%%

# visualize the distibution of reviews

USA_Restaurants['review_count'].describe()
USA_Restaurants['review_count'].median()

sns.distplot(USA_Restaurants['review_count'], hist=False, kde=True, bins=1, color = 'darkblue',hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4}) 

sns.distplot(USA_Restaurants['review_count'].apply(np.log1p));

# Dist by state

sns.countplot(data = USA_Restaurants, x = 'state')

USA_Restaurants['state'].value_counts()


#%%%

# Merge review and Resturaunt Datasets
full_data_1 = pd.merge(USA_Restaurants, yelp_review, on = 'business_id', how = 'inner')

# Subset into NV observations

full_data = full_data_1[full_data_1['state'].str.contains("AZ")]

del USA_Restaurants
del yelp_review
#%%%

# Basic analysis of reviews

full_data['length'].plot(bins = 50, kind = 'hist')

full_data['length'].describe().apply(lambda x: format(x, 'f'))

full_data.hist(column = 'length', by = 'is_open', bins = 10, figsize = (10,4))




#%%%

# Correlation of review length

df = full_data[['length', 'is_open']]

df.corr(method = 'pearson')

scipy.stats.pointbiserialr(full_data['is_open'],full_data['length'] )

#%%%

# Analysis between user stars and open and closed status

x = full_data['stars_y'].value_counts()
x=x.sort_index()
plt.figure(figsize=(10,6))
ax= sns.barplot(x.index, x.values, alpha=0.8)
plt.title("Star Rating Distribution in USA Restaurants")
plt.ylabel('count')
plt.xlabel('Star Ratings')
rects = ax.patches
labels = x.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')
plt.show()

full_data.hist(column = 'stars_y', by = 'is_open', bins = 50, figsize = (10,4))

#Correlation

df = full_data[['stars_y', 'is_open']]

df.corr(method = 'pearson')

scipy.stats.pointbiserialr(full_data['is_open'],full_data['stars_y'] )


#%%%

# Split the data into open and closed dataframes

open_business = full_data[full_data['is_open'] == 1]

closed_business = full_data[full_data['is_open'] == 0]

open_business.hist(column = 'length')
closed_business.hist(column = 'length')

open_business['length'].mean()
closed_business['length'].mean()


#%%

# Sentiment Analysis.

sid = SentimentIntensityAnalyzer()  # Perform sentiment intensity analyzation with VADER

###########################################################################

# Open Business

vader_open_scores = []

#  1) Create List of Polarity Scores.
#     Note: VADER Produces a list of lists

for x in open_business['review_text']:
    polarity = sid.polarity_scores(x) # Calculate the polatiry of the review
    vader_open_scores.append(polarity)
    
vader_sentiment_open = pd.DataFrame(vader_open_scores)
vader_sentiment_open.columns = ['Vader_Compound_Score', 'Vader_Negative_Score', 'Vader_Neutral_Score', 'Vader_Positive_Score']
    
#  2 Create list of VADER Classification  The VADER documentation recommeds
    # a threshold of +- 0.5
    
vader_classification_open = []

threshold = 0.5
for x in vader_sentiment_open['Vader_Compound_Score']:
    if x > threshold:
        vader_classification_open.append('positive')
    elif x < -threshold:
        vader_classification_open.append('negative')
    else:
        vader_classification_open.append('neutral')
        
vader_sentiment_open['Vader_Classification'] = vader_classification_open


# 3) Count the Positive, Neg, and Neutural Reviews. 

VADER_open_Counts = {"positive":0, "neutral":0, "negative":0}  # initialize dict for counting
for x in vader_sentiment_open['Vader_Classification']:
    if x == 'neutral': 
        VADER_open_Counts["neutral"] +=1 # Count the neutural review
    elif x == 'positive':
        VADER_open_Counts["positive"] +=1 # Count the positive
    else:
        VADER_open_Counts['negative'] +=1 # Count the negative
print(VADER_open_Counts)

VADER_open_Counts_df = pd.DataFrame(VADER_open_Counts, index=[0])

#4) Add to open data

open_business.reset_index(drop=True, inplace=True)
vader_sentiment_open.reset_index(drop=True, inplace=True)

open_business = pd.concat([open_business, vader_sentiment_open], axis=1)


# Visualize Histogram

objects = ['Neutral', 'Positive', 'Negative']
y_pos = np.arange(len(objects))
amount = [VADER_open_Counts.get('neutral'), VADER_open_Counts.get('positive'), VADER_open_Counts.get('negative')]

plt.bar(y_pos, amount, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Number of reviews')
plt.title('VADER Sentiment Analysis of Open Resturaunts')

# create piechart

labels = ['Neutral', 'Positive', 'Negative']
sizes = [VADER_open_Counts.get('neutral'), VADER_open_Counts.get('positive'), VADER_open_Counts.get('negative')]
fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct='%1.1f%%')
ax.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
ax.set_title('Open Resturaunts')

###########################################################################

# Closed Analysis

vader_closed_scores = []

#  1) Create List of Polarity Scores.
#     Note: VADER Produces a list of lists

for x in closed_business['review_text']:
    polarity = sid.polarity_scores(x) # Calculate the polatiry of the review
    vader_closed_scores.append(polarity)
    
vader_sentiment_closed = pd.DataFrame(vader_closed_scores)
vader_sentiment_closed.columns = ['Vader_Compound_Score', 'Vader_Negative_Score', 'Vader_Neutral_Score', 'Vader_Positive_Score']
    
#  2) Create list of VADER Classification
    
vader_classification_closed = []

threshold = 0.5
for x in vader_sentiment_closed['Vader_Compound_Score']:
    if x > threshold:
        vader_classification_closed.append('positive')
    elif x < -threshold:
        vader_classification_closed.append('negative')
    else:
        vader_classification_closed.append('neutral')
        
vader_sentiment_closed['Vader_Classification'] = vader_classification_closed
    
# 3) Count the Positive, Neg, and Neutural Reviews.  The VADER documentation recommeds
    # a threshold of +- 0.5

VADER_closed_Counts = {"positive":0, "neutral":0, "negative":0}  # initialize dict for counting
for x in vader_sentiment_closed['Vader_Classification']:
    if x == 'neutral': 
        VADER_closed_Counts["neutral"] +=1 # Count the neutural review
    elif x == 'positive':
        VADER_closed_Counts["positive"] +=1 # Count the positive
    else:
        VADER_closed_Counts['negative'] +=1 # Count the negative
print(VADER_closed_Counts)

VADER_closed_Counts_df = pd.DataFrame(VADER_closed_Counts, index=[0])

#4) Add to closed data

closed_business.reset_index(drop=True, inplace=True)
vader_sentiment_closed.reset_index(drop=True, inplace=True)

closed_business = pd.concat([closed_business, vader_sentiment_closed], axis=1)


# Visualize Histogram of Positive, Negative, and Neutural Reviews

objects = ['Neutral', 'Positive', 'Negative']
y_pos = np.arange(len(objects))
amount = [VADER_closed_Counts.get('neutral'), VADER_closed_Counts.get('positive'), VADER_closed_Counts.get('negative')]

plt.bar(y_pos, amount, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Number of reviews')
plt.title('VADER Sentiment Analysis of Closed Resturaunts')

# create piechart of Positive, Negative, and Neutural Reviews

labels = ['Neutral', 'Positive', 'Negative']
sizes = [VADER_closed_Counts.get('neutral'), VADER_closed_Counts.get('positive'), VADER_closed_Counts.get('negative')]
fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct='%1.1f%%')
ax.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
ax.set_title('Closed Resturaunts')






#%%%

# TextBlob analysis of Open Busienesses

TextBlob_Analysis_Open = pd.DataFrame(columns = ['TextBlob'])

TextBlob_Analysis_Open['TextBlob'] = open_business['review_text'].apply(TextBlob)  # create a textblob of the review

length = TextBlob_Analysis_Open.TextBlob.count()

sentiment_open = []

pbar = ProgressBar(max_value = length)

for sentence in pbar(TextBlob_Analysis_Open['TextBlob']):
    polarity = sentence.sentiment.polarity # Calculate the polatiry of the review
    sentiment_open.append(polarity)

TextBlob_Analysis_Open['TextBlob_Polarity'] = sentiment_open

textblob_sentiment = []

for s in sentiment_open['Text_Blob_Polarity']:
    if s > 0:
        textblob_sentiment.append('positive')
    elif s < 0:
        textblob_sentiment.append('negative')
    else:
        textblob_sentiment.append('neutral')
    
TextBlob_Analysis_Open['Text_Blob_Sentiment'] = textblob_sentiment  

del TextBlob_Analysis_Open['TextBlob']

#%%%

# TextBlob analysis of Closed Busienesses

TextBlob_Analysis_Closed = pd.DataFrame(columns = ['TextBlob'])

TextBlob_Analysis_Closed['TextBlob'] = closed_business['review_text'].apply(TextBlob)  # create a textblob of the review

length = TextBlob_Analysis_Closed.TextBlob.count()

sentiment_closed = []

pbar = ProgressBar(max_value = length)

for sentence in TextBlob_Analysis_Closed['TextBlob']:
    polarity = sentence.sentiment.polarity # Calculate the polatiry of the review
    sentiment_closed.append(polarity)

TextBlob_Analysis_Closed['TextBlob_Polarity'] = sentiment_closed

textblob_sentiment = []

for s in sentiment_closed:
    if s > 0:
        textblob_sentiment.append('positive')
    elif s < 0:
        textblob_sentiment.append('negative')
    else:
        textblob_sentiment.append('neutral')
    
TextBlob_Analysis_Closed['Text_Blob_Sentiment'] = textblob_sentiment  

del TextBlob_Analysis_Closed['TextBlob']

#######################################################################

# Next count the amount of positive, neg, neutural reviews

Text_Blob_Sentiment_Counts_Open = {"positive":0, "neutral":0, "negative":0}  # initialize dict for counting
for x in TextBlob_Analysis_Open['Text_Blob_Sentiment']:
    if x == 'neutral': 
        Text_Blob_Sentiment_Counts_Open["neutral"] +=1 # Count the neutural review
    elif x == 'positive':
        Text_Blob_Sentiment_Counts_Open["positive"] +=1 # Count the positive
    else:
        Text_Blob_Sentiment_Counts_Open['negative'] +=1 # Count the negative
print(Text_Blob_Sentiment_Counts_Open)

Text_Blob_Sentiment_Counts_Closed = {"positive":0, "neutral":0, "negative":0}  # initialize dict for counting
for x in TextBlob_Analysis_Closed['Text_Blob_Sentiment']:
    if x == 'neutral': 
        Text_Blob_Sentiment_Counts_Closed["neutral"] +=1 # Count the neutural review
    elif x == 'positive':
        Text_Blob_Sentiment_Counts_Closed["positive"] +=1 # Count the positive
    else:
        Text_Blob_Sentiment_Counts_Closed['negative'] +=1 # Count the negative
print(Text_Blob_Sentiment_Counts_Closed)

# Visualize Open Histogram

objects = ['Neutral', 'Positive', 'Negative']
y_pos = np.arange(len(objects))
amount = [Text_Blob_Sentiment_Counts_Open.get('neutral'), Text_Blob_Sentiment_Counts_Open.get('positive'), Text_Blob_Sentiment_Counts_Open.get('negative')]

plt.bar(y_pos, amount, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Number of reviews')
plt.title('TextBlob Sentiment Anaysis of Open Resturaunts')

# create open piechart

labels = ['Neutral', 'Positive', 'Negative']
sizes = [Text_Blob_Sentiment_Counts_Open.get('neutral'), Text_Blob_Sentiment_Counts_Open.get('positive'), Text_Blob_Sentiment_Counts_Open.get('negative')]
fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct='%1.1f%%')
ax.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
ax.set_title('TextBlob Sentiment Anaysis of Open Resturaunts')

# Visualize Closed Histogram

objects = ['Neutral', 'Positive', 'Negative']
y_pos = np.arange(len(objects))
amount = [Text_Blob_Sentiment_Counts_Closed.get('neutral'), Text_Blob_Sentiment_Counts_Closed.get('positive'), Text_Blob_Sentiment_Counts_Closed.get('negative')]

plt.bar(y_pos, amount, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Number of reviews')
plt.title('TextBlob Sentiment Anaysis of Closed Resturaunts')

# create closed piechart

labels = ['Neutral', 'Positive', 'Negative']
sizes = [Text_Blob_Sentiment_Counts_Closed.get('neutral'), Text_Blob_Sentiment_Counts_Closed.get('positive'), Text_Blob_Sentiment_Counts_Closed.get('negative')]
fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct='%1.1f%%')
ax.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
ax.set_title('TextBlob Sentiment Anaysis of Closed Resturaunts')

##########################################################


# Combine open and closed datasets back together

complete_data = pd.concat([open_business, closed_business], axis = 0)

# We begin by dropping variables we don't want to consider for analysis

del complete_data['Vader_Negative_Score'] 
del complete_data['Vader_Neutral_Score'] 
del complete_data['Vader_Positive_Score']
del complete_data['business_id']
del complete_data['name']
del complete_data['address']
del complete_data['Country']
del complete_data['review_id']
del complete_data['user_id']
del complete_data['date']
del complete_data['review_text']
del complete_data['categories']
del complete_data['state']

# Need to rename some cities

complete_data['city'].replace('Phx', 'Phoenix', inplace=True)
complete_data['city'].replace('Pheonix AZ', 'Phoenix', inplace=True)
complete_data['city'].replace('Glendale Az', 'Glendale', inplace=True)
complete_data['city'].replace('Gelndale', 'Glendale', inplace=True)
complete_data['city'].replace('MESA', 'Mesa', inplace=True)
complete_data['city'].replace('Mesa AZ', 'Mesa', inplace=True)
complete_data['city'].replace('Schottsdale', 'Scottsdale', inplace=True)
complete_data['city'].replace('Scottdale', 'Scottsdale', inplace=True)


# We first need to one-hot encode for dummy variables.  WARNING: USES LOTS OF RAM

transformed_complete_data = pd.get_dummies(complete_data)

# Create full correlation matrix

corr_matrix = transformed_complete_data.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

transformed_complete_data.drop(df[to_drop], axis=1)

def multi_collinearity_heatmap(df, figsize=(11,9)):
    
    """
    Creates a heatmap of correlations between features in the df. A figure size can optionally be set.
    """
    
    # Set the style of the visualization
    sns.set(style="white")

    # Create a covariance matrix
    corr = df.corr()

    # Generate a mask the size of our covariance matrix
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=figsize)

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5}, vmax=corr[corr != 1.0].max().max());


multi_collinearity_heatmap(transformed_complete_data, figsize=(30,30))

######################

# Random Forest Classifier

X = transformed_complete_data.loc[:, transformed_complete_data.columns != 'is_open'] # dependent
y = transformed_complete_data.loc[:, transformed_complete_data.columns == 'is_open'] # target variable

os = SMOTE(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0) # Train and test
columns = X_train.columns

os_data_X,os_data_y = os.fit_sample(X_train, y_train.values.ravel())  # SMOTE for overfitting
os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
os_data_y= pd.DataFrame(data=os_data_y,columns=['y'])
# we can Check the numbers of our data
print("length of oversampled data is ",len(os_data_X))
print("Number of no subscription in oversampled data",len(os_data_y[os_data_y['y']==0]))
print("Number of subscription",len(os_data_y[os_data_y['y']==1]))
print("Proportion of no subscription data in oversampled data is ",len(os_data_y[os_data_y['y']==0])/len(os_data_X))
print("Proportion of subscription data in oversampled data is ",len(os_data_y[os_data_y['y']==1])/len(os_data_X))


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(os_data_X, os_data_y, test_size = 0.3, random_state = 0)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state = 0)
model.fit(X_train, y_train.values.ravel())

y_pred = model.predict(X_test)

y_actu = pd.Series(y_test.values.ravel(), name='Actual')
y_pred = pd.Series(y_pred, name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred)

df_confusion

TP = 161307
FP = 4168
FN = 2331
TN = 163472



df_conf_norm = df_confusion / df_confusion.sum(axis=1)

df_conf_norm


#######################

from sklearn.linear_model import SGDClassifier
sg = SGDClassifier(random_state=42)
sg.fit(X_train,y_train.values.ravel())
pred = sg.predict(X_test)
from sklearn.metrics import classification_report,accuracy_score
print(classification_report(y_test, pred))
print(accuracy_score(y_test, pred))

#parameter tuning 
from sklearn.model_selection import GridSearchCV
#model
model = SGDClassifier(random_state=42)
#parameters
params = {'loss': ["hinge", "log", "perceptron"],
          'alpha':[0.001, 0.0001, 0.00001]}
#carrying out grid search
clf = GridSearchCV(model, params)
clf.fit(X_train, y_train.values.ravel())
#the selected parameters by grid search
print(clf.best_estimator_)

#final model by taking suitable parameters  Alpha = 0.00001
clf = SGDClassifier(random_state=42, loss="hinge", alpha=0.00001)
clf.fit(X_train, y_train.values.ravel())
pred = clf.predict(X_test)

print(classification_report(y_test, pred))
print(accuracy_score(y_test, pred))

# bad.  Now try alpba = 0.0001
clf = SGDClassifier(random_state=42, loss="hinge", alpha=0.0001)
clf.fit(X_train, y_train.values.ravel())
pred = clf.predict(X_test)

print(classification_report(y_test, pred))
print(accuracy_score(y_test, pred))

#   Now try alpba = 0.001
clf = SGDClassifier(random_state=42, loss="hinge", alpha=0.001)
clf.fit(X_train, y_train.values.ravel())
pred = clf.predict(X_test)

print(classification_report(y_test, pred))
print(accuracy_score(y_test, pred))

# Suspicious Reviews

complete_data['suspicious_pos'] = np.where((complete_data['stars_y'] <= 2) & (complete_data['Vader_Classification'] == 'positive'), 'true', 'false')


complete_data['suspicious_neg'] = np.where((complete_data['stars_y'] >= 4) & (complete_data['Vader_Classification'] == 'negative'), 'true', 'false')

