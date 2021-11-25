#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#A social media sentiment analysis tells you how people feel about your brand online. Rather than a simple count of mentions or comments, sentiment analysis considers emotions and opinions. It involves collecting and analyzing information in the posts people share about your brand on social media.
#Sentiment analysis involves natural language processing because it deals with human-written text. The ability to categorize opinions expressed in the text —and especially to determine whether the writer's attitude is positive, negative, or neutral—is highly valuable. In this project, we will use the process known as sentiment analysis to categorize the opinions of people.


# In[2]:


import pandas as pd                              #data science/data analysis
import numpy as np                               #numerical python
import re                                        #matches string
import nltk                                      #natural language toolkit
import matplotlib.pyplot as plt                  #data visualisation 
import seaborn as sns                            #data visualisation 
from nltk.corpus import stopwords                #common used words

# ML Libraries                                                      #tools for machine learning
from sklearn.feature_extraction.text import TfidfVectorizer         #compute word counts using count vectorizer
from sklearn.model_selection import train_test_split                #splits model into train and test
from sklearn.linear_model import LogisticRegression                 #relation btw data


# In[3]:


train = pd.read_csv(r'E:\mini project 21\train_tweet.csv')
test = pd.read_csv(r'E:\mini project 21\test_tweets.csv')

print(train.shape)
print(test.shape)


# In[4]:


train.head()


# In[5]:


train.isnull().any()
test.isnull().any()


# In[6]:


#checking out the POSITIVE comments from the train set

train[train['label'] == 0].head(30)


# In[7]:


# checking out the NEGATIVE comments from the train set 

train[train['label'] == 1].head(30)


# In[8]:


train['label'].value_counts().plot.bar(color = 'blue', figsize = (6, 4))


# In[9]:


# checking the distribution of tweets in the data
length_test = test['tweet'].str.len().plot.hist(color = 'red', figsize = (6, 4))


# In[10]:


#checking for train.csv
length_train = train['tweet'].str.len().plot.hist(color = 'orange', figsize = (6, 4))


# In[11]:


train['len'] = train['tweet'].str.len()
test['len'] = test['tweet'].str.len()

# create a function to compute the negative,positive analysis

def getAnalysis(score):
   
    if score ==0:
        return 'positive'
    else:
        return 'negative'
    
train['Analysis']=train['label'].apply(getAnalysis)    

train.head(20)


# In[12]:


train.groupby('label').describe()


# In[13]:


train.groupby('len').mean()['label'].plot.bar(color = 'blue', figsize = (40,16))
plt.title('variation of length')
plt.xlabel('Length')
plt.show()


# In[14]:


#finding most frequent words
from sklearn.feature_extraction.text import CountVectorizer                         #converts text into vector form


cv = CountVectorizer(stop_words = 'english')
words = cv.fit_transform(train.tweet)

sum_words = words.sum(axis=0)

words_freq = [(word, sum_words[0, i]) for word, i in cv.vocabulary_.items()]
words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)

frequency = pd.DataFrame(words_freq, columns=['word', 'freq'])

frequency.head(30).plot(x='word', y='freq', kind='bar', figsize=(15, 7), color = 'red')
plt.title("Most Frequently Occuring Words - Top 30")


# In[15]:


# collecting the hashtags

def hashtag_extract(x):
    hashtags = []
    
    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)

    return hashtags


# In[16]:


# extracting hashtags from positive or neutral tweets
HT_regular = hashtag_extract(train['tweet'][train['label'] == 0])

# extracting hashtags from negative tweets
HT_negative = hashtag_extract(train['tweet'][train['label'] == 1])

# unnesting list
HT_regular = sum(HT_regular,[])
HT_negative = sum(HT_negative,[])


# In[17]:


a = nltk.FreqDist(HT_regular)
d = pd.DataFrame({'Hashtag': list(a.keys()),
                  'Count': list(a.values())})

# selecting top 10 most frequent hashtags     
d = d.nlargest(columns="Count", n = 10) 
plt.figure(figsize=(16,5))
ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.show()


# In[18]:


a = nltk.FreqDist(HT_negative)
d = pd.DataFrame({'Hashtag': list(a.keys()),
                  'Count': list(a.values())})

# selecting top 10 most frequent hashtags     
d = d.nlargest(columns="Count", n = 10) 
plt.figure(figsize=(16,5))
ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.show()


# In[19]:


# tokenizing the words present in the training set
tokenized_tweet = train['tweet'].apply(lambda x: x.split()) 

# importing gensim
import gensim               #to handle large text collections

# creating a word to vector model
model_w2v = gensim.models.Word2Vec(
            tokenized_tweet, # desired no. of features/independent variables 
            window=5, # context window size
            min_count=2,
            sg = 1, # 1 for skip-gram model
            hs = 0,
            negative = 10, # for negative sampling
            workers= 2, # no.of cores
            seed = 34)

model_w2v.train(tokenized_tweet, total_examples= len(train['tweet']), epochs=20)


# In[20]:


model_w2v.wv.most_similar(positive = "dinner")


# In[21]:


model_w2v.wv.most_similar(positive = "apple")


# In[22]:


model_w2v.wv.most_similar(negative = "hate")


# In[23]:


from tqdm import tqdm               #to add progress bar 
tqdm.pandas(desc="progress-bar")
from gensim.models.doc2vec import TaggedDocument        #Replaces "sentence as a list of words" 


# In[24]:


def add_label(twt):
    output = []
    for i, s in zip(twt.index, twt):
        output.append(TaggedDocument(s, ["tweet_" + str(i)]))
    return output

# label all the tweets
labeled_tweets = add_label(tokenized_tweet)

labeled_tweets[:6]


# In[25]:


# removing unwanted patterns from the data

import re
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# In[26]:


train_corpus = []

for i in range(0, 31962):
  review = re.sub('[^a-zA-Z]', ' ', train['tweet'][i])
  review = review.lower()
  review = review.split()
  
  ps = PorterStemmer()
  
  # stemming
  review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
  
  # joining them back with space
  review = ' '.join(review)
  train_corpus.append(review)


# In[27]:


test_corpus = []

for i in range(0, 17197):
  review = re.sub('[^a-zA-Z]', ' ', test['tweet'][i])
  review = review.lower()
  review = review.split()
  
  ps = PorterStemmer()
  
  # stemming
  review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
  
  # joining them back with space
  review = ' '.join(review)
  test_corpus.append(review)


# In[28]:


# creating bag of words

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features = 2500)                       #transform the data by removing mean of each feature then scale it
x = cv.fit_transform(train_corpus).toarray()
y = train.iloc[:, 1]

print(x.shape)
print(y.shape)


# In[29]:


# creating bag of words

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features = 2500)
x_test = cv.fit_transform(test_corpus).toarray()

print(x_test.shape)


# In[30]:


# splitting the training data into train and valid sets

from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.25, random_state = 42)

print(x_train.shape)
print(x_valid.shape)
print(y_train.shape)
print(y_valid.shape)


# In[31]:


# standardization

from sklearn.preprocessing import StandardScaler                    #transforms data such as, mean value 0 and standard deviation of 1

sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_valid = sc.transform(x_valid)
x_test = sc.transform(x_test)


# In[32]:


#model accuracy
#random forest classifier builds multiple decision trees and merges them together to get a more accurate and stable prediction
from sklearn.ensemble import RandomForestClassifier 
#A confusion matrix is a technique for summarizing the performance of a classification algorithm.
from sklearn.metrics import confusion_matrix
# F1-score, is a measure of a model's accuracy on a dataset
from sklearn.metrics import f1_score

model = RandomForestClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_valid)

print("Training Accuracy :", model.score(x_train, y_train))
print("Validation Accuracy :", model.score(x_valid, y_valid))

# calculating the f1 score for the validation set
print("F1 score :", f1_score(y_valid, y_pred))

# confusion matrix
cm = confusion_matrix(y_valid, y_pred)
print(cm)


# In[33]:


#Logistic regression is used to describe data and to explain the relationship between one dependent binary variable and one or 
#more nominal, ordinal, interval or ratio-level independent variables.
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=10000)
model.fit(x_train, y_train)

y_pred = model.predict(x_valid)

print("Training Accuracy :", model.score(x_train, y_train))
print("Validation Accuracy :", model.score(x_valid, y_valid))

# calculating the f1 score for the validation set
print("f1 score :", f1_score(y_valid, y_pred))

# confusion matrix
cm = confusion_matrix(y_valid, y_pred)
print(cm)


# In[34]:


from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_valid)

print("Training Accuracy :", model.score(x_train, y_train))
print("Validation Accuracy :", model.score(x_valid, y_valid))

# calculating the f1 score for the validation set
print("f1 score :", f1_score(y_valid, y_pred))

# confusion matrix
cm = confusion_matrix(y_valid, y_pred)
print(cm)


# In[35]:


X=train['tweet']
tfidf=TfidfVectorizer(max_features=10000, ngram_range=(1,2))
X=tfidf.fit_transform(X)
y=train['label']


# In[36]:


X.shape


# In[37]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0)


# In[38]:


X_train.shape, X_test.shape
