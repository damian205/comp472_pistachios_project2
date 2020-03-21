#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
df_of_train_tweets = None
with open('training-tweets.txt', encoding='utf8') as file:
    language_list = []
    content_list = []
    for line in file.readlines():
        try:
            tweet_id, user_name, language, tweet = line.split(maxsplit=3)
            language_list.append(language.strip())
            content_list.append(tweet.strip())
        except ValueError:
            pass
    data = {'Language': language_list, 'Content': content_list}
    df_of_train_tweets = pd.DataFrame.from_dict(data)        


# In[2]:


import string
def accepted_character(a_character, accepted_char_set):
    if a_character in accepted_char_set:
        return True
    else:
        return False


# In[3]:


class Language:
    def __init__(self, symbol, description, dataset, ngram=None, probability=None):
        self.symbol = symbol
        self.description = description
        self.dataset = dataset
        self.ngram = []
        self.probability = probability
        
    def add_ngram(self, ngram):
        self.ngram.append(ngram)


# In[4]:


language_symbol = {
    'eu': 'basque',
    'ca': 'catalan',
    'gl': 'galician',
    'es': 'spanish',
    'en': 'english',
    'pt': 'portugese'
}


# In[5]:


list_of_languages = []
total_nb_of_tweets = df_of_train_tweets.shape[0]
for i in language_symbol.keys():
    language_dataset = df_of_train_tweets[df_of_train_tweets['Language'] == i]
    prob_of_language = language_dataset.shape[0]/total_nb_of_tweets
    a_language = Language(i, language_symbol.get(i), language_dataset, None, prob_of_language)
    list_of_languages.append(a_language)


# In[6]:


list_of_letters = [character for character in string.ascii_letters] #readjust for the nb of vocabulary
unigram_dataset = pd.DataFrame(np.zeros((52,2)), columns=['Probability', 'Instances'])
unigram_dataset['Characters'] = list_of_letters
unigram_dataset = unigram_dataset.set_index('Characters')
# print(unigram_dataset)


# In[7]:


for language in list_of_languages: # is way too slow, need to verify results
    language.ngram.append(unigram_dataset.copy())
    for line in language.dataset['Content']:
        for character in line:
            if character in string.ascii_letters:
                language.ngram[0].loc[character]['Instances'] += 1


# In[8]:


for language in list_of_languages:
    total_instances = language.ngram[0]['Instances'].sum()
    for index, row in language.ngram[0].iterrows():
        row['Probability'] = row['Instances']/total_instances


# In[9]:


df_of_test_tweets = None
with open('test-tweets-given.txt', encoding='utf8') as file:
    language_list = []
    content_list = []
    for line in file.readlines():
        try:
            tweet_id, user_name, language, tweet = line.split(maxsplit=3)
            language_list.append(language.strip())
            content_list.append(tweet.strip())
        except ValueError:
            pass
    data = {'Language': language_list, 'Content': content_list}
    df_of_test_tweets = pd.DataFrame.from_dict(data)


# In[10]:


import math
def prob_of_language(list_of_languages, line):
    best_probability = float('-inf')
    best_language = None
    for language in list_of_languages:
        total_probability = math.log(language.probability)
        for character in line:
            if character in string.ascii_letters:
                character_probability = language.ngram[0].loc[character]['Probability']
                total_probability *= math.log(character_probability)
        if total_probability > best_probability:
            best_probability = total_probability
            best_language = language.symbol
    return best_language


# In[11]:


list_of_guesses = []
for index, row in df_of_test_tweets.iterrows():
    list_of_guesses.append(prob_of_language(list_of_languages, row['Content']))
df_of_test_tweets['guess'] = list_of_guesses


# In[12]:


df_of_test_tweets


# In[14]:


guess_status = []
for index, row in df_of_test_tweets.iterrows():
    if row['Language'] == row['guess']:
        guess_status.append(True)
    else:
        guess_status.append(False)
df_of_test_tweets['Status'] = guess_status


# In[15]:


df_of_test_tweets


# In[19]:


(df_of_test_tweets[df_of_test_tweets['Status'] == True].shape[0])/df_of_test_tweets.shape[0]


# In[13]:


list_of_languages[0].ngram[0].loc['a']['Probability']


# In[ ]:




