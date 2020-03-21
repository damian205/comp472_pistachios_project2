import pandas as pd
import numpy as np
import string
import math
from Language import *

#store training dataset
df_of_train_tweets = None
#store list of language models
list_of_languages = []

language_symbol = {
    'eu': 'basque',
    'ca': 'catalan',
    'gl': 'galician',
    'es': 'spanish',
    'en': 'english',
    'pt': 'portugese'
}

n_gram = {
    'unigram':1,
    'bigram' : 2,
    'trigram':3
}

vocabulary_lowercase = string.ascii_lowercase
vocabulary_all_case = string.ascii_letters

def main():
    #create training dataset
    training_tweets = readTweetsFromFile('training-tweets.txt')
    global df_of_train_tweets 
    df_of_train_tweets = pd.DataFrame.from_dict(training_tweets) 
    #create model for each language, based on training set
    createLanguageModel(vocabulary_all_case)
    #build n-grams for each language
    buidNgramModel(vocabulary_all_case, n_gram['bigram'])
    #score new tweets
    #scoreNewTweets('test-tweets-given.txt')
    
def scoreNewTweets(filename):
    test_tweets = readTweetsFromFile(filename)
    df_of_test_tweets = pd.DataFrame.from_dict(test_tweets) 
    list_of_guesses = []
    for index, row in df_of_test_tweets.iterrows():
        list_of_guesses.append(prob_of_language(row['Content']))
    df_of_test_tweets['guess'] = list_of_guesses
    
    guess_status = []
    for index, row in df_of_test_tweets.iterrows():
        if row['Language'] == row['guess']:
            guess_status.append(True)
        else:
            guess_status.append(False)
    df_of_test_tweets['Status'] = guess_status

#to-do this is super slow!
def prob_of_language(line):
    best_probability = float('-inf')
    best_language = None
    for language in list_of_languages:
        total_probability = math.log(language.probability)
        for character in line:
            if character in language.vocabulary:
                character_probability = language.unigram.loc[character]['Probability']
                total_probability *= math.log(character_probability)
        if total_probability > best_probability:
            best_probability = total_probability
            best_language = language.symbol
    return best_language

def readTweetsFromFile(fileName):
    with open(fileName, encoding='utf8') as file:
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
        return data

def createLanguageModel(vocabulary):
    global list_of_languages
    total_nb_of_tweets = df_of_train_tweets.shape[0]
    for i in language_symbol.keys():
        language_dataset = df_of_train_tweets[df_of_train_tweets['Language'] == i]
        prob_of_language = language_dataset.shape[0]/total_nb_of_tweets
        a_language = Language(i, language_symbol.get(i), language_dataset, vocabulary, prob_of_language)
        list_of_languages.append(a_language)

def buidNgramModel(vocabulary, n_gram):

    n_gram_dataset = createDataset(vocabulary, n_gram)
    
    #create n-gram for each language
    global list_of_languages
    for language in list_of_languages:
        if n_gram == 1:
            #calculate instances of each character
            language.unigram = n_gram_dataset.copy()
            list_of_letters = [character for character in language.vocabulary]
            for character in list_of_letters:
                language.unigram.loc[character]['Instances'] = ''.join(language.dataset['Content']).count(character)
            #calculate probablility of each character
            total_instances = language.unigram['Instances'].sum()
            for index, row in language.unigram.iterrows():
                row['Probability'] = row['Instances']/total_instances
        elif n_gram == 2:
            #populate characters
            language.bigram = n_gram_dataset.copy()
            language_as_String = ''.join(language.dataset['Content'])
            list_of_letters_x = [character for character in language.vocabulary]
            list_of_letters_y = [character for character in language.vocabulary]
            for x in list_of_letters_x:
                for y in list_of_letters_y:
                    language.bigram.loc[x, y] = language_as_String.count(x+y)
                language.bigram.loc[x,'instances'] = language.bigram.loc[x].sum()
            #calculate probablility of each character
            #to do need to calculate sum first
            for index, row in language.bigram.iterrows():
                row['Probability'] = row['instances']/total_instances


def createDataset(vocabulary, n_gram):
    #depending on the vocabulary and size of n_gram, create Dataframe with all letters
    list_of_letters = [character for character in vocabulary] 
    n_gram_dataset = None
    if n_gram == 1:
        dataset = pd.DataFrame(np.zeros((len(vocabulary),2)), columns=['Probability', 'Instances'])
        dataset['Characters'] = list_of_letters
        n_gram_dataset = dataset.set_index('Characters')
    elif n_gram == 2:
        new_list_of_letters = list_of_letters.copy()
        new_list_of_letters.append('instances')
        n_gram_dataset = pd.DataFrame(np.zeros((len(vocabulary),len(vocabulary)+1)), columns=new_list_of_letters, index=list_of_letters)
    return n_gram_dataset

main()