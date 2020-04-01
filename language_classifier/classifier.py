import pandas as pd
import numpy as np
import string
import math
from Language import *
from decimal import Decimal

#store training dataset
df_of_train_tweets = None
#store list of language models
list_of_languages = []
#store parameters
vocabulary = None
size = None
smoothing = None

language_symbol = {
    'eu': 'basque',
    'ca': 'catalan',
    'gl': 'galician',
    'es': 'spanish',
    'en': 'english',
    'pt': 'portugese'
}

n_gram = {
    'unigram': 1,
    'bigram' : 2,
    'trigram': 3
}

#TO-DO make sure A is counted as a and vice versa for vocabulary_all_case. now it is not doing it, skips A
vocabulary_lowercase = string.ascii_lowercase
vocabulary_all_case = string.ascii_letters

def main():
    set_parameters()
    trainModel('training-tweets.txt')
    scoreNewTweets('test-tweets-given.txt')

#TODO take input parameters from user input?
def set_parameters():
    #clean data
    clean_up()
    #set parameters
    global vocabulary, size, smoothing
    vocabulary = vocabulary_lowercase
    size = n_gram['unigram']
    smoothing = 0.5

def trainModel(training_filename):
    #create training dataset
    training_tweets = readTweetsFromFile(training_filename)
    global df_of_train_tweets 
    df_of_train_tweets = pd.DataFrame.from_dict(training_tweets) 
    #create model for each language, based on training set
    createLanguageModel()
    #build n-grams for each language
    buidNgramModel()
    #score new tweets

def scoreNewTweets(filename):
    test_tweets = readTweetsFromFile(filename)
    df_of_test_tweets = pd.DataFrame.from_dict(test_tweets) 
    list_of_langauge_guesses = []
    list_of_probabilities = []
    for index, row in df_of_test_tweets.iterrows():
        #ToDO can we call this without tolower?
        if vocabulary == vocabulary_lowercase:
            resulting_touple = prob_of_language(row['Content'].lower())
        else:
            resulting_touple = prob_of_language(row['Content'])
        list_of_langauge_guesses.append(resulting_touple[0])
        list_of_probabilities.append('%.2E' % Decimal(resulting_touple[1]))

    df_of_test_tweets['guess'] = list_of_langauge_guesses
    df_of_test_tweets['probability'] = list_of_probabilities

    guess_status = []
    for index, row in df_of_test_tweets.iterrows():
        if row['Language'] == row['guess']:
            guess_status.append('correct')
        else:
            guess_status.append('wrong')
    df_of_test_tweets['Status'] = guess_status

    with open('trace.txt', 'w') as file:
        for index, row in df_of_test_tweets.iterrows():
            file.write(row['TweetID'] + "  " + row['guess'] + "  "+ row['probability'] + " " + row['Language'] + "  " + row['Status'] + "\n")
        file.close()

def prob_of_language(line):
    if size == n_gram['unigram']:
        return prob_of_language_unigram(line)
    elif size == n_gram['bigram']:
        return prob_of_language_bigram(line)

def prob_of_language_bigram(line):
    best_probability = float('-inf')
    best_language = None
    for language in list_of_languages:
        language_probablitity = math.log10(language.probability)
        for i in range(len(line)-2):
            #skip out of vacabulary characters
            if (line[i] not in vocabulary) or (line[i+1] not in vocabulary):
                continue
            #skip probability 0
            if language.bigram.loc[line[i], line[i+1]] != 0:
                language_probablitity += math.log10(language.bigram.loc[line[i], line[i+1]])

        if language_probablitity > best_probability:
            best_probability = language_probablitity
            best_language = language.symbol   
    return (best_language, best_probability)

#to-do this is super slow!
def prob_of_language_unigram(line):
    best_probability = float('-inf')
    best_language = None
    for language in list_of_languages:
        total_probability = math.log(language.probability)
        for character in line:
            if character in language.vocabulary:
                character_probability = language.unigram.loc[character]['Probability']
                total_probability += math.log(character_probability)
        if total_probability > best_probability:
            best_probability = total_probability
            best_language = language.symbol
    return (best_language, best_probability)



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
        data = {'TweetID': tweet_id, 'Language': language_list, 'Content': content_list}
        return data

def createLanguageModel():
    global list_of_languages
    total_nb_of_tweets = df_of_train_tweets.shape[0]
    for i in language_symbol.keys():
        language_dataset = df_of_train_tweets[df_of_train_tweets['Language'] == i]
        prob_of_language = language_dataset.shape[0]/total_nb_of_tweets
        a_language = Language(i, language_symbol.get(i), language_dataset, vocabulary, prob_of_language)
        list_of_languages.append(a_language)

def buidNgramModel():
    n_gram_dataset = createDataset()
    #create n-gram for each language
    global list_of_languages
    for language in list_of_languages:
    #language = list_of_languages[0]
        if size == 1:
            #calculate instances of each character
            language.unigram = n_gram_dataset.copy()
            list_of_letters = [character for character in language.vocabulary]
            for character in list_of_letters:
                language.unigram.loc[character]['Instances'] = ''.join(language.dataset['Content']).count(character) + smoothing
            #calculate probablility of each character
            total_instances = language.unigram['Instances'].sum()
            for index, row in language.unigram.iterrows():
                row['Probability'] = row['Instances']/total_instances
        elif size == 2:
            #populate characters
            language.bigram = n_gram_dataset.copy()
            language_as_String = ''.join(language.dataset['Content'])
            list_of_letters_x = [character for character in language.vocabulary]
            list_of_letters_y = [character for character in language.vocabulary]
            #first count each character and add it to the dataset
            for x in list_of_letters_x:
                for y in list_of_letters_y:
                    language.bigram.loc[x, y] = language_as_String.count(x+y) + smoothing
                language.bigram.loc[x,'instances'] = language.bigram.loc[x].sum() 
            #calculate probability of each set of characters
            for x in list_of_letters_x:
                for y in list_of_letters_y:
                    language.bigram.loc[x, y] = language.bigram.loc[x, y] / language.bigram.loc[x,'instances']
            print(language.bigram)
        #elif size == 3:
            #TODO


#Create an empty datagram. It will later be copied over for each language
def createDataset():
    #depending on the vocabulary and size of n_gram, create Dataframe with all letters
    list_of_letters = [character for character in vocabulary] 
    n_gram_dataset = None
    if size == 1:
        dataset = pd.DataFrame(np.zeros((len(vocabulary),2)), columns=['Probability', 'Instances'])
        dataset['Characters'] = list_of_letters
        n_gram_dataset = dataset.set_index('Characters')
    elif size == 2:
        new_list_of_letters = list_of_letters.copy()
        new_list_of_letters.append('instances')
        n_gram_dataset = pd.DataFrame(np.zeros((len(vocabulary),len(vocabulary)+1)), columns=new_list_of_letters, index=list_of_letters)

    return n_gram_dataset

def clean_up():
    list_of_languages.clear()

main()