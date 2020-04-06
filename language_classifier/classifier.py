import pandas as pd
import numpy as np
import string
import math
from Language import *
from Evaluation import *
from decimal import Decimal
import sys
# -*- coding: utf-8 -*-
#store training dataset
df_of_train_tweets = None
#store list of language models
list_of_languages = []
#store parameters
vocabulary = None
size = None
vocabulary_integer = None
size_integer = None
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
    set_parameters(int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]))
    print_info(int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]))
    trainModel('training-tweets.txt', int(sys.argv[1]))
    scoreNewTweets('test-tweets-given.txt', int(sys.argv[1]))


def print_info(vocabulary_choice, ngram_choice, smoothing_value):
    if vocabulary_choice == 0:
        vocab = 'lowercase'
    elif vocabulary_choice == 1:
        vocab = 'allcase'
    elif vocabulary_choice == 2:
        vocab = 'isalpha'
    elif vocabulary_choice == 3:
        vocab = 'special'
    if ngram_choice == 1:
        ngram = 'unigram'
    elif ngram_choice == 2:
        ngram = 'bigram'
    elif ngram_choice == 3:
        ngram = 'trigram'
    print(f'You have chosen {vocab} {ngram} {smoothing_value}')


#TODO take input parameters from user input?
def set_parameters(vocabulary_choice, ngram_size, smoothing_value):
    #clean data
    clean_up()
    #set parameters
    global vocabulary, vocabulary_integer, size, size_integer, smoothing
    if vocabulary_choice == 0:
        vocabulary = string.ascii_lowercase 
        vocabulary_integer = 0
    elif vocabulary_choice == 1:
        vocabulary = string.ascii_letters
        vocabulary_integer = 1
    elif vocabulary_choice == 2: 
        vocabulary = string.ascii_letters
        vocabulary_integer = 2
    elif vocabulary_choice == 3:
        vocabulary = string.ascii_letters +u'ó' +u'ñ' + u'í' +u'é' +u'á' +u'ú' +u'ü' +u'¿'+u'¡' +u'Á' +u'É' +u'Í' + u'Ó' +u'Ñ' + u'Ü' + u'?' +u'!'
        vocabulary_integer = 3
    
    size = ngram_size
    size_integer = ngram_size
    smoothing = smoothing_value


def trainModel(training_filename, vocabulary_choice):
    #create training dataset
    training_tweets = readTweetsFromFile(training_filename)
    global df_of_train_tweets 
    df_of_train_tweets = pd.DataFrame.from_dict(training_tweets) 
    #create model for each language, based on training set
    divide_tweets_by_language()
    #build n-grams for each language
    buildNgramModel(vocabulary_choice)


def scoreNewTweets(filename, vocabulary_choice):
    test_tweets = readTweetsFromFile(filename)
    df_of_test_tweets = pd.DataFrame.from_dict(test_tweets) 
    list_of_language_guesses = []
    list_of_probabilities = []
    for index, row in df_of_test_tweets.iterrows():
        if vocabulary_choice == 0:
            resulting_tuple = prob_of_language(row['Content'].lower(), vocabulary_choice)
        else:
            resulting_tuple = prob_of_language(row['Content'], vocabulary_choice)
        list_of_language_guesses.append(resulting_tuple[0])
        list_of_probabilities.append('%.2E' % Decimal(resulting_tuple[1]))

    df_of_test_tweets['guess'] = list_of_language_guesses
    df_of_test_tweets['probability'] = list_of_probabilities
    evaluate_scores(df_of_test_tweets)
    

def evaluate_scores(result_df):
    guess_status = []
    correct_guess = 0
    evaluation_dictionary = {}
    #create evaluation dict to store results
    for lan in language_symbol.keys():
        evaluation_dictionary[lan] = Evaluation()

    for index, row in result_df.iterrows():
        evaluation_dictionary.get(row['guess']).all_predicted += 1
        evaluation_dictionary.get(row['Language']).all_actual += 1
        if row['Language'] == row['guess']:
            guess_status.append('correct')
            correct_guess += 1
            evaluation_dictionary.get(row['Language']).true_positive += 1
        else:
            guess_status.append('wrong')
    result_df['Status'] = guess_status

    #calculate stats
    accuracy = correct_guess / len(result_df) 
    for lang in evaluation_dictionary:
        if evaluation_dictionary.get(lang).all_predicted != 0:
            evaluation_dictionary.get(lang).precision = evaluation_dictionary.get(lang).true_positive / evaluation_dictionary.get(lang).all_predicted
        if evaluation_dictionary.get(lang).all_actual != 0:
            evaluation_dictionary.get(lang).recall = evaluation_dictionary.get(lang).true_positive / evaluation_dictionary.get(lang).all_actual
        if evaluation_dictionary.get(lang).precision != 0 or evaluation_dictionary.get(lang).recall != 0:
            evaluation_dictionary.get(lang).f1 = (2 * evaluation_dictionary.get(lang).precision * evaluation_dictionary.get(lang).recall) / (evaluation_dictionary.get(lang).precision+evaluation_dictionary.get(lang).recall)
    create_output_files(result_df, accuracy, evaluation_dictionary)


def create_output_files(result, accuracy, evaluation):
    #create trace file
    trace_filename = 'trace_'+str(vocabulary_integer)+'_'+str(size_integer)+'_'+str(smoothing)+'.txt'
    #create evaluation file
    evaluation_filename = 'eval_'+str(vocabulary_integer)+'_'+str(size_integer)+'_'+str(smoothing)+'.txt'
    with open(trace_filename, 'w', encoding="utf-8") as file:
        for index, row in result.iterrows():
            file.write(row['TweetID'] + "  " + row['guess'] + "  "+ row['probability'] + "  " + row['Language'] + "  " + row['Status'] + "\n")
        file.close()
        
    with open(evaluation_filename, 'w', encoding="utf-8") as file:
        #accuracy
        file.write(str(accuracy) + "\n")
        #all precisions
        for index in evaluation:
            file.write(str(evaluation.get(index).precision) + "  " )
        file.write("\n")
        #all recalls
        for index in evaluation:
            file.write(str(evaluation.get(index).recall) + "  " )
        file.write("\n")
        #all f1s
        macrof1 =0
        weighedf1 = 0
        for index in evaluation:
            macrof1 += evaluation.get(index).f1
            weighedf1 += (evaluation.get(index).f1 * evaluation.get(index).all_actual)
            file.write(str(evaluation.get(index).f1) + "  " )
        file.write("\n")
        #macros
        macrof1 = macrof1 / len(evaluation)
        file.write(str(macrof1) + "  " + str(weighedf1/100))    
        file.close()


def prob_of_language(line, vocabulary_choice):
    if size == n_gram['unigram']:
        return prob_of_language_unigram(line, vocabulary_choice)
    elif size == n_gram['bigram']:
        return prob_of_language_bigram(line, vocabulary_choice)
    elif size == n_gram['trigram']:
        return prob_of_language_trigram(line, vocabulary_choice)


def prob_of_language_bigram(line, vocabulary_choice):
    best_probability = float('-inf')
    best_language = None
    for language in list_of_languages:
        language_probablitity = math.log10(language.probability)
        for i in range(len(line)-1):
            bigram_substring = line[i] + line[i + 1]
            if vocabulary_choice == 2:
                if valid_alpha_characters(bigram_substring):
                    if bigram_substring in language.bigram.index:
                        language_probablitity *= math.log10(language.bigram.loc[bigram_substring]['Probability'])
                    else:
                        language.bigram.loc['NOT APPEAR']['Instances'] += 1
                        language.bigram.loc['NOT APPEAR']['Probability'] = \
                            language.bigram.loc['NOT APPEAR']['Instances']/language.bigram['Instances'].sum()
            else:
                if valid_characters(bigram_substring):
                    language_probablitity *= math.log10(language.bigram.loc[line[i], line[i + 1]])
        if language_probablitity > best_probability:
            best_probability = language_probablitity
            best_language = language.symbol   
    return (best_language, best_probability)


#to-do this is super slow!
def prob_of_language_unigram(line, vocabulary_choice):
    best_probability = float('-inf')
    best_language = None
    for language in list_of_languages:
        language_probability = math.log10(language.probability)
        for character in line:
            if vocabulary_choice == 2:
                if valid_alpha_characters(character):
                    if character in language.unigram.index:
                        language_probability *= math.log10(language.unigram.loc[character]['Probability'])
                    else:
                        language.unigram.loc['NOT APPEAR']['Instances'] += 1
                        language.unigram.loc['NOT APPEAR']['Probability'] = \
                            language.unigram.loc['NOT APPEAR']['Instances'] / language.unigram['Instances'].sum()
                        language_probability *= math.log10(language.unigram.loc['NOT APPEAR']['Probability'])
            else:
                if valid_characters(character):
                    language_probability *= math.log10(language.unigram.loc[character]['Probability'])
        if language_probability > best_probability:
            best_probability = language_probability
            best_language = language.symbol
    return (best_language, best_probability)


def prob_of_language_trigram(line, vocabulary_choice):
    best_probability = float('-inf')
    best_language = None
    for language in list_of_languages:
        language_probability = math.log10(language.probability)
        for i in range(len(line)-2):
            trigram_substring = line[i] + line[i + 1] + line[i + 2]
            if vocabulary_choice == 2:
                if valid_alpha_characters(trigram_substring):
                    if trigram_substring in language.trigram.index:
                        language_probability *= math.log10(language.trigram.loc[trigram_substring]['Probability'])
                    else:
                        language.trigram.loc['NOT APPEAR']['Instances'] += 1
                        language.trigram.loc['NOT APPEAR']['Probability'] = \
                            language.trigram.loc['NOT APPEAR']['Instances']/language.trigram['Instances'].sum()
            else:
                if valid_characters(trigram_substring):
                    language_probability *= math.log10(language.trigram.loc[trigram_substring]['Probability'])
        language_probability *= math.log10(language.trigram.loc['NOT APPEAR']['Probability'])
        if language_probability > best_probability:
            best_probability = language_probability
            best_language = language.symbol
    return (best_language, best_probability)


def valid_alpha_characters(characters):
    for character in characters:
        if character.isalpha():
            pass
        else:
            return False
    return True


def valid_characters(characters):
    for character in characters:
        if character in vocabulary:
            pass
        else:
            return False
    return True


def readTweetsFromFile(fileName):
    with open(fileName, encoding='utf8') as file:
        language_list = []
        content_list = []
        tweetid_list = []
        for line in file.readlines():
            try:
                tweet_id, user_name, language, tweet = line.split(maxsplit=3)
                language_list.append(language.strip())
                content_list.append(tweet.strip())
                tweetid_list.append(tweet_id.strip())
            except ValueError:
                pass 
        data = {'TweetID': tweetid_list, 'Language': language_list, 'Content': content_list}
        return data


def divide_tweets_by_language():
    global list_of_languages
    total_nb_of_tweets = df_of_train_tweets.shape[0]
    for i in language_symbol.keys():
        language_dataset = df_of_train_tweets[df_of_train_tweets['Language'] == i]
        prob_of_language = language_dataset.shape[0]/total_nb_of_tweets
        a_language = Language(i, language_symbol.get(i), language_dataset, vocabulary, prob_of_language)
        list_of_languages.append(a_language)


def buildNgramModel(vocabulary_choice):
    global list_of_languages, smoothing
    if vocabulary_choice == 2:  # isalpha stuff
        global list_of_languages
        for language in list_of_languages:
            language_as_string = ''.join(language.dataset['Content'])
            alpha_characters = list(filter(lambda a_character: a_character.isalpha(), language_as_string))
            alpha_characters = np.array(alpha_characters)
            alpha_characters_unique = np.unique(alpha_characters)
            if size == 1:
                list_of_instances = []
                for character in alpha_characters_unique:
                    list_of_instances.append(language_as_string.count(character) + smoothing)
                unigram = pd.DataFrame({'Characters': alpha_characters_unique, 'Instances': list_of_instances})
                unigram.set_index('Characters', inplace=True)
                unigram.loc['NOT APPEAR'] = [smoothing]
                unigram['Probability'] = unigram['Instances'] / unigram['Instances'].sum()
                language.unigram = unigram
            elif size == 2:
                all_permutations_as_list = []
                for x in alpha_characters_unique:
                    for y in alpha_characters_unique:
                        all_permutations_as_list.append(x+y)
                all_permutations_as_list = np.array(all_permutations_as_list)
                list_of_instances = [language_as_string.count(character) + smoothing for character in
                                     np.array(all_permutations_as_list)]
                bigram = pd.DataFrame({'Characters': all_permutations_as_list, 'Instances': list_of_instances})
                bigram.set_index('Characters', inplace=True)
                bigram.loc['NOT APPEAR'] = [smoothing]
                bigram['Probability'] = bigram['Instances'] / bigram['Instances'].sum()
                language.bigram = bigram
            elif size == 3:
                all_permutations_as_list = []
                for x in alpha_characters_unique:
                    for y in alpha_characters_unique:
                        for z in alpha_characters_unique:
                            all_permutations_as_list.append(x+y+z)
                all_permutations_as_list = np.array(all_permutations_as_list)
                list_of_instances = [language_as_string.count(character) + smoothing for character in
                                     np.array(all_permutations_as_list)]
                trigram = pd.DataFrame(
                    {'Characters': all_permutations_as_list, 'Instances': list_of_instances})
                trigram.set_index('Characters', inplace=True)
                trigram.loc['NOT APPEAR'] = [smoothing]
                trigram['Probability'] = trigram['Instances'] / trigram['Instances'].sum()
                language.trigram = trigram
    else:
        n_gram_dataset = createDataset()
        #create n-gram for each language
        for language in list_of_languages:
            if size == 1:
                #calculate instances of each character
                language.unigram = n_gram_dataset.copy()
                list_of_letters = [character for character in language.vocabulary]
                language_as_string = ''.join(language.dataset['Content'])
                for character in list_of_letters:
                    language.unigram.loc[character]['Instances'] = language_as_string.count(character) + smoothing
                #calculate probablility of each character
                total_instances = language.unigram['Instances'].sum()
                language.unigram['Probability'] = language.unigram['Instances']/total_instances
            elif size == 2:
                #populate characters
                language.bigram = n_gram_dataset.copy()
                language_as_string = ''.join(language.dataset['Content'])
                list_of_letters_x = [character for character in language.vocabulary]
                list_of_letters_y = [character for character in language.vocabulary]
                #first count each character and add it to the dataset
                for x in list_of_letters_x:
                    for y in list_of_letters_y:
                        language.bigram.loc[x, y] = language_as_string.count(x+y) + smoothing
                    language.bigram.loc[x,'instances'] = language.bigram.loc[x].sum()
                #calculate probability of each set of characters
                for x in list_of_letters_x:
                    for y in list_of_letters_y:
                        language.bigram.loc[x, y] = language.bigram.loc[x, y] / language.bigram.loc[x,'instances']
            elif size == 3:
                language_as_string = ''.join(language.dataset['Content'])
                all_permutations_as_list = []
                list_of_letters = [character for character in language.vocabulary]
                for x in list_of_letters:
                    for y in list_of_letters:
                        for z in list_of_letters:
                            all_permutations_as_list.append(x+y+z)
                all_permutations_as_list = np.array(all_permutations_as_list)
                list_of_instances = [language_as_string.count(character) + smoothing for character in
                                     np.array(all_permutations_as_list)]
                language.trigram = pd.DataFrame({'Characters': all_permutations_as_list, 'Instances': list_of_instances})
                language.trigram.set_index('Characters', inplace=True)
                language.trigram['Probability'] = language.trigram['Instances']/language.trigram['Instances'].sum()


#Create an empty datagram. It will later be copied over for each language
def createDataset():
    #depending on the vocabulary and size of n_gram, create Dataframe with all letters
    list_of_letters = [character for character in vocabulary]
    n_gram_dataset = None
    if size == 1:
        dataset = pd.DataFrame(np.zeros((len(list_of_letters),2)), columns=['Probability', 'Instances'])
        dataset['Characters'] = list_of_letters
        n_gram_dataset = dataset.set_index('Characters')
    elif size == 2:
        new_list_of_letters = list_of_letters.copy()
        new_list_of_letters.append('instances')
        n_gram_dataset = pd.DataFrame(np.zeros((len(list_of_letters),len(list_of_letters)+1)), columns=new_list_of_letters, index=list_of_letters)
    return n_gram_dataset


def clean_up():
    list_of_languages.clear()


main()