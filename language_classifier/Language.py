class Language:
    def __init__(self, symbol, description, dataset, vocabulary, probability, unigram=None, bigram=None, trigram=None):
        self.symbol = symbol
        self.description = description
        self.dataset = dataset
        self.vocabulary = vocabulary
        self.unigram = unigram
        self.bigram = bigram
        self.trigram = trigram
        self.probability = probability
