import sys
import getopt
import os
import math
import operator
from timeit import default_timer as timer

class NaiveBayes:
    class TrainSplit:
        """Represents a set of training/testing data. self.train is a list of Examples, as is self.dev and self.test. 
        """
        def __init__(self):
            self.train = []
            self.dev = []
            self.test = []

    class Example:
        """Represents a document with a label. klass is 'aid' or 'not' by convention.
             words is a list of strings.
        """
        def __init__(self):
            self.klass = ''
            self.words = []

    def __init__(self):
        """NaiveBayes initialization"""
        self.FILTER_STOP_WORDS = False
        self.USE_BIGRAMS = False
        self.BEST_MODEL = False
        self.stopList = set(self.readFile('data/english.stop'))
        self.aid = [] #will want size of this
        self.notaid = [] #will want size of this
        self.vocab = set() #for the total vocabulary
        self.num_docs = 0
        self.num_aid_docs = 0
        self.num_notaid_docs = 0
        self.timesRan = 0
        self.logprior_aid = 0
        self.logprior_not = 0
        self.count_aid = {}
        self.count_not = {}
        #TODO: add other data structures needed in classify() and/or addExample() below
        


    #############################################################################
    # TODO TODO TODO TODO TODO 
    # Implement the Multinomial Naive Bayes classifier with add-1 smoothing
    # If the FILTER_STOP_WORDS flag is true, you must remove stop words
    # If the USE_BIGRAMS flag is true, your methods must use bigram features instead of the usual 
    # bag-of-words (unigrams)
    # If either of the FILTER_STOP_WORDS or USE_BIGRAMS flags is on, the other is meant to be off. 
    # Hint: Use filterStopWords(words) defined below
    # Hint: Remember to add start and end tokens in the bigram implementation
    # Hint: When doing add-1 smoothing with bigrams, V = # unique bigrams in data. 

    def classify(self, words):
        """ TODO
            'words' is a list of words to classify. Return 'aid' or 'not' classification.
        """
        if self.FILTER_STOP_WORDS:
            words = self.filterStopWords(words)

        # first time, find all counts and add them to a dictionary
        if self.timesRan == 0:
            self.logprior_aid = math.log(self.num_aid_docs/self.num_docs)
            self.logprior_not = math.log(self.num_notaid_docs/self.num_docs)
            for v in self.vocab:
                numAid = self.aid.count(v)
                numNot = self.notaid.count(v)
                self.count_aid[v] = numAid
                self.count_not[v] = numNot

        self.timesRan +=1

        p_aid = self.logprior_aid
        p_not = self.logprior_not

        for w in words:
            numAid = 0
            if w in self.count_aid:
                numAid = self.count_aid[w]
            numNot = 0
            if w in self.count_not:
                numNot = self.count_not[w]

            logliklihood_aid = math.log((numAid + 1)/(len(self.aid) + len(self.vocab)))
            logliklihood_not = math.log((numNot + 1)/(len(self.notaid) + len(self.vocab)))

            p_aid += logliklihood_aid
            p_not += logliklihood_not

        if p_aid > p_not:
            return 'aid'
        else:
            return 'not'
    
    # round 1: Train Accuracy: 0.82946878266654
    # Dev Accuracy: 0.731441896618733

    def addExample(self, klass, words):
        self.timesRan = 0
        if self.FILTER_STOP_WORDS:
            words = self.filterStopWords(words)

        self.num_docs+=1
        if klass == 'aid':
            self.num_aid_docs+=1
        else:
            self.num_notaid_docs+=1
        for w in words:
            if klass == 'aid':
                self.aid.append(w)
            elif klass == 'not':
                self.notaid.append(w)
            self.vocab.add(w)


        """
         * TODO
         * Train your model on an example document with label klass ('aid' or 'not') and
         * words, a list of strings.
         * You should store whatever data structures you use for your classifier 
         * in the NaiveBayes class.
         * Returns nothing
        """
        
        pass
        
    # END TODO (Modify code beyond here with caution)
    #############################################################################
    
    def readFile(self, fileName):
        """
         * Code for reading a file.  you probably don't want to modify anything here, 
         * unless you don't like the way we segment files.
        """
        contents = []
        f = open(fileName,encoding="utf8")
        for line in f:
            contents.append(line)
        f.close()
        result = self.segmentWords('\n'.join(contents)) 
        return result

    def segmentWords(self, s):
        """
         * Splits lines on whitespace for file reading
        """
        return s.split()

    def buildSplit(self,include_test=True):
    
        split = self.TrainSplit()
        datasets = ['train','dev']
        if include_test:
            datasets.append('test')
        for dataset in datasets:
            for klass in ['aid','not']:
                dataFile = os.path.join('data',dataset,klass + '.txt')
                with open(dataFile,'r', encoding="utf8") as f:
                    docs = [line.rstrip('\n') for line in f]
                    for doc in docs:
                        example = self.Example()
                        example.words = doc.split()
                        example.klass = klass
                        if dataset == 'train':
                            split.train.append(example)
                        elif dataset == 'dev':
                            split.dev.append(example)
                        else:
                            split.test.append(example)
        return split


    def filterStopWords(self, words):
        """Filters stop words."""
        filtered = []
        for word in words:
            if not word in self.stopList and word.strip() != '':
                filtered.append(word)
        return filtered
    
def evaluate(FILTER_STOP_WORDS,USE_BIGRAMS):
    classifier = NaiveBayes()
    classifier.FILTER_STOP_WORDS = FILTER_STOP_WORDS
    classifier.USE_BIGRAMS = USE_BIGRAMS
    split = classifier.buildSplit(include_test=False)
   
    for example in split.train:
        classifier.addExample(example.klass,example.words)


    train_accuracy = calculate_accuracy(split.train,classifier)
    dev_accuracy = calculate_accuracy(split.dev,classifier)

    print('Train Accuracy: {}'.format(train_accuracy))
    print('Dev Accuracy: {}'.format(dev_accuracy))


def calculate_accuracy(dataset,classifier):
    acc = 0.0
    if len(dataset) == 0:
        return 0.0
    else:
        for example in dataset:
            guess = classifier.classify(example.words)
            if example.klass == guess:
                acc += 1.0
        return acc / len(dataset)

        
def main():
    start = timer()
    FILTER_STOP_WORDS = False
    USE_BIGRAMS = False
    (options, args) = getopt.getopt(sys.argv[1:], 'fb')
    if ('-f','') in options:
      FILTER_STOP_WORDS = True
    elif ('-b','') in options:
      USE_BIGRAMS = True

    evaluate(FILTER_STOP_WORDS,USE_BIGRAMS)
    elapsed_time = timer() - start # in seconds
    print(elapsed_time)

if __name__ == "__main__":
        main()
