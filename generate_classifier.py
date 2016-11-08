import re
import math
import collections
import itertools
import os
import nltk
import nltk.classify.util
import pickle
from nltk import precision
from nltk import recall
from nltk.classify import NaiveBayesClassifier
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.corpus import movie_reviews

CLASSIFIER_PATH = 'lib/NaiveBayesClassifier.pickle'

def word_feats(words):
    return dict([(word, True) for word in words])


def create_word_scores():
    negids = movie_reviews.fileids('neg')
    posids = movie_reviews.fileids('pos')
    # creates lists of all positive and negative words
    posfeats = [word_feats(movie_reviews.words(fileids=[f])) for f in posids]
    negfeats = [word_feats(movie_reviews.words(fileids=[f])) for f in negids]

    posWords = []
    negWords = []
    for feat in posfeats:
        posWords.append(feat.keys())

    for feat in negfeats:
        negWords.append(feat.keys())

    # build frequency distibution of all words and then frequency
    # distributions of words within positive and negative labels
    word_fd = FreqDist()
    cond_word_fd = ConditionalFreqDist()
    for word in posWords:
        word_fd[word[0].lower()] += 1
        cond_word_fd['pos'][word[0].lower()] += 1
    for word in negWords:
        word_fd[word[0].lower()] += 1
        cond_word_fd['neg'][word[0].lower()] += 1

    # finds the number of positive and negative words, as well as the total number of words
    pos_word_count = cond_word_fd['pos'].N()
    neg_word_count = cond_word_fd['neg'].N()
    total_word_count = pos_word_count + neg_word_count

    # builds dictionary of word scores based on chi-squared test
    word_scores = {}
    for word, freq in word_fd.iteritems():
        pos_score = BigramAssocMeasures.chi_sq(
            cond_word_fd['pos'][word], (freq, pos_word_count), total_word_count)
        neg_score = BigramAssocMeasures.chi_sq(
            cond_word_fd['neg'][word], (freq, neg_word_count), total_word_count)
        word_scores[word] = pos_score + neg_score

    return word_scores

# finds the best 'number' words based on word scores


def find_best_words(word_scores, number):
    best_vals = sorted(word_scores.iteritems(), key=lambda (w, s): s, reverse=True)[:number]
    best_words = set([w for w, s in best_vals])
    return best_words

# creates feature selection mechanism that only uses best words


def best_word_features(words):
    return dict([(word, True) for word in words if word in best_words])


# this function takes a feature selection mechanism and returns its
# performance in a variety of metrics
def evaluate_features(feature_select):
    negids = movie_reviews.fileids('neg')
    posids = movie_reviews.fileids('pos')

    negFeatures = [(feature_select(movie_reviews.words(fileids=[f])), 'negative') for f in negids]
    posFeatures = [(feature_select(movie_reviews.words(fileids=[f])), 'positive') for f in posids]

    # selects 3/4 of the features to be used for training and 1/4 to be used for testing
    posCutoff = int(math.floor(len(posFeatures) * 3 / 4))
    negCutoff = int(math.floor(len(negFeatures) * 3 / 4))
    trainFeatures = posFeatures[:posCutoff] + negFeatures[:negCutoff]
    testFeatures = posFeatures[posCutoff:] + negFeatures[negCutoff:]

    # trains a Naive Bayes Classifier
    classifier = NaiveBayesClassifier.train(trainFeatures)

    # initiates referenceSets and testSets
    referenceSets = collections.defaultdict(set)
    testSets = collections.defaultdict(set)

    # puts correctly labeled sentences in referenceSets and the predictively
    # labeled version in testsets
    for i, (features, label) in enumerate(testFeatures):
        referenceSets[label].add(i)
        predicted = classifier.classify(features)
        testSets[predicted].add(i)

    # prints metrics to show how well the feature selection did
    print 'train on %d instances, test on %d instances' % (len(trainFeatures), len(testFeatures))
    print 'accuracy:', nltk.classify.util.accuracy(classifier, testFeatures)
    print 'pos precision:', nltk.precision(referenceSets['positive'], testSets['positive'])
    print 'pos recall:', nltk.recall(referenceSets['positive'], testSets['positive'])
    print 'neg precision:', nltk.precision(referenceSets['negative'], testSets['negative'])
    print 'neg recall:', nltk.recall(referenceSets['negative'], testSets['negative'])
    classifier.show_most_informative_features(10)
    return classifier

def main():
    word_scores = create_word_scores()
    classifier = evaluate_features(word_feats)
    f = open(CLASSIFIER_PATH, 'wb')
    pickle.dump(classifier, f)
    f.close()
    print 'classifier saved at %s' % CLASSIFIER_PATH

if __name__ == '__main__':
    main()
