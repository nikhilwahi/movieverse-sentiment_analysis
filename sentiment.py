import pickle
import os
dir_path = os.path.dirname(os.path.realpath(__file__))

CLASSIFIER_PATH = dir_path+'/'+'lib/NaiveBayesClassifier.pickle'


def word_feats(words):
    return dict([(word, True) for word in words])


def analyse_review(newerest_review):
    f = open(CLASSIFIER_PATH, 'rb')
    classifier = pickle.load(f)
    f.close()
    emotion = classifier.classify(word_feats(newerest_review.split()))
    return emotion
