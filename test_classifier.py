import pickle

CLASSIFIER_PATH = 'lib/NaiveBayesClassifier.pickle'


def word_feats(words):
    return dict([(word, True) for word in words])


def main():
    f = open(CLASSIFIER_PATH, 'rb')
    classifier = pickle.load(f)
    f.close()
    newerest_review = "I hated the movie"
    emotion = classifier.classify(word_feats(newerest_review.split()))
    print emotion

main()
