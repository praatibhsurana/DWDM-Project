from sklearn import model_selection, naive_bayes
from sklearn.metrics import accuracy_score


def nb_predict(train_x_tfidf, train_y, test_x_tfidf, test_y):

    # Naive Bayes
    naive = naive_bayes.MultinomialNB()
    naive.fit(train_x_tfidf, train_y)

    predictions_nb = naive.predict(test_x_tfidf)

    return predictions_nb
