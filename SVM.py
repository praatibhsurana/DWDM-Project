from sklearn import model_selection, svm
from sklearn.metrics import accuracy_score


def svm_predict(train_x_tfidf, train_y, test_x_tfidf):

    # Support Vector Machine
    SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    SVM.fit(train_x_tfidf, train_y)

    predictions_SVM = SVM.predict(test_x_tfidf)

    return predictions_SVM
