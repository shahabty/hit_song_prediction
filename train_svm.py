from loader import *
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from sklearn import svm

def main():
    train_set = OneMillion()
    test_set = OneMillion(is_test = True)
    X_train, y_train = shuffle(train_set.data, train_set.labels[:,1])
    X_test, y_test = shuffle(test_set.data, test_set.labels[:,1])
    print("SVM:")
    print(support_vector_machine(X_train, y_train, X_test, y_test))
    return 

def support_vector_machine(X_train, y_train, X_test, y_test):
    print('Training in progress...')
    svm_clf = svm.SVC(kernel='rbf', degree=10, gamma='auto')
    svm_clf.fit(X_train, y_train)
    print('Testing in progress...')
    y_pred = svm_clf.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    cf = confusion_matrix(y_test, y_pred)
    return precision, recall, f1 , cf

main()
