from loader import *
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle


def main():
    train_set = OneMillion()
    test_set = OneMillion(is_test = True)
    X_train, y_train = shuffle(train_set.data, train_set.labels[:,1])
    X_test, y_test = shuffle(test_set.data, test_set.labels[:,1])
    print("Random Forest:")
    print(random_forest(X_train, y_train, X_test, y_test))
    return

def random_forest(X_train, y_train, X_test, y_test):
    print('Training in progress...')
    rf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth = None, max_features='auto')
    rf.fit(X_train, y_train)
    print('Testing in progress...')
    y_pred = rf.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    cf = confusion_matrix(y_test, y_pred)
    return precision, recall, f1 , cf

main()
