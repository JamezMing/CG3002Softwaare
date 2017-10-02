from sklearn import svm
from sklearn.externals import joblib
import os
import numpy as np
import LSTMWrapper

class SVMClassifier:

    def __init__(self, kernal='rbf', decision_func_shape='ovo', C=1.0, degree = 5, max_iter=100000, filename = 'svm_classifier_model.pkl'):
        if os.path.isfile(filename):
            self.classifier = joblib.load(filename)

        else:
            self.classifier = svm.SVC(decision_function_shape=decision_func_shape, kernel=kernal, C=C,
                                                         max_iter=max_iter, degree=degree)
        self.lstm_model = LSTMWrapper.LSTM_Model()

    def __train__(self, data, label):
        print("Start Training")
        self.classifier.fit(data, label)
        _ = joblib.dump(self.classifier, './svm_classifier_model.pkl')

    def train_svm(self):
        train_data, train_label = self.lstm_model.genSVMTrainData()
        svm_traind = []
        svm_trainl = []

        for i in range (0, len(train_data)):
            curr_win = []
            window_size = 5
            start_index = 0
            end_index = start_index + window_size
            while(end_index < len(train_data[i])):
                curr_win = train_data[i][start_index:end_index]
                print(curr_win)
                svm_traind.append(curr_win)
                svm_trainl.append(np.argmax(train_label[i]))
                start_index = start_index + 1
                end_index = start_index + window_size

        self.__train__(svm_traind, svm_trainl)

    def predict(self, window_data):
        return self.classifier.predict(window_data)


svm = SVMClassifier()
svm.train_svm()