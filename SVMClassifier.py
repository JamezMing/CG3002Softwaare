from sklearn import svm
import numpy as np
import dataImport
from sklearn.externals import joblib

class SVMClassifier:
    def __init__(self, kernal='rbf', decision_func_shape = 'ovo', C=1.0, max_iter = 100000 ):
        self.classifier = svm.SVC(decision_function_shape=decision_func_shape, kernel=kernal, C=C, max_iter=max_iter)

    def train(self, data, label):
        print("Start Training")
        self.classifier.fit(data, label)
        _ = joblib.dump(self.classifier, './svm_classifier_model.pkl')

    def predict(self, data):
        return self.classifier.predict(data)


dataset = dataImport.dataFile(100, 40)
dataset.init_dataset()
data, label = dataset.get_test_set()
data = np.reshape(data, (len(data), -1))
svc = SVMClassifier()
label_index = [np.argmax(x) for x in label]
svc.train(data, label_index)
dataset.init_dataset()
testdata, testlabel = dataset.get_test_set()
testdata = np.reshape(testdata, (len(testdata), -1))
conf_mat = np.zeros((6,6))

dataset.init_dataset()
'''for i in range(len(dataset.test_data)):
    clip = dataset.test_data[i]
    gtruth = dataset.test_label[i]
    datawindows = dataset.genSlidingWindowData(clip)
    print (datawindows.shape)
    datawindows = np.reshape(datawindows, (len(datawindows), -1))
    voteMat =np.zeros((6,1))
    for j in range(len(datawindows)):
        index_v = svc.predict([datawindows[j]])
        voteMat[index_v] = voteMat[index_v] + 1
    vote_res = np.argmax(voteMat)
    conf_mat[gtruth][vote_res] = conf_mat[gtruth][vote_res] + 1'''


for i in range(len(testdata)):
    voteMat =np.zeros((6,1))
    pred_res = svc.predict([testdata[i]])[0]
    label_res = np.argmax(testlabel[i])
    conf_mat[label_res][pred_res] = conf_mat[label_res][pred_res] + 1
print (conf_mat)
TP = np.zeros((6,1))
FP = np.zeros((6,1))
FN = np.zeros((6,1))
NumEle = np.zeros((6,1))

for i in range(0,6):
    TP[i] = conf_mat[i][i] + TP[i]
    NumEle[i] = sum(conf_mat[i])

for i in range(0,6):
    for j in range(0,6):
        if i!=j:
            FP[i] = conf_mat[i][j] + FP[i]
            FN[j] = conf_mat[i][j] + FN[j]

pre = np.divide(TP*1.0,(TP+FP))
rec = np.divide(TP, (FN + TP))
F1 = 2*np.divide((np.multiply(pre, rec)), (pre + rec))
acc = np.divide(TP, NumEle)

print("F1 score is: " + str(np.average(F1)))
print("Recall is: " + str(np.average(rec)))
print("Precision is: " + str(np.average(pre)))