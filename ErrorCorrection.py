import dataImport
import LSTMWrapper
import numpy as np

dataset = dataImport.dataFile(batch_size=40, window_size=30, step_size=5)
dataset.init_dataset()
data = dataset.alldata
label = dataset.alllabel

model = LSTMWrapper.LSTM_Model()
sess = model.restore_last_session()
data = dataImport.dataFile(batch_size=40, window_size=30, step_size=5)
data.init_dataset()
index = int(input("Please input a data index number for classify"))
databatch, labelbatch = data.genSlidingWindowBatch(index)
pred = model.predict(sess, databatch)
total = np.zeros((1,6))
for p in pred:
    total = total + p
print (np.argmax(total))
print (np.argmax(labelbatch))


