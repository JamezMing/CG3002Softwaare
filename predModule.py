import LSTMWrapper
import WindowSVM

class predModule:
    def __init__(self, pool_unit_size = 5):
        self.lstm_model = LSTMWrapper.LSTM_Model()
        self.window_pool = []
        self.unit_size = pool_unit_size
        self.svm_model = WindowSVM.SVMClassifier()

    def feed_window_data(self, data):
        data = self.lstm_model.predict(data)
        if (len(self.window_pool) < self.unit_size):
            self.window_pool.append(data)
            result = data
        else:
            self.window_pool.pop()
            self.window_pool.append(data)
            result = self.svm_model.predict(self.window_pool)
        return result
