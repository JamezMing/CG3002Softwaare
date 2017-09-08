import numpy as np
import random

batch_size = 1000

class dataFile:
    def __init__(self, filepath = 'WISDM_ar_v1.1_raw.txt', window_size = 200):
        self.dFile = open(filepath)
        self.dict = {'Jogging': 0, 'Walking':1, 'Upstairs':2, 'Downstairs':3, 'Sitting':4, 'Standing':5}
        self.window_size = window_size

    def init_dataset(self):
        self.start_index = 0
        self.end_index = self.start_index + self.window_size
        lines = self.dFile.readlines()
        action_list = []
        previous_action = lines[0].split(',')[1]
        previous_actor = 0
        curr_action = []
        label_list = []
        for l in lines:
            l = l.strip()[:-1]
            entry = l.split(',')
            if len(entry) != 6:
                continue
            actor = entry[0]
            action_type = int(self.dict[entry[1]])
            if (actor!= previous_actor or action_type!=previous_action) and len(curr_action) != 0:
                action_list.append(curr_action)
                label = previous_action
                label_list.append((np.eye(6)[label]))
                curr_action = []
                previous_action = action_type
            else:
                curr_action.append(list(map(float, entry[3:])))
                previous_action = action_type
                previous_actor = actor

        self.data_vec = action_list
        self.label_vec = label_list


    def pop_data(self):
        data = self.data_vec.pop()
        label = self.label_vec.pop()
        package = list(zip(data, label))
        random.shuffle(package)
        data, label = zip(*package)
        return data, label


    def pop_serialized_data(self, step = 10):

        data_batch = []
        label_batch = []
        curr_batch, curr_label = self.pop_data()
        print (len(curr_batch))
        for i in range(0, batch_size):
            data_batch.append(curr_batch[self.start_index: self.end_index])
            label_batch.append(label_batch[self.start_index: self.end_index])
            self.start_index = self.start_index + step
            self.end_index = self.start_index + self.window_size
            if self.end_index > len(curr_batch):
                self.end_index = len(curr_batch)
                break
        return data_batch, label_batch










f = dataFile()
f.init_dataset()
a, b = f.pop_serialized_data()
print (a)





