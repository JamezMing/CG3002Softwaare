import numpy as np
import random

log_file = open('log_gen', 'a+')


class dataFile:
    def __init__(self, filepath = 'WISDM_ar_v1.1_raw.txt', batch_size = 100, window_size = 100):
        self.dFile = open(filepath)
        self.dict = {'Jogging': 0, 'Walking':1, 'Upstairs':2, 'Downstairs':3, 'Sitting':4, 'Standing':5}
        self.batch_size = batch_size
        self.start_index = 0
        self.window_size = window_size
        self.end_index = self.start_index + self.window_size


    def init_dataset(self):
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
        package = list(zip(action_list, label_list))
        random.shuffle(package)
        action_list, label_list = zip(*package)
        self.curr_data = action_list[0]
        self.curr_label = label_list[0]
        self.data_vec = action_list[1:]
        self.label_vec = label_list[1:]


    def pop_data(self):
        data = self.data_vec[0]
        self.data_vec = self.data_vec[1:]
        label = [self.label_vec[0]]*len(data)
        self.label_vec = self.label_vec[1:]
        return data, label

    def hasNextClip(self):
        if (len(self.data_vec) != 0):
            return True
        else:
            return False

    def sliding_window_batches(self, step_size = 2):
        window_data = []
        window_label = []
        curr_window = []
        curr_label = []



        while (len(window_data) < self.batch_size):
            log_file.write("End Index :" + str(self.end_index) + " Win Size: " + str(self.window_size) + " Step Size: " + str(
                step_size) + " Data Length: " + str(len(self.curr_data)) + " Num Clips Left: " + str(len(self.data_vec)) + '\n')
            #If the clip has not come to the end
            if(self.end_index + self.window_size + step_size <= len(self.curr_data)):
                curr_window = self.curr_data[self.start_index:self.end_index]
                curr_label = [self.curr_label] * len(curr_window)
                self.start_index = self.start_index + step_size
                self.end_index = self.start_index + self.window_size
            #If the Clip has come to the end, put whatever is left into the batch, amend the insufficent ones by copying the last frame, and start the next one in batch from a new clip
            else:
                curr_window = self.curr_data[self.start_index:]
                num_frame_left = self.window_size - len(curr_window)
                curr_window.append(curr_window[-1] * num_frame_left)
                if (self.hasNextClip()):
                    self.curr_data, self.curr_label = self.pop_data()
                    self.start_index = 0
                    self.end_index = self.start_index + self.window_size

            window_data.append(curr_window)
            window_label.append(curr_label)

            for a in window_data:
                if (len(a) != 100):
                    log_file.write(str(len(a)) + '\n')

        return window_data, window_label



        '''pop = False
        self.start_index = 0
        self.end_index = self.start_index + window_size
        while(len(window_data) < self.batch_size):
            if(pop == True):
                self.start_index = 0
                self.end_index = len(data) - len(curr_window)
                data, label = self.pop_data()
                pop = False
            print("End Index :" + str(self.end_index) + " Win Size: " + str(window_size) + " Step Size: " + str(
                step_size) + " Data Length: " + str(len(data)) + " Num Clips Left: " + str(len(self.data_vec)))

            if self.end_index + window_size + step_size < len(data) and pop == False:
                curr_window = data[self.start_index:self.end_index]
                curr_label = [label]*len(curr_window)
                self.start_index = self.end_index+ step_size
                self.end_index = self.start_index + window_size

            else:
                curr_window = data[self.start_index:]
                curr_label = [label] * len(curr_window)
                if (self.hasNextClip()):
                    curr_window.append(data[self.start_index: self.end_index])
                    curr_label.append([label] * len(curr_window))
                    pop = True

                else:
                    curr_window.append(curr_window[-1]*(len(data) - len(curr_window)))
                    curr_label.append([label]*(len(data) - len(curr_window)))
            window_data.append(curr_window)
            window_label.append(curr_label)
        return window_data, window_label'''

#Sdeing W
dataSet = dataFile()
dataSet.init_dataset()
while(dataSet.hasNextClip()):
    (a,b) = dataSet.sliding_window_batches()





