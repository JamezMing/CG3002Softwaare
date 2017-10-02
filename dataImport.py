import numpy as np
import random

log_file = open('log_gen', 'a+')


class dataFile:
    def __init__(self, batch_size, window_size, step_size = 10, filepath = 'WISDM_ar_v1.1_raw.txt'):
        self.dict = {'Jogging': 0, 'Walking':1, 'Upstairs':2, 'Downstairs':3, 'Sitting':4, 'Standing':5}
        self.batch_size = batch_size
        self.start_index = 0
        self.window_size = window_size
        self.end_index = self.start_index + self.window_size
        self.filepath = filepath
        self.step_size = step_size

    def __prepare_data__(self):
        data_list = [[], [], [], [], [], []]
        self.dFile = open(self.filepath)
        lines = self.dFile.readlines()
        extend_data = []
        for li in lines:
            entry = li.split(',')
            if len(entry) != 6:
                continue
            action_type = int(self.dict[entry[1]])
            data_list[action_type].append(li)
        for i in range(2, 6):
            extend_data.extend(data_list[i])
        return extend_data


    def init_dataset(self):
        exlines = self.__prepare_data__()
        self.dFile = open(self.filepath)
        lines = self.dFile.readlines()
        lines.extend(exlines)
        action_list = []
        previous_action = lines[0].split(',')[1]
        previous_actor = 0
        curr_action = []
        label_list = []

        for li in lines:
            l = li.strip()[:-1]
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
        self.alldata = action_list
        self.alllabel = label_list
        total_len = len(action_list)
        test_len = np.floor(total_len*0.1)
        train_len = total_len - test_len
        self.test_data = list(action_list[(train_len+1).astype(int):])
        self.test_label = list(label_list[(train_len+1).astype(int):])

        self.curr_data = action_list[0]
        self.curr_label = label_list[0]
        self.data_vec = list(action_list[1:(train_len).astype(int)])
        self.label_vec = list(label_list[1:(train_len).astype(int)])
        three_taken = False
        four_taken = False
        five_taken = False
        two_taken = False

        i = 0
        index = len(self.label_vec)

        while(i < index):

            if np.argmax(self.label_vec[i]) == 2 and two_taken == False:
                self.test_data.append(self.data_vec.pop(i))
                self.test_label.append(self.label_vec.pop(i))
                two_taken = True
                index = index - 1

            if np.argmax(self.label_vec[i]) == 3 and three_taken == False:
                self.test_data.append(self.data_vec.pop(i))
                self.test_label.append(self.label_vec.pop(i))
                three_taken = True
                index = index - 1

            if np.argmax(self.label_vec[i]) == 4 and four_taken == False:
                self.test_data.append(self.data_vec.pop(i))
                self.test_label.append(self.label_vec.pop(i))
                four_taken = True
                index = index - 1

            if np.argmax(self.label_vec[i]) == 5 and five_taken == False:
                self.test_data.append(self.data_vec.pop(i))
                self.test_label.append(self.label_vec.pop(i))
                five_taken = True
                index = index - 1

            i = i+1



        print("Current Dataset Contains: " + str(total_len) + " data, the training data has " + str(
            len(self.data_vec)) + " samples.")

    def __data_stats__(self, labelist):
        stats = np.zeros((6,1))
        for label in labelist:
            index = np.argmax(label)
            stats[index] = stats[index] + 1
        return stats



    def pop_data(self):
        data = self.data_vec[0]
        self.data_vec = self.data_vec[1:]
        label = self.label_vec[0]
        self.label_vec = self.label_vec[1:]
        return data, label

    def hasNextClip(self):
        if (len(self.data_vec) != 0):
            return True
        else:
            return False

    def get_test_set(self):
        data = self.test_data
        label = self.test_label
        test_mat = []
        test_label = []
        while(len(data) > 0):
            start_index = 0
            curr_clip = []
            curr_data = data.pop()
            curr_label = label.pop()
            while (start_index + self.window_size < len(curr_data)):
                curr_window = curr_data[start_index:start_index + self.window_size]
                curr_clip.append(curr_window)
                test_label.append(curr_label)
                start_index = start_index + self.step_size
            test_mat.extend(curr_clip)
        return test_mat, test_label


    def genSlidingWindowBatch(self, dataindex):
        dataClip = self.alldata[dataindex]
        label = self.alllabel[dataindex]
        window_data = []
        curr_window = []
        sindex = 0
        eindex = sindex + self.window_size
        while(eindex < len(dataClip)):
            curr_window = dataClip[sindex:eindex]
            window_data.append(curr_window)
            curr_window = []
            sindex = eindex + self.step_size
            eindex = sindex + self.window_size
        return window_data, label


    def genSlidingWindowData(self, data):
        dataClip = data
        window_data = []
        curr_window = []
        sindex = 0
        eindex = sindex + self.window_size
        while(eindex < len(dataClip)):
            curr_window = dataClip[sindex:eindex]
            window_data.append(curr_window)
            curr_window = []
            sindex = eindex + self.step_size
            eindex = sindex + self.window_size
        return window_data




    def sliding_window_batches(self):
        window_data = []
        window_label = []
        curr_window = []
        curr_label = []

        while (len(window_data) < self.batch_size):
            #log_file.write("End Index :" + str(self.end_index) + " Win Size: " + str(self.window_size) + " Step Size: " + str(
            #    step_size) + " Data Length: " + str(len(self.curr_data)) + " Num Clips Left: " + str(len(self.data_vec)) + '\n')
            #If the clip has not come to the end
            if(self.end_index + self.step_size < len(self.curr_data)):
                curr_window = self.curr_data[self.start_index:self.end_index]
                curr_label = self.curr_label
                self.start_index = self.start_index + self.step_size
                self.end_index = self.start_index + self.window_size

            #If the Clip has come to the end, put whatever is left into the batch, amend the insufficent ones by copying the last frame, and start the next one in batch from a new clip
            elif(self.end_index > len(self.curr_data)-1):
                curr_window = self.curr_data[self.start_index:]
                num_frame_left = self.window_size - len(curr_window)
                curr_window = curr_window + [curr_window[-1] for i in range(num_frame_left)]
                curr_label = self.curr_label
                if (self.hasNextClip()):
                    self.curr_data, self.curr_label = self.pop_data()
                    self.start_index = 0
                    self.end_index = self.start_index + self.window_size
                else:
                    self.init_dataset()
                    self.curr_data, self.curr_label = self.pop_data()
                    self.start_index = 0
                    self.end_index = self.start_index + self.window_size

            else:
                curr_window = self.curr_data[self.start_index:(self.start_index + self.window_size)]
                curr_label = self.curr_label
                if (self.hasNextClip()):
                    self.curr_data, self.curr_label = self.pop_data()
                    self.start_index = 0
                    self.end_index = self.start_index + self.window_size
                else:
                    self.init_dataset()
                    self.curr_data, self.curr_label = self.pop_data()
                    self.start_index = 0
                    self.end_index = self.start_index + self.window_size


            window_data.append(curr_window)
            window_label.append(curr_label)
            for a in window_data:
                if (len(a) != self.batch_size):
                    log_file.write(str(len(a)) + '\n')

        return window_data, window_label





