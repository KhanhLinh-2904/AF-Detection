import scipy.io
import os
import csv
import heartpy as hp
import numpy as np

class Data_Parser:
    def __init__(self, data_path, sample_rate):
        self.sample_rate = sample_rate
        self.data_path = data_path

    def read_data(self):
        all_files = os.listdir(self.data_path)
        #key "A00002" value ".mat"
        trainSet_dict, labels_dict = dict(), dict()
        #key "A00002" value "AF"
        trainingSet, labels = list(),list()

        for felement in all_files:
            full_path = os.path.join(self.data_path,felement)
            #Taking ECG data from .mat
            if ".mat" in felement:
                #Using scipy.io to read .mat and store data in variable mat_data
                mat_data = scipy.io.loadmat(full_path)
                #Taking name "A00002.mat"
                file_name = full_path.split('/')[-1]
                #Taking name "A00002"
                file_name = file_name.split('.')[0]
                #Store value ECG
                trainSet_dict[file_name] = mat_data['val'][0]

            #Open REFERENCE.csv reading labels
            if "REFERENCE.csv" == felement:
                with open(full_path, newline='') as csvfile:
                    # Specify data by "space"
                    linereader = csv.reader(csvfile, delimiter=' ', quotechar='|')
                    for index, line in enumerate(linereader):
                        line = line[0].split(',')
                        file_name = line[0]
                        label = line[1]
                        labels_dict[file_name] = label
        # convert dict to list
        for file_key in trainSet_dict:
            trainingSet.append(trainSet_dict[file_key])
            labels.append(labels_dict[file_key])
        #convert to numpy
        np_trainingSet = np.asarray(trainingSet, dtype=object)
        # print("np_trainingSet: ", np_trainingSet)
        np_label = np.asarray(labels)
        # print("np_label: ", np_label)
        return np_trainingSet, np_label

    def create_datasets(self, trainingSet, labels):
        #classify data into 3 types: AF, normal, other
        normal_dataset, normal_labels = list(), list()
        atrfib_dataset, atrfib_labels = list(), list()
        other_dataset, other_labels = list(), list()

        for index, label in enumerate(labels):
            if label == 'A':
                atrfib_dataset.append(trainingSet[index])
                atrfib_labels.append(1)
            if label == 'N':
                normal_dataset.append(trainingSet[index])
                normal_labels.append(0)
            if label == 'O':
                other_dataset.append(trainingSet[index])
                other_labels.append(0)
                
        ar_atrfib_dataset = np.asarray(atrfib_dataset, dtype=object)
        ar_atrfib_labels = np.asarray(atrfib_labels)
        ar_normal_dataset = np.asarray(normal_dataset, dtype=object)
        ar_normal_labels = np.asarray(normal_labels)
        ar_other_dataset = np.asarray(other_dataset, dtype=object)
        ar_other_labels = np.asarray(other_labels)

        return ar_atrfib_dataset, ar_atrfib_labels, ar_normal_dataset, ar_normal_labels, ar_other_dataset, ar_other_labels

    def read_training_dataset(self):
        #If folder has no "Training" folder which stores all preprocessed ECG data
        # Create one
        if not os.path.exists("Training"):
            os.makedirs("Training")
            data, labels = self.read_data()
            atrfib_dataset, atrfib_labels, normal_dataset, normal_labels, other_dataset, other_labels = self.create_datasets(data, labels)
           
            np.savez("Training/atrfib_dataset.npz", *atrfib_dataset)
            np.savez("Training/normal_dataset.npz", *normal_dataset)
            np.savez("Training/other_dataset.npz", *other_dataset)

            np.savez("Training/atrfib_labels.npz", *atrfib_labels)
            np.savez("Training/normal_labels.npz", *normal_labels)
            np.savez("Training/other_labels.npz", *other_labels)

            return atrfib_dataset, atrfib_labels, normal_dataset, normal_labels, other_dataset, other_labels
        else:
            container = np.load("Training/atrfib_dataset.npz")
            atrfib_dataset = [container[key] for key in container]
            atrfib_dataset = np.asarray(atrfib_dataset, dtype=object)
            print("len atrfib_dataset: ", len(atrfib_dataset))
            

            container = np.load("Training/normal_dataset.npz")
            normal_dataset = [container[key] for key in container]
            normal_dataset = np.asarray(normal_dataset, dtype=object)
            print("len normal_dataset: ", len(normal_dataset))
            

            container = np.load("Training/other_dataset.npz")
            other_dataset = [container[key] for key in container]
            other_dataset = np.asarray(other_dataset, dtype=object)

            container = np.load("Training/atrfib_labels.npz")
            atrfib_labels = [container[key] for key in container]
            atrfib_labels = np.asarray(atrfib_labels)
            print("len other_dataset: ", len(other_dataset))
            

            container = np.load("Training/normal_labels.npz")
            normal_labels = [container[key] for key in container]
            normal_labels = np.asarray(normal_labels)

            container = np.load("Training/other_labels.npz")
            other_labels = [container[key] for key in container]
            other_labels = np.asarray(other_labels)

            return atrfib_dataset, atrfib_labels, normal_dataset, normal_labels, other_dataset, other_labels
    
    def read_testing_dataset(self):
        #If folder has no "Testing" folder which stores all preprocessed ECG data
        # Create one
        if not os.path.exists("Testing"):
            os.makedirs("Testing")
            data, labels = self.read_data()
            atrfib_dataset, atrfib_labels, normal_dataset, normal_labels, other_dataset, other_labels = self.create_datasets(data, labels)
           
            np.savez("Testing/atrfib_dataset.npz", *atrfib_dataset)
            np.savez("Testing/normal_dataset.npz", *normal_dataset)
            np.savez("Testing/other_dataset.npz", *other_dataset)

            np.savez("Testing/atrfib_labels.npz", *atrfib_labels)
            np.savez("Testing/normal_labels.npz", *normal_labels)
            np.savez("Testing/other_labels.npz", *other_labels)

            return atrfib_dataset, atrfib_labels, normal_dataset, normal_labels, other_dataset, other_labels
        else:
            container = np.load("Testing/atrfib_dataset.npz")
            atrfib_dataset = [container[key] for key in container]
            atrfib_dataset = np.asarray(atrfib_dataset, dtype=object)
            print("len atrfib_dataset: ", len(atrfib_dataset))
            

            container = np.load("Testing/normal_dataset.npz")
            normal_dataset = [container[key] for key in container]
            normal_dataset = np.asarray(normal_dataset, dtype=object)
            print("len normal_dataset: ", len(normal_dataset))
            

            container = np.load("Testing/other_dataset.npz")
            other_dataset = [container[key] for key in container]
            other_dataset = np.asarray(other_dataset, dtype=object)
            print("len other_dataset: ", len(other_dataset))
            

            container = np.load("Testing/atrfib_labels.npz")
            atrfib_labels = [container[key] for key in container]
            atrfib_labels = np.asarray(atrfib_labels)

            container = np.load("Testing/normal_labels.npz")
            normal_labels = [container[key] for key in container]
            normal_labels = np.asarray(normal_labels)

            container = np.load("Testing/other_labels.npz")
            other_labels = [container[key] for key in container]
            other_labels = np.asarray(other_labels)

            return atrfib_dataset, atrfib_labels, normal_dataset, normal_labels, other_dataset, other_labels