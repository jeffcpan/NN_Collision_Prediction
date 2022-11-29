import torch
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class Nav_Dataset(dataset.Dataset):
    def __init__(self):
        self.data = np.genfromtxt('saved/training_data.csv', delimiter=',')
        # STUDENTS: it may be helpful for the final part to balance the distribution of your collected data

        # normalize data and save scaler for inference
        self.scaler = MinMaxScaler()
        self.normalized_data = self.scaler.fit_transform(self.data) #fits and transforms
        pickle.dump(self.scaler, open("saved/scaler.pkl", "wb")) #save to normalize at inference

    def __len__(self):
    # STUDENTS: __len__() returns the length of the dataset
        return len(self.data)

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            idx = idx.item()

        #print("getItem called")

        return {'input': self.normalized_data[idx][:-1].astype('float32'), 'label': self.normalized_data[idx][-1].astype('float32')}

# STUDENTS: for this example, __getitem__() must return a dict with entries {'input': x, 'label': y}
# x and y should both be of type float32. There are many other ways to do this, but to work with autograding
# please do not deviate from these specifications.


class Data_Loaders():
    def __init__(self, batch_size):
        #print("part1")
        self.nav_dataset = Nav_Dataset()
# STUDENTS: randomly split dataset into two data.DataLoaders, self.train_loader and self.test_loader
# make sure your split can handle an arbitrary number of samples in the dataset as this may vary
        lenFull = len(self.nav_dataset.data)
        lenTrain = int(0.8 * lenFull)
        lenTest = lenFull - lenTrain

        training_data, testing_data = data.random_split(self.nav_dataset, [lenTrain, lenTest])
        self.train_loader = data.DataLoader(training_data)
        self.test_loader = data.DataLoader(testing_data)

#You'll need to create an instance of Nav_Dataset in your DataLoader class first and then split into your train_loader and test_loader.

def main():
    batch_size = 16
    data_loaders = Data_Loaders(batch_size)
    # STUDENTS : note this is how the dataloaders will be iterated over, and cannot be deviated from
    #print("type test: ", type(data_loaders.train_loader))


    #print("navadataset type: ", type(data_loaders.nav_dataset))
    
    for idx, sample in enumerate(data_loaders.train_loader):
        #print("printing train_loader: ", sample)
        _, _ = sample['input'], sample['label']
    for idx, sample in enumerate(data_loaders.test_loader):
        _, _ = sample['input'], sample['label']

if __name__ == '__main__':
    main()
