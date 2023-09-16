import os
import torch
import threading
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

import config as cfg

class MAIC(Dataset):
    def __init__(self, dtypes:list, fold:int, is_train:bool):
        self.data = []
        self.is_train = is_train
        self.fold = fold
        for dtype in dtypes:
            df           = pd.read_csv(cfg.CSV_PATH[dtype])
            self.path    = cfg.PATHS[dtype]
            self.files   = df['FILENAME'].to_list()
            self.genders = np.array([cfg.GENDER[g] for g in df["GENDER"]]).astype(np.float32)
            self.ages    = df["AGE"].to_numpy().astype(np.float32)
            self.folds   = df["FOLD"].tolist()
            self.load_data_in_parallel()
        print(f"{len(self.data)} DATA LOADED!")
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data_item  = self.data[idx]
        return data_item
        
    def load_data(self, idx):
        if self.is_train:
            if self.folds[idx] != self.fold:
                file_name = self.files[idx]
                gender = self.genders[idx]
                age = self.ages[idx]
                data = {'data': None, 'gender': gender, 'age': age}
                data['data'] = torch.FloatTensor(np.load(f'{self.path}/{file_name}.npy').reshape(12, -1))
                self.data.append(data)
        else:
            if self.folds[idx] == self.fold:
                file_name = self.files[idx]
                gender = self.genders[idx]
                age = self.ages[idx]
                data = {'data': None, 'gender': gender, 'age': age}
                data['data'] = torch.FloatTensor(np.load(f'{self.path}/{file_name}.npy').reshape(12, -1))
                self.data.append(data)
                
    def load_data_in_parallel(self):
        threads = []
        for idx in tqdm(range(len(self.files))):
            thread = threading.Thread(target=self.load_data, args=(idx,))
            thread.start()
            threads.append(thread)
        
        for thread in threads:
            thread.join()
    
if __name__ == '__main__':
    dataset = MAIC(cfg.DTYPES['whole'])
    dataloader = DataLoader(dataset, 32, shuffle=False, drop_last=False)
    for data in tqdm(dataloader):
        print(data['data'].shape)
        print(data['age'].shape)
        print(data['gender'].shape)
        break
    # print(dataset.files)
