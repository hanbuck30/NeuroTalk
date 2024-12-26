import csv
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import warnings

epsilon = np.finfo(float).eps

# 경고를 무시하거나 처리하는 방법 설정
def custom_warning_handler(message, category, filename, lineno, file=None, line=None):
    if issubclass(category, RuntimeWarning):
        print(f"RuntimeWarning 발생: {message}")
    else:
        # 다른 경고는 기본 동작을 따릅니다
        warnings.warn(message, category)

# 경고 필터 설정
warnings.showwarning = custom_warning_handler

class myDataset(Dataset):
    def __init__(self, mode, data="./", task = "SpokenEEG", recon="Y_mel"):
        self.sample_rate = 8000
        self.n_classes = 13
        self.mode = mode
        self.iter = iter
        self.savedata = data
        self.task = task
        self.recon = recon
        self.max_audio = 32768.0
        self.ta = self.recon[:-5]
        print(self.savedata + f'/train/{self.ta}Y/')
        self.lenth = len(os.listdir(self.savedata + f'/train/{self.ta}Y/')) #780 # the number data
        self.lenthtest = len(os.listdir(self.savedata + f'/test/{self.ta}Y/')) #260
        self.lenthval = len(os.listdir(self.savedata + f'/val/{self.ta}Y/')) #260
        

    def __len__(self):
        if self.mode == 2:
            return self.lenthval
        elif self.mode == 1:
            return self.lenthtest
        else:
            return self.lenth

    def __getitem__(self, idx):
        '''
        :param idx:
        :return:
        '''

        if self.mode == 2:
            forder_name = self.savedata + '/val/'
        elif self.mode == 1:
            forder_name = self.savedata + '/test/'
        else:
            forder_name = self.savedata + '/train/'
        
        # tasks
        allFileList = os.listdir(forder_name + self.task + "/")
        allFileList.sort()
        # print(allFileList, len(allFileList))
        file_name = forder_name + self.task + '/' + allFileList[idx]
        # print(file_name)
        # if self.task.find('vec') != -1: # embedding vector
        #     input, avg_input, std_input = self.read_vector_data(file_name) 
        if self.task.find('mel') != -1:
            input, avg_input, std_input = self.read_mel_data(file_name)
        elif self.task.find('Voice') != -1: # voice
            input, avg_input, std_input = self.read_voice_data(file_name)
        else: # EEG
            input, avg_input, std_input = self.read_data(file_name) 
            
            
        # recon target
        allFileList = os.listdir(forder_name + self.recon + "/")
        allFileList.sort()
        file_name = forder_name + self.recon + '/' + allFileList[idx]
        
        # if self.recon.find('vec') != -1: # embedding vector
        #     target, avg_target, std_target = self.read_vector_data(file_name) 
        if self.recon.find('mel') != -1:
            target, avg_target, std_target = self.read_mel_data(file_name)
        elif self.recon.find('Voice') != -1: # voice
            target, avg_target, std_target = self.read_voice_data(file_name)
        else: # EEG
            target, avg_target, std_target = self.read_data(file_name)
        
        # voice
        #allFileList = os.listdir(forder_name + "Voice/")
        #allFileList.sort()
        #file_name = forder_name + "Voice/"+ allFileList[idx]
        #voice, _, _ = self.read_voice_data(file_name)
        # voice=[]
        # target label
        allFileList = os.listdir(forder_name + f"{self.ta}Y/")
        allFileList.sort()
        file_name = forder_name + f'{self.ta}Y/' + allFileList[idx]
        # print(file_name)
        target_cl,_,_ = self.read_raw_data(file_name) 
        target_cl = np.squeeze(target_cl)


        # to tensor
        input = torch.tensor(input, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)
        

        #return input, target, target_cl, voice, (avg_target, std_target, avg_input, std_input)
        return input, target, target_cl, (avg_target, std_target, avg_input, std_input)

   
    def read_vector_data(self, file_name,n_classes):
        with open(file_name, 'r', newline='') as f:
            lines = csv.reader(f)
            data = []
            for line in lines:
                data.append(line)

        data = np.array(data).astype(np.float32)
        (r,c) = data.shape
        data = np.reshape(data,(n_classes,r//n_classes,c))
        
        max_ = np.max(data).astype(np.float32)
        min_ = np.min(data).astype(np.float32)
        avg = (max_ + min_) / 2
        std = (max_ - min_) / 2
        data   = np.array((data - avg) / std).astype(np.float32)

        return data, avg, std
    
    
    def read_voice_data(self, file_name):
        with open(file_name, 'r', newline='') as f:
            lines = csv.reader(f)
            data = []
            for line in lines:
                data.append(line)
        data = np.array(data).astype(np.float32)
        
        data = np.array(data / self.max_audio).astype(np.float32)
        avg = np.array([0]).astype(np.float32)

        return data, avg, self.max_audio


    def read_data(self, file_name):
        with open(file_name, 'r', newline='') as f:
            lines = csv.reader(f)
            data = []
            for line in lines:
                data.append(line)

        data = np.array(data).astype(np.float32)
        
        max_ = np.max(data).astype(np.float32)
        min_ = np.min(data).astype(np.float32)
        
        avg = (max_ + min_) / 2
        std = (max_ - min_) / 2
        data = np.array((data - avg) / std).astype(np.float32)
        
        return data, avg, std

    def read_mel_data(self, file_name):
        with open(file_name, 'r', newline='') as f:
            lines = csv.reader(f)
            data = []
            for line in lines:
                data.append(line)

        data = np.array(data).astype(np.float32)
        
        max_ = np.max(data).astype(np.float32)
        min_ = np.min(data).astype(np.float32)
        
        avg = (max_ + min_) / 2
        std = (max_ - min_) / 2
        #data = np.array((data - avg) / std).astype(np.float32)
        
        return data, avg, std




    def read_raw_data(self, file_name):
        with open(file_name, 'r', newline='') as f:
            lines = csv.reader(f)
            data = []
            for line in lines:
                data.append(line)

        data = np.array(data).astype(np.float32)
        avg = np.array([0]).astype(np.float32)
        std = np.array([1]).astype(np.float32)
            
        return data, avg, std


