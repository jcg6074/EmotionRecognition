import random
import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import Dataset


def sinusoid_encoding_table(n_seq, d_hidn):
    
    def cal_angle(position, i_hidn):
        return position / np.power(10000, 2 * (i_hidn // 2) / d_hidn)
    
    def get_posi_angle_vec(position):
        return [cal_angle(position, i_hidn) for i_hidn in range(d_hidn)]

    sinusoid_table = np.array([get_posi_angle_vec(i_seq) for i_seq in range(n_seq)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # even index (sin) 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # odd index (cos)
    return sinusoid_table


class SeedDataset(Dataset):
    def __init__(self, dataset):    
        self.frames = 64           
        self.dim_embed = 64         
        self.channels = 62          

        self.dataset = torch.load(dataset)  
        self.frame_encoding_table = torch.FloatTensor(sinusoid_encoding_table(self.frames + 1, self.dim_embed))
        self.frame_encodding = nn.Embedding.from_pretrained(self.frame_encoding_table, freeze=True)

        
    def __len__(self):
        return len(self.dataset["input"])
    
    def __getitem__(self, idx):
        inputs = self.dataset["input"][idx, ...]
        labels = self.dataset["label"][idx, ...]
        masks = self.dataset["mask"][idx, ...]
        
        frame_pos_masks = masks.eq(0)
        frame_pos = torch.arange(inputs.size(0)) + 1
        frame_pos = frame_pos.masked_fill(frame_pos_masks, 0)
        frame_pos_embeddings = self.frame_encodding(frame_pos)
        frame_pos_embeddings = frame_pos_embeddings.reshape(self.frames, 1, self.dim_embed)
        frame_pos_embeddings = frame_pos_embeddings.expand(self.frames, self.channels, self.dim_embed)
        
        return inputs, frame_pos_embeddings, labels


class SeedDatasetV3(Dataset):
    def __init__(self, dataset, training=True):    
        self.frames = 64   # 변경 265/ 64           
        self.dim_embed = 64         
        self.channels = 62
        self.training = training  

        self.dataset = torch.load(dataset)  
        self.frame_encoding_table = torch.FloatTensor(sinusoid_encoding_table(self.frames + 1, self.dim_embed))
        self.frame_encodding = nn.Embedding.from_pretrained(self.frame_encoding_table, freeze=True)

        
    def __len__(self):
        return len(self.dataset["input"])
    
    def __getitem__(self, idx):
        inputs = self.dataset["input"][idx, ...]
        labels = self.dataset["label"][idx, ...]
        masks = self.dataset["mask"][idx, ...]

        if self.training:
            T_max = masks.sum()
            T = random.randint(0, int(0.1 * T_max))
            N = random.randint(0, int(0.1 * self.channels))
            t1 = random.randint(0, T_max - T)
            t2 = t1 + T
            n1 = random.randint(0, self.channels - N)
            n2 = n1 + N
            inputs[t1:t2, n1:n2, :] = 0.0

        frame_pos_masks = masks.eq(0)
        frame_pos = torch.arange(inputs.size(0)) + 1
        frame_pos = frame_pos.masked_fill(frame_pos_masks, 0)
        frame_pos_embeddings = self.frame_encodding(frame_pos)
        frame_pos_embeddings = frame_pos_embeddings.reshape(self.frames, 1, self.dim_embed)
        frame_pos_embeddings = frame_pos_embeddings.expand(self.frames, self.channels, self.dim_embed)
        
        return inputs, frame_pos_embeddings, masks, labels


class SeedDatasetV2(Dataset):
    def __init__(self, dataset, frames=265, channels=62, dim=128, training=True):    
        self.frames = frames
        self.max_frames = 265        
        self.dim = dim         
        self.channels = channels
        self.training = training     

        self.dataset = torch.load(dataset)  
        self.frame_encoding_table = torch.FloatTensor(sinusoid_encoding_table(self.frames + 1, self.dim))
        self.frame_encodding = nn.Embedding.from_pretrained(self.frame_encoding_table, freeze=True)

        
    def __len__(self):
        return len(self.dataset["input"])
    
    def __getitem__(self, idx):
        inputs = self.dataset["input"][idx, ...]
        labels = self.dataset["label"][idx, ...]
        masks = self.dataset["mask"][idx, ...]
        # print(labels)
        
            
        if self.training:
            T = inputs.shape[0]
            t1 = random.randint(0, T - self.frames)
            t2 = t1 + self.frames
            inputs = inputs[t1:t2, ...]
            # frame_pos_embeddings = frame_pos_embeddings[t1:t2, ...]
            # labels = labels[t1:t2, ...]
            masks = masks[t1:t2, ...]
            # print(t1, t2, inputs.shape, labels.shape, masks.shape)

        frame_pos_masks = masks.eq(0)
        frame_pos = torch.arange(inputs.size(0)) + 1
        frame_pos = frame_pos.masked_fill(frame_pos_masks, 0)
        frame_pos_embeddings = self.frame_encodding(frame_pos)
        frame_pos_embeddings = frame_pos_embeddings.reshape(self.frames, 1, self.dim)
        frame_pos_embeddings = frame_pos_embeddings.expand(self.frames, self.channels, self.dim)
        
        return inputs, frame_pos_embeddings, labels
    