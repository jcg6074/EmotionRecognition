import random
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import OneHotEncoder
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
    def __init__(self, dataset, training=True, target=None, emotion=None):    
        self.frames = 394   # 변경 265/ 64           
        self.dim_embed = 64         
        self.channels = 14
        self.training = training  
        self.target = target
        self.emotion = emotion

        self.dataset = torch.load(dataset)  
        self.frame_encoding_table = torch.FloatTensor(sinusoid_encoding_table(self.frames + 1, self.dim_embed))
        self.frame_encodding = nn.Embedding.from_pretrained(self.frame_encoding_table, freeze=True)

        
    def __len__(self):
        return len(self.dataset["input"])
    
    def __getitem__(self, idx):
        inputs = self.dataset["input"][idx, ...]
        labels = self.dataset["label"][idx, ...]
        masks = self.dataset["mask"][idx, ...]
        inputs = torch.tensor(inputs, dtype=torch.float32)
        inputs = inputs.permute(1,0,2)
        masks = torch.tensor(masks, dtype=torch.float32)
        targets = np.zeros([394,15,5])
        if self.emotion is not None:
            if self.emotion == "valence":
                labels = labels[0]
            elif self.emotion == "arousal":
                labels = labels[1]
            elif self.emotion == "dominance":
                labels = labels[2]
            
            if labels > 3:
                labels = 1
            else:
                labels = 0
            labels = torch.tensor(labels, dtype=torch.int64)
        labels = nn.functional.one_hot(labels.to(torch.int64), num_classes = 2)
        lab = torch.argmax(labels)
        origin_input = inputs

        cls_token = labels.unsqueeze(dim=-1)
        cls_token = cls_token.unsqueeze(dim=-1).repeat(1,394,5).reshape(394,2,5)
        origin_input = torch.cat((cls_token,origin_input),dim=1)
        af_mask = torch.zeros(394,16,5)
        target_mask = torch.zeros(394,14,5)
        if self.training:
            T_max = masks.sum()
            T = random.randint(0, int(0.1 * T_max))
            N = random.randint(0, int(self.channels-4))
            t1 = random.randint(0, T_max - T)
            t2 = t1 + T
            n1 = random.randint(0, self.channels - N)
            n2 = N + 3
            inputs[t1:t2, n1:n2, :] = 0.0
            af_mask[t1:t2, n1+2:n2+2, :] = 1.0
            if self.target is not None:
                if lab == 0:
                    targets = self.target[0]
                elif lab == 1:
                    targets = self.target[1]


        #targets[:,:1,:] = lab+1
        targets = torch.tensor(targets, dtype=torch.float32)
        af_mask[:,:2,:] = 1.0
        frame_pos_masks = masks.eq(0)
        frame_pos = torch.arange(inputs.size(0)) + 1
        frame_pos = frame_pos.masked_fill(frame_pos_masks, 0)
        frame_pos_embeddings = self.frame_encodding(frame_pos)
        frame_pos_embeddings = frame_pos_embeddings.reshape(self.frames, 1, self.dim_embed)
        frame_pos_embeddings = frame_pos_embeddings.expand(self.frames, self.channels, self.dim_embed)
        
        return inputs, frame_pos_embeddings, masks, labels, targets, origin_input, af_mask, target_mask


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
    