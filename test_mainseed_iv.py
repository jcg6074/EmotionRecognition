import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import argparse

from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score

from src.test_dataset_iv import SeedDatasetV3 as Dataset
from src.test_seediv import FastSlow as Model
from src.utils import fix_random_seeds, LabelSmoothingLoss
from scipy.io import savemat
#from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from collections import Counter
import scipy.io as sio

parser = argparse.ArgumentParser()
parser.add_argument("--experiment", type=str, default="seed_iv_independent")
parser.add_argument("--subjects", type=int, default=15)
parser.add_argument("--classes", type=int, default=4)
parser.add_argument("--model", type=str, default="FastSlow")
parser.add_argument("--batch_size", type=float, default=8)
parser.add_argument("--epochs", type=float, default=300)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--lr_min", type=float, default=1e-6)
parser.add_argument("--smoothing", type=float, default=0.1)
parser.add_argument("--weight_decay", type=float, default=0.05)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--checkpoint_dir", type=str, default="checkpoint")
args = parser.parse_args()


def train(model, loader, criterion, mse, optimizer, scheduler, phase):
    model.train()
    loss_avg = 0.0
    for i, (input, position_embedding, mask, label, target, origin_input, af_mask, target_mask) in enumerate(loader): 
        optimizer.zero_grad()
        # args.device : "cuda:1"
        input = input.to(args.device)
        position_embedding = position_embedding.to(args.device)
        mask = mask.to(args.device)
        label = label.to(args.device)
        origin_input = origin_input.to(args.device)
        af_mask = af_mask.to(args.device)
        target = target.to(args.device)
        # print(position_embedding.shape)
        y,recon = model(input, position_embedding, mask)
        if phase == 0:
            loss = criterion(y, label)
        if phase == 1:
            #loss2 = (recon-origin_input)**2
            #loss2 = (loss2*af_mask).sum() / af_mask.sum()ps 
            loss = criterion(y, label) + mse(recon*af_mask, origin_input*af_mask)
        elif phase == 2:
            loss = criterion(y, label)*0.01 + mse(recon*af_mask, target*af_mask)
        loss.backward()
        loss_avg += loss.item()

        optimizer.step() 
        scheduler.step()
    

    return loss_avg / (i + 1)  


def evaluate(model, loader):
    model.eval()
    predictions = []
    labels = [] 
    recons = []
    with torch.no_grad():
        for input, position_embedding, mask, label, target, origin_input, af_mask, target_mask in loader:
            input = input.to(args.device)
            position_embedding = position_embedding.to(args.device)
            mask = mask.to(args.device)
            vid_pred,recon = model(input, position_embedding, mask)
            prediction = F.softmax(vid_pred, dim=-1)
            prediction = np.squeeze(prediction.detach().to("cpu").numpy())
            predictions.append(prediction)
            label = label.numpy()
            recon = recon.detach().to("cpu").numpy()
            labels.append(label)
            recons.append(recon)
            
    predictions = np.vstack(predictions)
    labels = np.vstack(labels)
    recons = np.vstack(recons)

    predictions = np.argmax(predictions, axis = -1)
    labels = np.argmax(labels, axis = -1)
    accuracy = accuracy_score(labels, predictions)

    # labels, predictions는 따로 붙임
    return accuracy, labels, predictions,recons



# class FramewiseCELoss(torch.nn.Module):
#     def __init__(self, classes, smoothing=0.0, dim=-1):
#         super(FramewiseCELoss, self).__init__()
#         self.confidence = 1.0 - smoothing
#         self.smoothing = smoothing
#         self.classes = classes
#         self.dim = dim

#     def forward(self, outputs, targets):
#         B, T, C = outputs.shape
#         targets = targets.reshape(B, 1, C)
#         targets = torch.tile(targets, (1, T, 1))
#         outputs = outputs.reshape(B * T, C)
#         targets = targets.reshape(B * T, C)

#         outputs = outputs.log_softmax(dim=self.dim)
#         with torch.no_grad():
#             indicator = 1.0 - targets
#             smooth_targets = torch.zeros_like(targets)
#             smooth_targets.fill_(self.smoothing / (self.classes - 1))
#             smooth_targets = targets * self.confidence + indicator * smooth_targets
        
#         return torch.mean(torch.sum(-smooth_targets * outputs, dim=self.dim))


def main(phase=0, topk = None, target=None, subject_id=None, var = None):
    fix_random_seeds(0)

    logs = []
    for subject_id in range(0, args.subjects):  
        if phase==2:
            f = sio.loadmat("data/target/targetdata(" + str(subject_id) + ").mat")
            # f = sio.loadmat("data/middlespot(2).mat")
            data1 = np.array(f['class1'])
            data2 = np.array(f['class2'])
            data3 = np.array(f['class3'])
            data4 = np.array(f['class4'])
            T,C,B = data1.shape
            target = np.zeros((4, T, C, B))
            target[0] = data1
            target[1] = data2
            target[2] = data3
            target[3] = data4

        train_dataset = Dataset(f"./data/{args.experiment}/id{subject_id}_train.pt", target=target, variance = var)
        test_dataset = Dataset(f"./data/{args.experiment}/id{subject_id}_test.pt",training=False,target=target, variance = var)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1)

        model = Model(topk=topk)
        model = model.to(args.device)

        optimizer = AdamW(model.parameters(), args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, len(train_loader) * args.epochs, args.lr_min)
        criterion = LabelSmoothingLoss(4, smoothing=args.smoothing)
        mse = torch.nn.MSELoss()
        # criterion = nn.CrossEntropyLoss()
        train_recons=[]
        test_recons=[]
        train_pred = []
        # Train Loop
        best_accuracy = 0.0

        for epoch in range(args.epochs):
            train_dataset = Dataset(f"./data/{args.experiment}/id{subject_id}_train.pt", target=target, variance = var)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            loss = train(model, train_loader, criterion, mse, optimizer, scheduler, phase)
            train_accuracy, train_label, train_pred,train_recons = evaluate(model, train_loader)
            test_accuracy, test_label, test_pred,_ = evaluate(model, test_loader)
            lr = scheduler.get_last_lr()[0]
            # save
            save_dict = {
                "epoch": epoch + 1,
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "best_accuracy": best_accuracy,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            if phase == 1 and (train_accuracy>=1.0 or (epoch+1) == args.epochs):
                y = train_pred
                data1 = train_recons[np.squeeze(y == 0)]
                data2 = train_recons[np.squeeze(y == 1)]
                data3 = train_recons[np.squeeze(y == 2)]
                data4 = train_recons[np.squeeze(y == 3)]

                data1 = np.where(data1 == 0, np.nan, data1)
                data2 = np.where(data2 == 0, np.nan, data2)
                data3 = np.where(data3 == 0, np.nan, data3)
                data4 = np.where(data4 == 0, np.nan, data4)

                data1 = np.nanmean(data1, axis=0)
                data2 = np.nanmean(data2, axis=0)
                data3 = np.nanmean(data3, axis=0)
                data4 = np.nanmean(data4, axis=0)
                plt.figure(figsize=(15, 15))
                plt.plot(data1[:,:,0])
                # plt.legend(loc='lower right')
                plt.savefig("class1.jpg")
                plt.clf()
                plt.plot(data2[:,:,0])
                # plt.legend(loc='lower right')
                plt.savefig("class2.jpg")
                plt.clf()
                plt.plot(data3[:,:,0])
                # plt.legend(loc='lower right')
                plt.savefig("class3.jpg")
                plt.clf()
                plt.plot(data4[:,:,0])
                # plt.legend(loc='lower right')
                plt.savefig("class4.jpg")
                plt.clf()

                mat_dic = {"class1": data1, "class2": data2, "class3": data3, "class4": data4}
                savemat("data/target/targetdata(" + str(subject_id) + ").mat", mat_dic)
                break

            
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                save_dict["best_accuracy"] = best_accuracy
                checkpoint_path = f"{args.checkpoint_dir}/seed_iv/id{subject_id}phase{phase}-var{var}best.pth"
                torch.save(save_dict, checkpoint_path)

                # confusion matrix
                # fig, ax = plt.subplots(figsize=(3,3))
                # plot = confusion_matrix(test_label, test_pred)
                # #print(test_label)
                # #print(test_pred)
                # f = sns.heatmap(plot, annot=True, fmt='.2f', cmap='Blues')
                # plt.savefig(f'./plot/seed_iv_trial1/subject_{subject_id}_acc_{best_accuracy:.4f}.png')

            # print
            print(f"Subject: {subject_id}, " +
                    f"Epoch: {epoch + 1:03d}, " + 
                    f"Train Acc: {train_accuracy:.4f}, " +
                    f"Test Acc: {test_accuracy:.4f}, " +
                    f"Best Acc: {best_accuracy:.4f}, "+
                    f"loss: {loss:.4f}")
            if phase == 2 and test_accuracy >= 1.0:
                break
        logs.append([best_accuracy, test_accuracy])
        with open(f"{args.checkpoint_dir}/seed_iv/log{phase}-var{var}.txt", "wt") as f:
            for log in logs:
                f.writelines(f"{log[0]:.4f}, {log[1]:.4f}\n")


if __name__=="__main__":
    #variance = [0,0.5,1,3,5]
    for k in range(1):
        #main(phase=0, topk=4, var= 0)

        main(phase=2, topk=4, var= 0)