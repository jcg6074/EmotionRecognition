import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import argparse

from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score

from src.datasets_iv import SeedDatasetV3 as Dataset
from src.models_v5d2_iv import FastSlow as Model
from src.utils import fix_random_seeds, LabelSmoothingLoss

#from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser()
parser.add_argument("--experiment", type=str, default="seed_iv_independent")
parser.add_argument("--subjects", type=int, default=15)
parser.add_argument("--classes", type=int, default=4)
parser.add_argument("--model", type=str, default="FastSlow")
parser.add_argument("--batch_size", type=float, default=8)
parser.add_argument("--epochs", type=float, default=100)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--lr_min", type=float, default=1e-6)
parser.add_argument("--smoothing", type=float, default=0.1)
parser.add_argument("--weight_decay", type=float, default=0.05)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--checkpoint_dir", type=str, default="checkpoint")
args = parser.parse_args()


def train(model, loader, criterion, optimizer, scheduler):
    model.train()
    loss_avg = 0.0
    for i, (input, position_embedding, mask, label) in enumerate(loader): 
        
        optimizer.zero_grad()
        
        # args.device : "cuda:1"
        input = input.to(args.device)
        position_embedding = position_embedding.to(args.device)
        mask = mask.to(args.device)
        label = label.to(args.device)
        # print(input.shape)
        # print(position_embedding.shape)
        # print(mask.shape)
        vid_pred = model(input, position_embedding, mask)
        
        loss = criterion(vid_pred, label)

        loss.backward()
        loss_avg += loss.item()

        optimizer.step() 
        scheduler.step()
    

    return loss_avg / (i + 1)  


def evaluate(model, loader):
    model.eval()
    predictions = []
    labels = [] 
    with torch.no_grad():
        for input, position_embedding, mask, label in loader:
            input = input.to(args.device)
            position_embedding = position_embedding.to(args.device)
            mask = mask.to(args.device)
            vid_pred = model(input, position_embedding, mask)
            prediction = F.softmax(vid_pred, dim=-1)
            prediction = np.squeeze(prediction.detach().to("cpu").numpy())
            predictions.append(prediction)
            label = label.numpy()
            labels.append(label)
            
    predictions = np.vstack(predictions)
    labels = np.vstack(labels)

    predictions = np.argmax(predictions, axis = -1)
    labels = np.argmax(labels, axis = -1)
    accuracy = accuracy_score(labels, predictions)

    # labels, predictions는 따로 붙임
    return accuracy, labels, predictions



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


def main():
    fix_random_seeds(0)

    logs = []
    for subject_id in range(0, args.subjects):       
        train_dataset = Dataset(f"./data/{args.experiment}/id{subject_id}_train.pt")
        test_dataset = Dataset(f"./data/{args.experiment}/id{subject_id}_test.pt")
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1)
        
        model = Model()
        model = model.to(args.device)

        optimizer = AdamW(model.parameters(), args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, len(train_loader) * args.epochs, args.lr_min)
        criterion = LabelSmoothingLoss(4, smoothing=args.smoothing)
        # criterion = nn.CrossEntropyLoss()
        
        # Train Loop
        best_accuracy = 0.0
        for epoch in range(args.epochs):
            loss = train(model, train_loader, criterion, optimizer, scheduler)
            train_accuracy, train_label, train_pred = evaluate(model, train_loader)
            test_accuracy, test_label, test_pred = evaluate(model, test_loader)
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
            
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                save_dict["best_accuracy"] = best_accuracy
                checkpoint_path = f"{args.checkpoint_dir}/seed_iv/id{subject_id}-best.pth"
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
                  f"Best Acc: {best_accuracy:.4f}")


        logs.append([best_accuracy, test_accuracy])

        with open(f"{args.checkpoint_dir}/seed_iv/log.txt", "wt") as f:
            for log in logs:
                f.writelines(f"{log[0]:.4f}, {log[1]:.4f}\n")


if __name__=="__main__":
    main()