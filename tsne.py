import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import argparse
from sklearn.manifold import TSNE
from collections import OrderedDict

from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score

#from src.datasets import SeedDatasetV3 as Dataset
#from src.models_v5d2 import FastSlow as Model #src.models_v5d2.py

#from src.test_dataset_iv import SeedDatasetV3 as Dataset
#from src.test_seediv import FastSlow as Model

from src.datasets_dreamer import SeedDatasetV3 as Dataset
from src.model_dreamer import FastSlow as Model

from src.utils import fix_random_seeds, LabelSmoothingLoss

#from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import scipy.io as sio
from scipy.io import savemat

parser = argparse.ArgumentParser()
parser.add_argument("--experiment", type=str, default="Dreamer_independent")
parser.add_argument("--subjects", type=int, default=23)
parser.add_argument("--classes", type=int, default=2)
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

def tsne(data,label,subject_id):
    tsne_np = TSNE(n_components=2, perplexity=30, learning_rate=200,random_state=22).fit_transform(data)
    class1 = tsne_np[np.squeeze(label == 0)]
    class2 = tsne_np[np.squeeze(label == 1)]
    class3 = tsne_np[np.squeeze(label == 2)]

    plt.figure(figsize=(15, 15))
    plt.scatter(class1[:, 0], class1[:, 1], label='left hand', c='r')
    plt.scatter(class2[:, 0], class2[:, 1], label='right hand', c='g')
    plt.scatter(class3[:, 0], class3[:, 1], label='feet', c='b')

    plt.gca().axes.yaxis.set_visible(False)
    plt.gca().axes.xaxis.set_visible(False)
    plt.gca().spines['right'].set_visible(False)  # 오른쪽 테두리 제거
    plt.gca().spines['top'].set_visible(False)  # 위 테두리 제거
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.savefig("tsne(2phase)"+str(subject_id)+".png",bbox_inches='tight')
    plt.clf()


def evaluate(model, loader,subject_id):
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
    return accuracy, labels, predictions, recons

fix_random_seeds(0)
pred = []
label = []
for subject_id in range(args.subjects):
    model = Model(topk=4)
    ckpt = torch.load(f"checkpoint/Dreamer/top/id{subject_id}phase2-topk4-valencebest.pth")["model"]
    tmp_dict = OrderedDict()
    for i, j in ckpt.items():   # 가중치의 모든 키 값 반복문
        name = i.replace("decoder","")  # 매치되지 않는 키 값 변경
        tmp_dict[name] = j
    model.load_state_dict(tmp_dict,strict=False)
    model = model.to(args.device)
    test_dataset = Dataset(f"./data/{args.experiment}/id{subject_id}_test.pt" ,training=False,emotion="valence")
    test_loader = DataLoader(test_dataset, batch_size=1)

    test_accuracy, test_label, test_pred,_ = evaluate(model, test_loader, subject_id)
    #print(test_accuracy)

    pred.extend(test_pred)
    label.extend(test_label)
    #confusion matrix
fig, ax = plt.subplots(figsize=(5,5))
plot = confusion_matrix(label, pred)
ck = np.zeros([2,2])
print(plot)
ck[0] = plot[0]/251*100
ck[1] = plot[1]/163*100
#ck[2] = plot[2]/270*100
#ck[3] = plot[3]/270*100

print(ck)
#labels = ["Neutral","Sad","Fear","Happy"]
#labels = ["Negative","Neutral","Positive"]
labels = ["Low","High"]

marks = np.arange(len(labels))

#f = sns.heatmap(ck, annot=True, fmt='.2f', cmap='Blues', square=False, annot_kws={'size': 15}, cbar=True)
f = ConfusionMatrixDisplay(confusion_matrix=ck, display_labels=labels)
rcParams['font.weight']='bold'
plt.rc('font', size=15)
f.plot(values_format=".2f", cmap='Blues', xticks_rotation='horizontal')
plt.xlabel("")
plt.ylabel("")
plt.xticks(marks, labels, fontsize=13)
plt.yticks(marks, labels, fontsize=13)
plt.tight_layout
plt.savefig(f'./plot/dreamer/Dreamer_valence_confusion.png')