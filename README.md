# FSEN: FastSlow Emotion Network for Cross-Subject EEG Emotion Recognition

Changgyun Jin, Hanul Kim and *Seong-Eun Kim<br/>
IEEE Transactions on Neural Systems & Rehabilitation Engineerin under review<br/>

## Abstart
As humanoid robots and artificial intelligence technologies advance, accurately recognizing human emotions has become
increasingly crucial in human-computer interactions. This capability enhances user experience across various applications such as
virtual reality, education, and healthcare. Electroencephalography (EEG) signals, which offer direct insights into neural activities linked
to emotions, are increasingly utilized in emotion recognition. However, accurate emotion detection from EEG signals faces challenges
such as high variability across individuals and label noise due to subjective reporting. Addressing these issues, we propose the
FastSlow Emotion Network (FSEN), which employs a cross-attention mechanism designed to mimic cross-frequency coupling,
essential for processing emotional states. FSEN uses a weakly supervised learning framework with signal-level labels to cope with
label noise and a two-phase multitask autoencoder to manage intersubject variability. This model was tested in a subject-independent
environment to validate its generalization performance on three publicly available datasets, SEED, SEED-IV, and Dreamer. FSEN
significantly outperformed the state-of-the-art algorithms, achieving accuracies of 99.4% on SEED, 87.68% on SEED-IV, and 92.5% for
valence and 84.97% for arousal on Dreamer. These results highlight FSEN’s robustness to noisy data and ability to minimize the
variance of data features per class. This validates its practical potential in real-world applications, offering substantial advancement in
EEG-based emotion recognition technolog

## FastSlow Emotion Network
![fig1_overview_new](https://github.com/user-attachments/assets/6e9116b8-0b56-47be-8e54-3e63be87ea01)

## How to use(SEED)
### Install requirements
``` Python
	pip install requirements.txt
```
### Data Preprocessing
>1. Download Data from the [Official website].(https://bcmi.sjtu.edu.cn/home/seed)
>2. Preprocess the data: 
``` Python
	python seedDownload.py
```
>> This will create a train/test file for each subject with pt files containing the preprocessed data and save everything inside

### Training
> To train with specific hyperparameters run
``` Python
	python seedDownload.py
```
>> Optimal hyperparameters can be found in the arguments descriprion.
