# NeuroTalk

## Table of Contents
- [Key Features](#Key-Features)
- [Training](#Training)
- [Evaluation](#Evaluation)
- [Wav and CSV Transformation](#Wav-and-CSV-Transformation)
- [CSV and Melspectrogram Transformation](#CSV-and-Melspectrogram-Transformation)
- [Training.py Flowchart](#Training.py-Flowchart)
- [Feature Embedding](#Feature-Embedding)
- [License](#license)
- [Reference](#reference)
- [Contact](#contact)

## Key Features
* It was implemented using only Python among computer languages. -> CSP is using matlab
* Starting with the reference, the goal is to classify the desired class, not to generate a voice. Through this, quadriplegic patients can perform various tasks such as manipulating a wheelchair only with thoughts or playing games in a virtual environment.
* When using data from multiple subjects, write the name of the data folder as an abbreviation of the subject's name and enter it in Subject_name in the , wavtocsv, conv-1d, eg_analysis, and make_mel_nr_resample file

## Training
* **classname = ['call', 'camera', 'down', 'left', 'message', 'music', 'off', 'on', 'receive', 'right', 'turn', 'up', 'volume']**
* 'call' = 1, 'camera' = 2 , 'down'=3 ...
**To train the model for spoken EEG in the paper, run this command:(In subject, enter the initials of the subject)**
```train
python train.py  --task SpokenEEG_vec --batch_size 30 --pretrain False --prefreeze False  --sub sub_number
```
**To train the model for Imagined EEG with pretrained model of spoken EEG in the paper, run this command:(In subject, enter the initials of the subject)**
```train(To be revised later)
python train.py  --trained_model pretrained_model/SpokenEEG/ --task ImaginedEEG_vec --batch_size 30 --pretrain True --prefreeze True  --sub sub_number
```

## Evaluation
**To evaluate the trained model for spoken EEG on an example data, run(In subject, enter the initials of the subject):**
```eval
python eval.py  --task SpokenEEG_vec --batch_size 30  --sub sub_number
```
**To evaluate the trained model for Imagined EEG on an example data, run(In subject, enter the initials of the subject):**
```eval
python eval.py  --task ImaginedEEG_vec --batch_size 30  --sub sub_number
```
  
## Wav and CSV Transformation
* It is very important to check whether the recorded voice is stereo type or mono type, and the code must be used by applying this.
* wavtocsv : Converting voice data to csv file.  
* csvtowav : Converting the converted csv file back to voice data.
  
## CSV and Melspectrogram Transformation
* When obtaining the sampling rate of the Mel spectrogram, the sampling rate of the voice, and the Mel-spectrogram, it is very important to match the max frequency and the min-frequency.
  
## Feature Embedding
* **Preprocessed EEG data and label are required**
* 8 Components(CSP features) and 16 segments(Using var to seperate segments).
* It has 13 Classes. So it has 13*8 = 104 Features.
* During CSP, a problem occurred, so I used standardScaler. You can use other Scaler to scaling.
* Using N-Fold cv to checking reliability(In this code, 5 Fold)
* Using BBCI in matlab to implement multi-CSP
* Training Imgined EEG's CSP filter and applying it to Imagined and spoken EEG. Because it has less noise than spoken.
* Sum of CSP results for each class(It's not already confirmed) 


## Reference
* Y.-E. Lee, S.-H. Lee, S.-H Kim, and S.-W. Lee, "Towards Voice Reconstruction from EEG during Imagined Speech," AAAI Conference on Artificial Intelligence (AAAI), 2023.
* https://github.com/youngeun1209/NeuroTalk
