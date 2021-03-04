# AlcoAudio Research

Detection of Alcohol induced intoxication through voice using Neural Networks

## Table of Contents

* [Dataset](#dataset)
* [Architectures](#architectures)
* [Setup](#setup--usage)
* [Usage](#usage)
  * [Data generation](#data-generation)
  * [Training the network](#training-the-network)
  * [Inference](#inference-of-the-best-model)
* [Future work](#future-work-todo)
  * [Data Representations](#improve-on-data-representations)
  * [New Architecture try outs](#try-new-architectures)
* [Known Issues](#known-issues)
* [Acknowledgement](#acknowledgements)


## **Dataset**

[Alcohol Language Corpus](https://www.phonetik.uni-muenchen.de/Bas/BasALCeng.html) is a curation of audio samples from 162 speakers. Audio samples are first recorded when speaker is sober. Then the speakers are given a chosen amount of alcohol to reach a particular intoxication state, and audio samples are recorded again. 

Audio samples are split into 8 seconds each. Below is the plot of a raw signal

![Raw Signal](plots/raw_signal.jpg) <!-- .element height=250 width=250 -->

These raw audio signals are converted into Mel filters using ```librosa```. Below is how it looks:

![FBank](plots/fbank.jpg) <!-- .element height=250 width=250 -->

## **Architectures**

Below are the architectures tried. All the files are under [networks](https://github.com/ShreeshaN/AlcoAudio/tree/master/alcoaudio/networks) folder. 


|Networks   |  Log Loss | UAR(Unweighted Average Recall)  |
|---|---|---|
|  [Convolutional Neural Networks](https://github.com/ShreeshaN/AlcoAudio/blob/master/alcoaudio/networks/convnet.py)(convnet) |  0.89 | 66.28  |
| [LSTM](https://github.com/ShreeshaN/AlcoAudio/blob/master/alcoaudio/networks/recurrent_net.py)(lstm)  |  1.59 | 58.12  |
| [Conv LSTMs](https://github.com/ShreeshaN/AlcoAudio/blob/master/alcoaudio/networks/crnn.py)(crnn)  |  1.17 | 62.27  |
| [One class Neural Networks](https://github.com/ShreeshaN/AlcoAudio/blob/OC_NN/alcoaudio/networks/oneclass_net.py)(ocnn)  |  1.81 |  55 |
| [Conv Auto Encoders](https://github.com/ShreeshaN/AlcoAudio/blob/autoencoders/alcoaudio/networks/convautoencoder_net.py)(cae)  | 0.92  | 65.53  |



## **Setup**

1. Download and run the requirements.txt to install all the dependencies.

      
       pip install -r requirements.txt
     
     
2. Create a [config](https://github.com/ShreeshaN/AlcoAudio/blob/master/alcoaudio/configs/shree_configs.json) file of your own

3. Install OpenSmile and set environment variable ```OPENSMILE_CONFIG_DIR``` to point to the config directory of OpenSmile installation.

## Usage

### **Data generation**

Run ```data_processor.py``` to generate data required for training the model. It reads the raw audio samples, splits into ```n``` seconds and generates Mel filters, also called as Filter Banks (```fbank``` paramater in config file. Other available audio features are ```mfcc``` & ```gaf```)

    python3 data_processor.py --config_file <config_filepath>

### **Training the network**

Using ```main.py``` one can train all the architectures mentioned in the above section.

    python3 main.py --config_file <config_filepath> --network convnet
        
### **Inference**

One can use our model for inference. The best model is being saved under [best_model](alcoaudio/best_model) folder
       
    python3 main.py --config_file --test_net True <config_filepath> --network convnet --datapath <data filepath>
       
   Remember to generate mel filters from raw audio data and use the generated ```.npy``` file for datapath parameter
   

## **Future work: TODO**

### **Improve on Data Representations**

 - [ ] Work on frequency variance in voice
 - [ ] Recurrence plots
 - [ ] Extract features using Praat and Opensmile
 - [ ] Normalise audio sample based on average amplitude

### **Try new architectures**

 - [ ] Conditional Variational AutoEncoder
 - [ ] Convolutional One class Neural Network

## Known Issues
1. As training progresses, test and valid log losses increase. The confidence with which the network miss predicts increase. The below graph depicts this behaviour
![CF plot](plots/cf_plot.png) <!-- .element height=250 width=250 -->

2. Mel filters or MFCC are not the best representation for this use case as these representations fail to capture variance in the amplitudes rather just try to mimic human voice. 
![Data 2d plot ](plots/data_plot.png) <!-- .element height=250 width=250 -->


## **Acknowledgements**
Our team would like to thank Professor [Emmanuel O. Agu](https://www.wpi.edu/people/faculty/emmanuel) for guiding the team throughout. I would like to thank team members [Pratik](https://github.com/PRAkTIKal24), [Arjun](https://github.com/arjunrao796123) and [Mitesh]() for all their contributions.

