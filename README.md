Implementation of Huawei's hand gesture file transfer nwith MobilenetV3
===
This is a mockup implementation of Huawei's air sharing gesture using hand landmarks detection trained with MobileNetV3

<!-- Abstract:xxx
## Papar Information
- Title:  `paper name`
- Authors:  `A`,`B`,`C`
- Preprint: [https://arxiv.org/abs/xx]()
- Full-preprint: [paper position]()
- Video: [video position]() -->

## How to Setup
- Clone the repository
  ```
  git clone https://github.com/NaufalM479/AirShare_1D-Mobilenet_XGB.git
  ```
- cd to the directory 
  ```
  cd AirShare_1D-Mobilenet_XGB
  ```
- Update the Submodules
  ```
  git submodule update --init --recursive
  ```
- Create and activate conda environment 
  ```
  conda create -n <ENV_NAME> python=3.10
  conda activate <ENV_NAME>
  ```
- Install the requirements
  ```
  pip install -r requirements.txt
  ```
- Install google hand landmarker model to Models/
  ```
  wget https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task -O Models/hand_landmarker.task
  ```
  
## Dataset Preparation
| Dataset | Author | Download |
| ---     | ---   | ---     |
| Stone Paper Scissors Hand Landmarks Dataset | Aryan Sinha | [download](https://www.kaggle.com/datasets/aryan7781/stone-paper-scissors-hand-landmarks-dataset/data) |

## Use
- to train train the model
  ```
  # please use devices with Nvidia RTX GPU installed
  # download the dataset on the project root directory as csv then do:

  python train_rtx.py
  ```
- to run the code
  ```
  # run the code to use on your device
  # do the same setup step and run the code for other devices

  python Send_Receive.py
  ```

## Code Details
### Trained Using
- software
  ```
  OS: Archlinux
  Python: 3.10.20 (anaconda)
  ```
- hardware
  ```
  CPU: AMD Ryzen 7 3700X
  GPU: Nvidia GeForce RTX 5070 ti
  ```


### Tested Platform
- software
  ```
  OS: Archlinux
  Python: 3.10.20 (anaconda)
  ```
- hardware 1
  ```
  CPU: Intel Ultra 5 125H
  GPU: Intel Arc Graphics
  ```
- hardware 2
  ```
  CPU: AMD Ryzen 7 5800H
  GPU: Nvidia GeForce RTX 3050 ti laptop
  ```
  

### Hyper parameters
#### MobilenetV3 Architecture
```
length=42
num_channel=1
num_filters=32
output_nums=2
```
#### Compilation
```
optimizer='adam'
loss='sparse_categorical_crossentropy'
```
#### Training
```
epochs=100
batch_size=128
patience=12
restore_best_weights=true
```

## Dependencies and Credits
- [MobileNet-1D-2D-Tensorflow-Keras](https://github.com/Sakib1263/MobileNet-1D-2D-Tensorflow-Keras.git)
- [air-share](https://github.com/amanverma-765/air-share.git)

<!-- ## License -->

<!-- ## Citing
If you use xxx,please use the following BibTeX entry.
```
``` -->


