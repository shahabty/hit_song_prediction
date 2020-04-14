# Hit Song Prediction
This repo contains implementations of random forest, support vector machine, and neural network for hit song prediction in Pytorch and Sklearn.

### Installation
1. Install Anaconda3.

2. Run the following commands to create conda environment and install all dependencies:

```console
username@PC:~$ conda env create -f environment.yml
username@PC:~$ conda activate hits
```
### Training and Testing
#### Neural Network Model
```console
username@PC:~$ python train_nn.py.
```
#### SVM Model
```console
username@PC:~$ python train_svm.py
```
#### Random Forest Model
```console
username@PC:~$ python train_random_forest.py
```
### Visualization
Please visit this [link](https://elastic-curran-c983f0.netlify.com/) for the visualization of our neural network results.

The code for the above visualization is available [here](https://github.com/saraboback/SongPredictionViz/tree/master/song-prediction-master).

### Evaluation
Our evaluation metrics are precision, recall, and F1-Score:

|     Model     | Precision | Recall | F1-Score |
|:-------------:|:---------:|:------:|:--------:|
|   SVM(High)   |    0.71   |  0.70  |   0.71   |
|    SVM(Low)   |    0.71   |  0.80  |   0.75   |
| SVM(Combined) |    0.71   |  0.80  |   0.75   |
|    RF(High)   |    0.90   |  0.93  |   0.92   |
|    RF(Low)    |    0.92   |  0.94  |   0.93   |
|  RF(Combined) |    0.91   |  0.94  |   0.93   |
|    NN(High)   |    0.87   |  0.82  |   0.84   |
|    NN(Low)    |    0.90   |  0.87  |   0.89   |
|  NN(Combined) |    0.90   |  0.90  |   0.90   |

High: only high-level features are considered in the training and testing.
Low: only low-level features are considered in the training and testing.
Combined: Both high-level and low-level features contribute to the prediction.
## Contributors
[Mahshid Alinoori](https://github.com/mahshidaln),
[Sara Boback](https://www.linkedin.com/in/sara-boback/).
