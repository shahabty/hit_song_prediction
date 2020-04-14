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

#### Random Forest: 

#### Neural Network:

#### Support vector machine:

## Contributors
[Mahshid Alinoori](https://github.com/mahshidaln),
[Sara Boback](https://www.linkedin.com/in/sara-boback/).
