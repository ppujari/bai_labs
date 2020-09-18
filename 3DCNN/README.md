# Video Classification using 3DCNN'S


## Dataset
The dataset will compose of videos from the HMDB51 dataset
We have only 2 classes namely smoking and nonsmoking

**I have put the dataset on google drive**


**To make your own dataset: <br>**
1.) make two folders : `smoking and nonsmoking` and put the `videos` in those folders respectively<br>
2.) put both of them in a parent folder called `videos`<br>
3.) put the videos folder into a parent folder called `datasets`<br>
4.) put the dataset folder in the frames folder<br>
5.) run `frames.py` in the frames folder to make your image dataset.<br>

Once you have made your dataset, bring it to the main 3DCNN directory (out of the frames folder)<br>


## Models 
### 1. 3D CNN (train from scratch)
Use several 3D kernels of size *(a,b,c)* and channels *n*,  *e.g., (a, b, c, n) = (3, 3, 3, 16)* to convolve with video input, where videos are viewed as 3D images. *Batch normalization* and *dropout* are also used.


## Training & testing
- For 3D CNN:
   1. The videos are resized as **(t-dim, channels, x-dim, y-dim) = (16, 3, 256, 342)** since CNN requires a fixed-size input. 
   The minimal frame number 16 is the consensus of all videos in hmdb51 dataset.
 
## Usage 
  - `3DCNN.py`: model parameters, training/testing process.
  - `function.py`: modules of 3DCNN data loaders, and some useful functions.

### 1. Prerequisites (dependencies)
- [Python 3.6](https://www.python.org/) (or above)
- [PyTorch](https://pytorch.org/)
- [Numpy 1.15.0](http://www.numpy.org/) 
- [Sklearn 0.19.2](https://scikit-learn.org/stable/)
- [Matplotlib](https://matplotlib.org/)
- [Pandas](https://pandas.pydata.org/)
- [tqdm](https://github.com/tqdm/tqdm)


### 2. Set parameters & path

```
data_path = "./dataset/images/"         #video path
save_model_path = "./Conv3D_ckpt/"
```

### 3. Train & test model

- For **3D CNN** model, in each folder run
```bash
$ python 3DCNN.py <br>
The training of the model works well.

The problem occurs during testing: <br> 
When we run the test.py folder, we face some errors which are either related to dimensions of the test video input which is not matching the parameters of the saved model, or when we try to tranpose the matrix to fit the dimensioned but fall short of a dimension. 
```


### 4. Model ouputs

By default, the model outputs:

- Training & testing loss/ accuracy: `epoch_train_loss/score.npy`, `epoch_test_loss/score.npy`

- Model parameters & optimizer: eg. `CRNN_epoch*.pth`, `CRNN_optimizer_epoch*.pth  where * can be any number base on the number of epochs you choose the give.`.

<br>
