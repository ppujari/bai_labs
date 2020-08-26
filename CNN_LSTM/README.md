# CNN LSTM 
Implementation of CNN LSTM with Alexnet/Resnet as backend for Video Classification


Please put your dataset as follows
```
./data
    /videos
       /smoke
       /nonsmoke


 Run video2jpg.py

 It will form the train and validation folders and split them accordingly
```
After the dataset has been created please ```run traincpu.py```

```
The model is implemented in model_cpu.py


Thought process:
Implements an LSTM with a CNN such as alexnet/resnet etc.
The forward function is for the LSTM.

We face a problem during testing which is :
Expected 4-dimensional input for 4-dimensional weight [64, 3, 11, 11], but got input of size [3, 224, 224] instead
After some debugging and online references we were unable to deduce the root cause 

What we think is that the error is occuring during training itself. i.e in model_cpu.py.


Here we get the features from the input and reshape it using VIEW in pytorch (lines 90-94)

			#print(f.shape)  #torch.Size([8, 256, 6, 6]) #before reshaping
			f = f.view(f.size(0), -1) # getting reshaped here.
			#print(f.shape) #torch.Size([8, 9216]) #after reshaping.

			If we do not reshape it using view we get a matrix multiplication error m1 x m2
```

Please let us know the steps further.

Some references 
[Link1](https://discuss.pytorch.org/t/expected-4-dimensional-input-for-4-dimensional-weight-64-20-7-7-but-got-input-of-size-30-9-instead/40903)
[Link2](https://discuss.pytorch.org/t/runtimeerror-expected-4-dimensional-input-for-4-dimensional-weight-64-3-4-4-but-got-3-dimensional-input-of-size-3-64-64-instead/46446/5)
[Linke3](https://discuss.pytorch.org/t/cnn-lstm-implementation-for-video-classification/52018)

