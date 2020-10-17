# CNN LSTM 
Implementation of CNN LSTM with Resnet backend for Video Classification

# Getting Started
## Prerequisites
* PyTorch (ver. 0.4+ required) (CPU only if no GPU availaible, download accordingly)
* FFmpeg, FFprobe (Download)
* Python 3


### Try on your own dataset 

```
mkdir data
mkdir data/video_data
```
Put your video dataset inside data/video_data
It should be in this form --

```
+ data 
    + video_data    
            - action1
            - action2
            + smoking 
                    - smoking0.avi
                    - smoking.mkv
                    - smoking1.mp4
            +nonsmoking
            		-nonsmoking0.avi
            		-nonsmomking.mkv
            		-nonsmking.mp4
```

Generate Images from the Video dataset
```
./utils/generate_data.sh

Once this is completed we can proceed for training/testing.
```

## Train
Once you have created the dataset, start training ->
We have already trained the MODEL so please run only the inference part. No need to train currently.
```
python main.py --use_cuda --gpu 0 --batch_size 8 --n_epochs 20 --num_workers 0  --annotation_path ./data/annotation/ucf101_01.json --video_path ./data/image_data/  --dataset ucf101 --sample_size 150 --lr_rate 1e-4 --n_classes 2 
```

## Note 
* All the weights will be saved to the snapshots folder 
* To resume Training from any checkpoint, Use
```
--resume_path <path-to-model> 
```


## Inference
```
Download the saved model from google drive and put them in the snapshots folder.
Make a test folder where you have your test videos and run the command below: 


python inference.py  --annotation_path ./data/annotation/ucf101_01.json  --dataset ucf101 --model cnnlstm --n_classes 2 --resume_path snapshots/cnnlstm-Epoch-20-Loss-0.9298445197796694.pth



You can use any "Epoch-num_epochs-Loss-loss_value.pth" model in the snapshots folder. Make sure that you give the correct path for the model to be used.
```


