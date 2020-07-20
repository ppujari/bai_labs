Instructions to run the code.
============================


* Clone the repo bAI Labs from Github
* Download the **data_new** folder from bAI Labs -> Videos -> Dataset -> **data_new**
* Put it in a path known to you and paste that path for eg.
```
  home/avi/Desktop/data_new
```

* A file called smoking.pth has already been created ( to save the model after being trained ). 
* If not found : 
* Please create an empty file and save it as **smoking.pth** in the Folder Smoking_detection_Cmd_Line_v2 .

* In the file **train_new.py**, for data_dir, please give the path where you have saved data_new 	  downloaded earlier. 
	
Run **train_new.py**

(can add arguments --eps 1, --lr 0.001, but default values are set)
(number of epochs and learning rate can be changed as per your requirement)

The file will output the train and validation loss for the dataset.
Note that validation loss has been printed as test_loss in the code.


```
python train_new.py
```

* Code should run for 1 epoch using default hyperparameters and give you the tes and train loss. 	T

* Once this runs, We have saved our trained model and are now ready for testing. 
* Our Test videos folder is named as test1 in the repository.
* To test the code on random videos :
```
  python testVideo.py --frames_dir path_to_test_folder

  In the present case, it can be run as 

  python testVideo.py --frames_dir test1
```

* This will give you stats about: 
  Whether the video is smoking or non smoking
  % of Smoking detected in the video (based on the frames)


