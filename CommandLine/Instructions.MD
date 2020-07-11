Instructions to run the file 
============================

* Clone the repo bAI Labs from Github
* Download the **data_new** folder from bAI Labs -> Videos -> Dataset -> **data_new**
* Put it in a path known to you and paste that path for eg.
```
  home/anurag/documents/data_new
```
  in the file **smoker_detector_base.py** at line 55, for data_dir
  
* Run the file using : (can add arguments --m alexnet/resnet, --eps number, --lr 0.001, but default values are set)
```
python smoker_detector_base.py
```

* Code should run for 1 epoch using default hyperparameters and give an accuracy, that means it has run correctly. 
