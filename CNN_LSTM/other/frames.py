import os
import myutils

path2data = "./dataset"
sub_folder = "videos"
sub_folder_jpg = "frames"
path2aCatgs = os.path.join(path2data, sub_folder)

listOfCategories = os.listdir(path2aCatgs)
listOfCategories, len(listOfCategories)

for cat in listOfCategories:
    print("category:", cat)
    path2acat = os.path.join(path2aCatgs, cat)
    listOfSubs = os.listdir(path2acat)
    print("number of sub-folders:", len(listOfSubs))
    print("-"*50)


extension = [".avi",".mp4"]
n_frames = 16
rep=""
for root, dirs, files in os.walk(path2aCatgs, topdown=False):
    for name in files:
        flag = 1
        for ext in extension:
            if ext in name:
                rep=ext
                flag = 0
        if flag == 1:
            continue
        path2vid = os.path.join(root, name)
        frames, vlen = myutils.get_frames(path2vid, n_frames= n_frames)
        #print(path2vid," ",vlen)
        path2store = path2vid.replace(sub_folder, sub_folder_jpg)
        #print(path2store)
        path2store = path2store.replace(rep, "")
        #print(path2store)
        os.makedirs(path2store, exist_ok= True)
        myutils.store_frames(frames, path2store)
    print("-"*50)    

