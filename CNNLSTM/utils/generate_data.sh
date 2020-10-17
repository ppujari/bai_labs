rm -rf ./data/annotation
mkdir ./data/annotation
rm -rf ./data/image_data
mkdir ./data/image_data

python utils/video_jpg_ucf101_hmdb51.py
python utils/n_frames_ucf101_hmdb51.py
python utils/gen_anns_list.py
python utils/ucf101_json.py