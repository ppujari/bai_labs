import torch
import numpy as np
from functions import *
import cv2
from PIL import Image
#torch.backends.cudnn.benchmark = True

def CenterCrop(frame, size):
    h, w = np.shape(frame)[0:2]
    th, tw = size
    x1 = int(round((w - tw) / 2.))
    y1 = int(round((h - th) / 2.))

    frame = frame[y1:y1 + th, x1:x1 + tw, :]
    return np.array(frame).astype(np.uint8)


def center_crop(frame):
    frame = frame[8:120, 30:142, :]
    return np.array(frame).astype(np.uint8)


#import cv2
#import numpy as np
def get_frames(filename, n_frames= 1):
    frames = []
    v_cap = cv2.VideoCapture(filename)
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_list= np.linspace(0, v_len-1, n_frames+1, dtype=np.int16)
    
    for fn in range(v_len):
        success, frame = v_cap.read()
        if success is False:
            continue
        if (fn in frame_list):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
            frames.append(frame)
    v_cap.release()
    return frames, v_len


def main():

    '''
    test_transforms=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor()])
    '''
    save_model_path = "./Conv3D_ckpt/"
    fc_hidden1, fc_hidden2 = 256, 256
    dropout = 0.0        # dropout probability
    k = 2 # number of target categories
    img_x, img_y =224,224 
    begin_frame, end_frame, skip_frame = 1, 16, 1
    selected_frames = np.arange(begin_frame, end_frame, skip_frame).tolist()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

    '''
    with open('./dataloaders/ucf_labels.txt', 'r') as f:
        class_names = f.readlines()
        f.close()
    '''
    class_names=['nonsmoke','smoke']
    # init model
    model=CNN3D(t_dim=len(selected_frames), img_x=img_x, img_y=img_y,
              drop_p=dropout, fc_hidden1=fc_hidden1,  fc_hidden2=fc_hidden2, num_classes=k)
    #model = C3D_model.C3D(num_classes=101)
    #checkpoint = torch.load('run/run_1/models/C3D_ucf101_epoch-39.pth.tar', map_location=lambda storage, loc: storage)
    #model.load_state_dict(checkpoint['state_dict'])
    #model.to(device)
    #model.eval()
    model.load_state_dict(torch.load(os.path.join(save_model_path, '3dcnn_epoch2.pth')))
    model.to(device)
    model.eval()

    # read video
    video = './test_data/2.avi'
    cap = cv2.VideoCapture(video)
    retaining = True

    clip = []
    while retaining:
        retaining, frame = cap.read()
        if not retaining and frame is None:
            continue
        #tmp=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #tmp_ = center_crop(cv2.resize(frame, (171, 128)))
        #tmp_ =cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        #tmp = tmp_ - np.array([[[90.0, 98.0, 102.0]]])
        clip.append(frame)
        if len(clip) == 16:
            img=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inputs = test_transforms(img).float()
            #inputs = np.expand_dims(inputs, axis=0)
            #inputs = np.transpose(inputs, (0, 4, 1, 2, 3))
            #inputs = torch.from_numpy(inputs)
            #inputs=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            inputs = inputs.unsqueeze_(0)

            inputs = torch.autograd.Variable(inputs, requires_grad=False)
            with torch.no_grad():
                outputs = model.forward(inputs)

            probs = torch.nn.Softmax(dim=1)(outputs)
            label = torch.max(probs, 1)[1].detach().cpu().numpy()[0]

            cv2.putText(frame, class_names[label].split(' ')[-1].strip(), (20, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 1)
            cv2.putText(frame, "prob: %.4f" % probs[0][label], (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 1)
            clip.pop(0)

        cv2.imshow('result', frame)
        cv2.waitKey(30)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()









