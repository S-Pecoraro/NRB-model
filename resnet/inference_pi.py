import torch
from torch import nn
from PIL import Image
from torchvision import transforms, models
import pandas as pd
import os
import os.path
from os import path
from shutil import copy
from tqdm import tqdm
import argparse
import cv2
import io
import numpy as np
from datetime import date
import time

parser = argparse.ArgumentParser()
parser.add_argument('--weights', nargs='+', type=str, default='model.pt', help='model.pt path(s)')
parser.add_argument('--source', type=str, default='images', help='source')  # folder, 0 for webcam
parser.add_argument('--save_img_from_stream', action='store_true', help='save img from frames')
parser.add_argument('--save_log', action='store_true', help='save detected object in log file')
opt = parser.parse_args()
print(opt)

FOLDER_PATH = opt.source

if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    ref_labels = pd.read_csv('labels.csv')

    for label in ref_labels.index.to_list():
        if not os.path.isdir(str(label)):
            os.mkdir(str(label))
    
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model.fc = nn.Linear(num_ftrs, len(ref_labels))
    
    model = model.to(device)
    model.load_state_dict(torch.load(opt.weights), strict = False) if torch.cuda.is_available() else model.load_state_dict(torch.load(opt.weights, map_location=torch.device('cpu')), strict = False)
    model.eval()
    time1 = time.time()
    if opt.source == str(0) : #camera is used
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
        cap.set(cv2.CAP_PROP_FPS, 40)
        if cap.isOpened():
                cv2.namedWindow("Resnet Stream", cv2.WINDOW_AUTOSIZE)
                while True :
                    ret, img_cv2 = cap.read()
                    image = Image.fromarray(img_cv2)
                    transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
                    pred = model(transform(image).unsqueeze(0).to(device)).squeeze()
                    pred = nn.functional.softmax(pred,dim=0)
                    predicted_class = ref_labels.loc[int(pred.argmax())]['label_name_fr']
                    conf = float(pred.max())
                    index_predicted_class = ref_labels.loc[int(pred.argmax())]['label_id']
                    time2 = time.time()
                    time_predict = time2-time1
                    string = str(predicted_class) + '(' + str(index_predicted_class)+ ') ' + ": " + str(conf)+ " ("+ str(time_predict) +"s)"+'\n' 
                    print(string)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(img_cv2, string, (10,30), font,1, (255,255,255),2)
                    cv2.imshow("Resnet Stream", img_cv2)
                    if opt.save_img_from_stream :                
                        img_path = os.path.join(os.getcwd(), str(index_predicted_class))
                        path, dirs, files= next(os.walk(img_path))
                        number_img = len(files)
                        file_path = img_path + '/' + str(number_img) +'.jpg'
                        print("Image save in "+file_path +'\n')
                        image.save(file_path, 'JPEG')
                    if opt.save_log:
                        today = date.today()
                        log_file= "log/"+today.strftime("%Y-%m-%d")+".txt"
                        if not(os.path.isdir('log')):
                            try:
                                os.mkdir('log')
                            except OSError:
                                print("Creation of the directory 'log' failed")
                            else :
                                print("Succesfully created the directory 'log'")
                        else:
                            f=open(log_file,"a")
                            f.write(string)
                            f.close()
                            print("Log saved")
                    if cv2.waitKey(1) == ord('q'):  # q to quit
                        cap.release()
                        cv2.destroyAllWindows()
                        break
        else :
                print("Cannot open camera")
                
    else :
        for im in tqdm(os.listdir(FOLDER_PATH)):
            try:
                img_path = os.path.join(FOLDER_PATH, im)
                image = Image.open(img_path).convert('RGB')
                transform = transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                            ])
                
                pred = model(transform(image).unsqueeze(0).to(device)).squeeze()
                pred = nn.functional.softmax(pred,dim=0)
                predicted_class = ref_labels.loc[int(pred.argmax())]['label_name_fr']
                conf = float(pred.max())
                print(str(predicted_class) + ": " + str(conf))
                copy(img_path, str(predicted_class))
            except:
                pass
    time_end = time.time() - time1
    print(str(time_end) + "s\n")