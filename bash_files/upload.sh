#!/bin/bash

# Placement dans le dossier
cd /media/nvidia/B27876FA7876BD23/NRB-model/

# Création d'un dossier daté à uploader

DATENOW=$(date +"%Y-%m-%d") #String au format YY-mm-dd
mkdir $DATENOW

mv /media/nvidia/B27876FA7876BD23/NRB-model/yolov5/runs/detect /media/nvidia/B27876FA7876BD23/NRB-model/yolov5/runs/yolo
mv /media/nvidia/B27876FA7876BD23/NRB-model/yolov5/runs/yolo /media/nvidia/B27876FA7876BD23/NRB-model/$DATENOW

mkdir /media/nvidia/B27876FA7876BD23/NRB-model/$DATENOW/resnet
mv /media/nvidia/B27876FA7876BD23/NRB-model/resnet/0 /media/nvidia/B27876FA7876BD23/NRB-model/$DATENOW/resnet
mv /media/nvidia/B27876FA7876BD23/NRB-model/resnet/1 /media/nvidia/B27876FA7876BD23/NRB-model/$DATENOW/resnet
mv /media/nvidia/B27876FA7876BD23/NRB-model/resnet/2 /media/nvidia/B27876FA7876BD23/NRB-model/$DATENOW/resnet
mv /media/nvidia/B27876FA7876BD23/NRB-model/resnet/3 /media/nvidia/B27876FA7876BD23/NRB-model/$DATENOW/resnet
mv /media/nvidia/B27876FA7876BD23/NRB-model/resnet/4 /media/nvidia/B27876FA7876BD23/NRB-model/$DATENOW/resnet
mv /media/nvidia/B27876FA7876BD23/NRB-model/resnet/5 /media/nvidia/B27876FA7876BD23/NRB-model/$DATENOW/resnet
mv /media/nvidia/B27876FA7876BD23/NRB-model/resnet/6 /media/nvidia/B27876FA7876BD23/NRB-model/$DATENOW/resnet
mv /media/nvidia/B27876FA7876BD23/NRB-model/resnet/7 /media/nvidia/B27876FA7876BD23/NRB-model/$DATENOW/resnet
mv /media/nvidia/B27876FA7876BD23/NRB-model/resnet/8 /media/nvidia/B27876FA7876BD23/NRB-model/$DATENOW/resnet
mv /media/nvidia/B27876FA7876BD23/NRB-model/resnet/9 /media/nvidia/B27876FA7876BD23/NRB-model/$DATENOW/resnet
mv /media/nvidia/B27876FA7876BD23/NRB-model/resnet/10 /media/nvidia/B27876FA7876BD23/NRB-model/$DATENOW/resnet
mv /media/nvidia/B27876FA7876BD23/NRB-model/resnet/11 /media/nvidia/B27876FA7876BD23/NRB-model/$DATENOW/resnet
mv /media/nvidia/B27876FA7876BD23/NRB-model/resnet/12 /media/nvidia/B27876FA7876BD23/NRB-model/$DATENOW/resnet

mv /media/nvidia/B27876FA7876BD23/NRB-model/resnet/log/*.txt /media/nvidia/B27876FA7876BD23/NRB-model/$DATENOW/resnet

# Upload vers Dropbox
cd /media/nvidia/B27876FA7876BD23/Dropbox-Uploader/
./dropbox_uploader.sh upload /media/nvidia/B27876FA7876BD23/NRB-model/$DATENOW/ /

# Suppression du dossier uploadé

#rm -r /media/nvidia/B27876FA7876BD23/NRB-model/$DATENOW/