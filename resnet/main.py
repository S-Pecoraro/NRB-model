from classModel import classModel
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models, transforms
import os
import pandas as pd
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str)
    parser.add_argument('--img_meta_path', type=str)
    parser.add_argument('--ref_label', type=str)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--logs', type=str, default='')
    opt = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img_dir = opt.img_path  # '../dataset_v2/images/'
    img_meta = opt.img_meta_path
    ref_label_path = opt.ref_label

    train_imgs = os.listdir(img_dir + 'train')
    val_imgs = os.listdir(img_dir + 'val')

    print(f'number of images in train dataset (labeled or not): {len(train_imgs)}')
    print(f'number of images in train dataset (labeled or not): {len(val_imgs)}')
    ref_labels = pd.read_csv(ref_label_path)
    data = pd.read_csv(img_meta, usecols=['filename', 'image_width', 'image_height', 'label'])
    # aggregation to select the most major class for an image in case of different labels on the same image
    data = data.groupby(['filename', 'image_width', 'image_height']).agg(
        lambda x: x.value_counts().index[0]).reset_index()
    data = data.merge(ref_labels, on='label', how='inner')

    train_df = data[data['filename'].isin(train_imgs)].reset_index(drop=True)
    val_df = data[data['filename'].isin(val_imgs)].reset_index(drop=True)
    print(f'number of images in train dataset labeled: {len(train_df)}')
    print(f'number of images in train dataset labeled: {len(val_df)}')

    writer = SummaryWriter('./runs/' + opt.logs, flush_secs=10)
    classification_model = classModel(device=device,
                                      data_dir=img_dir,
                                      writer=writer,
                                      workers=opt.workers,
                                      epochs=opt.epochs,
                                      batch_size=opt.batch)
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    classification_model.load_data(
        imgs_root_dir=img_dir,
        train_data=train_df,
        val_data=val_df,
        ref_labels=ref_labels,
        data_transforms=data_transforms)

    classification_model.imbalanced_dataset()

    model_ft = models.resnet50(pretrained=True)
    # for parameter in model_ft.parameters():
    #     parameter.requires_grad = False
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, len(ref_labels))

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    classification_model.train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler)
