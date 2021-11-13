import os
from os import listdir
from os.path import join

import numpy as np

from DataLoader import DataLoad
from Unet import UNet
from LossFunc import cross_loss

from torch.utils.data import DataLoader
import torch

# Nvidia Cuda Choice
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

# Train a epoch
def train_epoch(model, dataloader, optimizer, criterion, epoch, n_epochs):
    # set mode to train
    model.train()
    # each batch
    for batch_idx in range(dataloader.__len__()):
        # ? input & target
        input,target = dataloader.__iter__().__next__()

        # check cuda
        if torch.cuda.is_available():
            input = input.cuda()
            # todo .cuda

        # zero_grad
        optimizer.zero_grad()
        model.zero_grad()

        # stage
        # todo update
        res = '\t'.join(['Epoch: [%d/%d]' % (epoch + 1, n_epochs),
                         'Iter: [%d/%d]' % (batch_idx + 1, len(dataloader)),
                         'Loss %f' % (1),
                         'Dice %f' % (1)])
        print(res)
    # todo return loss

# Train Model
def train_net(net,n_epochs=200, batch_size=1, lr=1e-4, model1_name='Evaluation1_66_3.dat'):
    # Shape of the image
    shape = (224, 288, 288)

    # Dir paths
    train_image_dir = 'Txt/Txt_weak_acnet_66/image.txt'
    train_label_dir = 'Txt/Txt_weak_acnet_66/label.txt'
    train_mask_dir = 'Txt/Txt_weak_acnet_66/mask.txt'
    
    train_weak_image_dir = 'Txt/Txt_weak_acnet_66/weak_image.txt'
    train_weak_label_dir = 'Txt/Txt_weak_acnet_66/weak_label.txt'
    train_weak_mask_dir = 'Txt/Txt_weak_acnet_66/weak_mask.txt'
    
    test_image_dir = 'Txt/Txt_weak_acnet_66/test_image.txt'
    test_label_dir = 'Txt/Txt_weak_acnet_66/test_label.txt'
    
    # test_image_dir = 'data/train/temp'
    save_dir = 'Results/R_EEE_Learning_66_3'
    #save_dir2 = 'Results/R_EEE_Learning_66_1/out2'
    save_dir_m = 'Results/R_EEE_Learning_66_m_3'
    #save_dir_m_2 = 'Results/R_EEE_Learning_66_m_1/out2'    
    checkpoint_dir = 'Weights'
    checkpoint_dir2 = 'Weights_m'

    #net1.load_state_dict(torch.load(os.path.join(checkpoint_dir, model1_name)))
    #net2.load_state_dict(torch.load(os.path.join(checkpoint_dir, model2_name)))

    # Check Cuda
    if torch.cuda.is_available():
        net = net.cuda()

    # Load Data
    train_dataset = DataLoad(train_image_dir, train_label_dir, train_mask_dir, shape)
    # train_dataset = DatasetFromFolder3D(train_image_dir, train_artery_dir, shape)
   
    train_dataset_weak = DataLoad(train_weak_image_dir, train_weak_label_dir, train_weak_mask_dir, shape)
    # val_dataset = DatasetFromFolder3D(val_image_dir, val_target_dir, meanstd_dir=meanstd_dir, shape=shape, num_classes=num_classes)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # Optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # Loss function
    criterion1 = cross_loss()

    for epoch in range(n_epochs):
        # Train a epoch
        train_epoch(net, train_dataloader, optimizer, criterion1, epoch, n_epochs)
        # Checkpoint
        torch.save(net.state_dict(), os.path.join(checkpoint_dir, model1_name))
        # Print dice
        if (epoch > 100):
            predict(net, save_path=save_dir, shape=shape, img_path=test_image_dir, num_classes=1)
            tempdice = Dice(test_label_dir, save_dir)
            meandice = np.mean(tempdice)
            if (meandice > dice):
                dice = meandice
                epoch_m = epoch
                predict(net, save_path=save_dir_m, shape=shape, img_path=test_image_dir, num_classes=1)
                torch.save(net1.state_dict(), os.path.join(checkpoint_dir2, model1_name))
        print(dice)
        print(epoch_m)

def predict(model, save_path, shape, img_path, num_classes):
    print("Predict test data")
    # set mode to eval
    model.eval()

    # todo read file
    file = []
    file_num = len(file) 

    # todo read shape

    # todo random cut

    # todo predict

    # todo save

    pass