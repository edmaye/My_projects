import time
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.models as models
import numpy as np 
from tqdm import tqdm
from torchvision import models
from torch.autograd import Variable
from PIL import Image
from torch import nn
from network.deepv3 import DeepR101V3PlusD_HANet
from network.deepv3 import DeepWideResNet101V3PlusD_HANet_OS8
from network.deepv3 import DeepShuffleNetV3PlusD_HANet_OS32
from nets.unet_training import CE_Loss,Dice_loss
from utils.metrics import f_score
from torch.utils.data import DataLoader
from dataloader import unetDataset, unet_dataset_collate
import argparse
parser = argparse.ArgumentParser(description='Semantic Segmentation')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--arch', type=str, default='network.deepv3.DeepWV3Plus',
                    help='Network architecture. We have DeepSRNX50V3PlusD (backbone: ResNeXt50) \
                    and deepWV3Plus (backbone: WideResNet38).')
parser.add_argument('--dataset', type=str, default='cityscapes',
                    help='cityscapes, mapillary, camvid, kitti')
parser.add_argument('--cv', type=int, default=0,
                    help='cross-validation split id to use. Default # of splits set to 3 in config')

parser.add_argument('--class_uniform_pct', type=float, default=0,
                    help='What fraction of images is uniformly sampled')
parser.add_argument('--class_uniform_tile', type=int, default=1024,
                    help='tile size for class uniform sampling')
parser.add_argument('--coarse_boost_classes', type=str, default=None,
                    help='use coarse annotations to boost fine data with specific classes')

parser.add_argument('--img_wt_loss', action='store_true', default=False,
                    help='per-image class-weighted loss')
parser.add_argument('--cls_wt_loss', action='store_true', default=False,
                    help='class-weighted loss')
parser.add_argument('--batch_weighting', action='store_true', default=False,
                    help='Batch weighting for class (use nll class weighting using batch stats')

parser.add_argument('--jointwtborder', action='store_true', default=False,
                    help='Enable boundary label relaxation')
parser.add_argument('--strict_bdr_cls', type=str, default='',
                    help='Enable boundary label relaxation for specific classes')
parser.add_argument('--rlx_off_iter', type=int, default=-1,
                    help='Turn off border relaxation after specific epoch count')
parser.add_argument('--rescale', type=float, default=1.0,
                    help='Warm Restarts new learning rate ratio compared to original lr')
parser.add_argument('--repoly', type=float, default=1.5,
                    help='Warm Restart new poly exp')

parser.add_argument('--fp16', action='store_true', default=False,
                    help='Use Nvidia Apex AMP')
parser.add_argument('--local_rank', default=0, type=int,
                    help='parameter used by apex library')

parser.add_argument('--sgd', action='store_true', default=True)
parser.add_argument('--adam', action='store_true', default=False)
parser.add_argument('--amsgrad', action='store_true', default=False)

parser.add_argument('--freeze_trunk', action='store_true', default=False)
parser.add_argument('--hardnm', default=0, type=int,
                    help='0 means no aug, 1 means hard negative mining iter 1,' +
                    '2 means hard negative mining iter 2')

parser.add_argument('--trunk', type=str, default='resnet101',
                    help='trunk model, can be: resnet101 (default), resnet50')
parser.add_argument('--max_epoch', type=int, default=180)
parser.add_argument('--max_iter', type=int, default=30000)
parser.add_argument('--max_cu_epoch', type=int, default=100000,
                    help='Class Uniform Max Epochs')
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--crop_nopad', action='store_true', default=False)
parser.add_argument('--rrotate', type=int,
                    default=0, help='degree of random roate')
parser.add_argument('--color_aug', type=float,
                    default=0.25, help='level of color augmentation')
parser.add_argument('--gblur', action='store_true', default=False,
                    help='Use Guassian Blur Augmentation')
parser.add_argument('--bblur', action='store_true', default=False,
                    help='Use Bilateral Blur Augmentation')
parser.add_argument('--lr_schedule', type=str, default='poly',
                    help='name of lr schedule: poly')
parser.add_argument('--poly_exp', type=float, default=1.0,
                    help='polynomial LR exponent')
parser.add_argument('--bs_mult', type=int, default=2,
                    help='Batch size for training per gpu')
parser.add_argument('--bs_mult_val', type=int, default=1,
                    help='Batch size for Validation per gpu')
parser.add_argument('--crop_size', type=int, default=720,
                    help='training crop size')
parser.add_argument('--pre_size', type=int, default=None,
                    help='resize image shorter edge to this before augmentation')
parser.add_argument('--scale_min', type=float, default=0.5,
                    help='dynamically scale training images down to this size')
parser.add_argument('--scale_max', type=float, default=2.0,
                    help='dynamically scale training images up to this size')
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--snapshot', type=str, default=None)
parser.add_argument('--snapshot_pe', type=str, default=None)
parser.add_argument('--restore_optimizer', action='store_true', default=False)

parser.add_argument('--city_mode', type=str, default='train',
                    help='experiment directory date name')
parser.add_argument('--date', type=str, default='default',
                    help='experiment directory date name')
parser.add_argument('--exp', type=str, default='default',
                    help='experiment directory name')
parser.add_argument('--tb_tag', type=str, default='',
                    help='add tag to tb dir')
parser.add_argument('--ckpt', type=str, default='logs/ckpt',
                    help='Save Checkpoint Point')
parser.add_argument('--tb_path', type=str, default='logs/tb',
                    help='Save Tensorboard Path')
parser.add_argument('--syncbn', action='store_true', default=False,
                    help='Use Synchronized BN')
parser.add_argument('--dump_augmentation_images', action='store_true', default=False,
                    help='Dump Augmentated Images for sanity check')
parser.add_argument('--test_mode', action='store_true', default=False,
                    help='Minimum testing to verify nothing failed, ' +
                    'Runs code for 1 epoch of train and val')
parser.add_argument('-wb', '--wt_bound', type=float, default=1.0,
                    help='Weight Scaling for the losses')
parser.add_argument('--maxSkip', type=int, default=0,
                    help='Skip x number of  frames of video augmented dataset')
parser.add_argument('--scf', action='store_true', default=False,
                    help='scale correction factor')
parser.add_argument('--dist_url', default='tcp://127.0.0.1:', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--hanet', nargs='*', type=int, default=[0,0,0,0,0],
                    help='Row driven attention networks module')
parser.add_argument('--hanet_set', nargs='*', type=int, default=[0,0,0],
                    help='Row driven attention networks module')
parser.add_argument('--hanet_pos', nargs='*', type=int, default=[0,0,0],
                    help='Row driven attention networks module')
parser.add_argument('--pos_rfactor', type=int, default=0,
                    help='number of position information, if 0, do not use')
parser.add_argument('--aux_loss', action='store_true', default=False,
                    help='auxilliary loss on intermediate feature map')
parser.add_argument('--attention_loss', type=float, default=0.0)
parser.add_argument('--hanet_poly_exp', type=float, default=0.0)
parser.add_argument('--backbone_lr', type=float, default=0.0,
                    help='different learning rate on backbone network')
parser.add_argument('--hanet_lr', type=float, default=0.0,
                    help='different learning rate on attention module')
parser.add_argument('--hanet_wd', type=float, default=0.0001,
                    help='different weight decay on attention module')                    
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--pos_noise', type=float, default=0.0)
parser.add_argument('--no_pos_dataset', action='store_true', default=False,
                    help='get dataset with position information')
parser.add_argument('--use_hanet', action='store_true', default=False,
                    help='use hanet')
parser.add_argument('--pooling', type=str, default='mean',
                    help='pooling methods, average is better than max')


args = parser.parse_args()

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_one_epoch(net,epoch,epoch_size,epoch_size_val,gen,genval,Epoch,cuda,aux_branch):
    net = net.train()
    total_loss = 0
    total_f_score = 0

    val_toal_loss = 0
    val_total_f_score = 0
    start_time = time.time()
    with tqdm(total=epoch_size,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size: 
                break
            imgs, pngs, labels = batch

            with torch.no_grad():
                imgs = Variable(torch.from_numpy(imgs).type(torch.FloatTensor))
                pngs = Variable(torch.from_numpy(pngs).type(torch.FloatTensor)).long()
                labels = Variable(torch.from_numpy(labels).type(torch.FloatTensor))
                if cuda:
                    imgs = imgs.cuda()
                    pngs = pngs.cuda()
                    labels = labels.cuda()

            #-------------------------------#
            #   判断是否使用辅助分支并回传
            #-------------------------------#
            optimizer.zero_grad()
            if aux_branch:
                iuu,iu,aux_outputs, outputs = net(imgs)
                iuuloss  = CE_Loss(iuu, pngs, num_classes = NUM_CLASSES)
                iuloss  = CE_Loss(iu, pngs, num_classes = NUM_CLASSES)


                aux_loss  = CE_Loss(aux_outputs, pngs, num_classes = NUM_CLASSES)
                main_loss = CE_Loss(outputs, pngs, num_classes = NUM_CLASSES)
                loss      = aux_loss + main_loss+iuuloss+iuloss
                if dice_loss:
                    aux_dice  = Dice_loss(aux_outputs, labels)
                    main_dice = Dice_loss(outputs, labels)
                    loss      = loss + aux_dice + main_dice

            else:
                outputs = net(imgs)
                loss    = CE_Loss(outputs, pngs, num_classes = NUM_CLASSES)
                if dice_loss:
                    main_dice = Dice_loss(outputs, labels)
                    loss      = loss + main_dice

            with torch.no_grad():
                #-------------------------------#
                #   计算f_score
                #-------------------------------#
                _f_score = f_score(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_f_score += _f_score.item()
            
            waste_time = time.time() - start_time
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), 
                                'f_score'   : total_f_score / (iteration + 1),
                                's/step'    : waste_time,
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)

            start_time = time.time()

    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_size_val:
                break
            imgs, pngs, labels = batch
            with torch.no_grad():
                imgs = Variable(torch.from_numpy(imgs).type(torch.FloatTensor))
                pngs = Variable(torch.from_numpy(pngs).type(torch.FloatTensor)).long()
                labels = Variable(torch.from_numpy(labels).type(torch.FloatTensor))
                if cuda:
                    imgs = imgs.cuda()
                    pngs = pngs.cuda()
                    labels = labels.cuda()
                #-------------------------------#
                #   判断是否使用辅助分支
                #-------------------------------#
                if aux_branch:
                    iuu,iu,aux_outputs, outputs = net(imgs)
                    iuuloss  = CE_Loss(iuu, pngs, num_classes = NUM_CLASSES)
                    iuloss  = CE_Loss(iu, pngs, num_classes = NUM_CLASSES)


                    aux_loss  = CE_Loss(aux_outputs, pngs, num_classes = NUM_CLASSES)
                    main_loss = CE_Loss(outputs, pngs, num_classes = NUM_CLASSES)
                    val_loss      = aux_loss + main_loss+iuuloss+iuloss
                    if dice_loss:
                        aux_dice  = Dice_loss(aux_outputs, labels)
                        iuu_dice=Dice_loss(iuu,labels)
                        iu_dice=Dice_loss(iu,labels)
                        main_dice = Dice_loss(outputs, labels)
                        val_loss  = val_loss + aux_dice + main_dice+iuu_dice+iu_dice

                else:
                    outputs  = net(imgs)
                    val_loss = CE_Loss(outputs, pngs, num_classes = NUM_CLASSES)
                    if dice_loss:
                        main_dice = Dice_loss(outputs, labels)
                        val_loss  = val_loss + main_dice

                #-------------------------------#
                #   计算f_score
                #-------------------------------#
                _f_score = f_score(outputs, labels)

                val_toal_loss += val_loss.item()
                val_total_f_score += _f_score.item()
                
            
            pbar.set_postfix(**{'total_loss': val_toal_loss / (iteration + 1),
                                'f_score'   : val_total_f_score / (iteration + 1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)
            
    print('Finish Validation')
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss/(epoch_size+1),val_toal_loss/(epoch_size_val+1)))
    totalBig_loss = ('%.4f' % (total_loss/(epoch_size+1)))
    val_loss1232= ('%.4f' % (val_toal_loss/(epoch_size_val+1)))
    file_handle2=open('train_loss.csv',mode='a+')
  
    file_handle2.write(totalBig_loss+','+val_loss1232+'\n')
    file_handle2.close()

    print('Saving state, iter:', str(epoch+1))
    torch.save(model.state_dict(), 'logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth'%((epoch+1),total_loss/(epoch_size+1),val_toal_loss/(epoch_size_val+1)))



if __name__ == "__main__":
    inputs_size = [224,224,3]
    log_dir = "logs/"   
    #---------------------#
    #   分类个数+1
    #   2+1
    #---------------------#
    NUM_CLASSES = 2
    #--------------------------------------------------------------------#
    #   建议选项：
    #   种类少（几类）时，设置为True
    #   种类多（十几类）时，如果batch_size比较大（10以上），那么设置为True
    #   种类多（十几类）时，如果batch_size比较小（10以下），那么设置为False
    #---------------------------------------------------------------------# 
    dice_loss = True
    #-------------------------------#
    #   主干网络预训练权重的使用
    #  
    #-------------------------------#
    pretrained = False
    backbone = "ECAresnet"
    #---------------------#
    #   是否使用辅助分支
    #   会占用大量显存
    #---------------------#
    aux_branch = False
    #---------------------#
    #   下采样的倍数
    #   8和16
    #---------------------#
    downsample_factor = 16
    #-------------------------------#
    #   Cuda的使用
    #-------------------------------#
    Cuda = True

    model = DeepR101V3PlusD_HANet(num_classes=NUM_CLASSES,args=args, criterion=None, criterion_aux=None).train()
    
    #-------------------------------------------#
    #   权值文件的下载请看README
    #   权值和主干特征提取网络一定要对应
    #-------------------------------------------#
    # model_path = r"model_data/pspnet_mobilenetv2.pth"
    # # 加快模型训练的效率
    # print('Loading weights into state dict...')
    # model_dict = model.state_dict()
    # pretrained_dict = torch.load(model_path)
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict)
    # print('Finished!')

    if Cuda:
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.cuda()

    # 打开数据集的txt
    with open(r"VOCdevkit/VOC2007/ImageSets/Segmentation/train.txt","r") as f:
        train_lines = f.readlines()

    # 打开数据集的txt
    with open(r"VOCdevkit/VOC2007/ImageSets/Segmentation/val.txt","r") as f:
        val_lines = f.readlines()
        
    #------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Interval_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    #------------------------------------------------------#
    if True:
        lr = 1e-3
        Init_Epoch = 0
        Interval_Epoch = 60
        Batch_size = 6
        optimizer = optim.Adam(model.parameters(),lr)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.9)

        train_dataset = unetDataset(train_lines, inputs_size, NUM_CLASSES, True)
        val_dataset = unetDataset(val_lines, inputs_size, NUM_CLASSES, False)
        gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=0, pin_memory=True,
                                drop_last=True, collate_fn=unet_dataset_collate)
        gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=4,pin_memory=True, 
                                drop_last=True, collate_fn=unet_dataset_collate)

        epoch_size      = max(1, len(train_lines)//Batch_size)
        epoch_size_val  = max(1, len(val_lines)//Batch_size)

       
        for epoch in range(Init_Epoch,Interval_Epoch):
            fit_one_epoch(model,epoch,epoch_size,epoch_size_val,gen,gen_val,Interval_Epoch,Cuda,aux_branch)
            lr_scheduler.step()
    
    if True:
        lr = 1e-5
        Interval_Epoch = 60
        Epoch = 130
        Batch_size = 4
        optimizer = optim.Adam(model.parameters(),lr)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.9)

        train_dataset = unetDataset(train_lines, inputs_size, NUM_CLASSES, True)
        val_dataset = unetDataset(val_lines, inputs_size, NUM_CLASSES, False)
        gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                drop_last=True, collate_fn=unet_dataset_collate)
        gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=4,pin_memory=True, 
                                drop_last=True, collate_fn=unet_dataset_collate)

        epoch_size      = max(1, len(train_lines)//Batch_size)
        epoch_size_val  = max(1, len(val_lines)//Batch_size)

        # for param in model.backbone.parameters():
        #     param.requires_grad = True

        for epoch in range(Interval_Epoch,Epoch):
            fit_one_epoch(model,epoch,epoch_size,epoch_size_val,gen,gen_val,Epoch,Cuda,aux_branch)
            lr_scheduler.step()

