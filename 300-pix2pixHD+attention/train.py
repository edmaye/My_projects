import os
import numpy as np
import torch
import time
import sys
from collections import OrderedDict #collections是Python内建的一个集合模块，提供了许多有用的集合类。 OrderedDict是在使用dict时保持key的顺序，Key会按照插入的顺序排列，不是Key本身排序。
from torch.autograd import Variable #autograd为Tensors上的所有操作提供了自动区分，同时也是一个逐个运行的框架，意味着backprop由代码运行定义，每一次迭代都可以不同。
from pathlib import Path #pathlib提供了使用语义表达来表示文件系统路径的类，这些类适合多种操作系统。
import warnings

warnings.filterwarnings('ignore') #忽略警告错误的输出
mainpath = os.getcwd() #返回当前工作目录
pix2pixhd_dir = Path(mainpath+'/src/pix2pixHD/')
sys.path.append(str(pix2pixhd_dir)) #添加自己的搜索目录时，可以通过列表的append()方法，对于模块和自己写的脚本不在同一个目录下，在脚本开头加sys.path.append(‘xxx’)


from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
import src.config.train_opt as opt #opt代表一些参数值的设置

os.environ['CUDA_VISIBLE_DEVICES'] = "0" #使用指定的GPU和GPU显存
torch.multiprocessing.set_sharing_strategy('file_system') #用于在相同数据的不同进程中共享视图，set_sharing_strategy设置共享CPU张量的策略
torch.backends.cudnn.benchmark = True #设置这个 flag 可以让内置的 cuDNN 的 auto-tuner 自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题

####################################################################################################################################################################
def main():
    iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt') #查看当前epoch值
    #返回包含input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor, 'feat': feat_tensor, 'path': A_path}的dataset
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data() 
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    start_epoch, epoch_iter = 1, 0
    total_steps = (start_epoch - 1) * dataset_size + epoch_iter
    display_delta = total_steps % opt.display_freq
    print_delta = total_steps % opt.print_freq
    save_delta = total_steps % opt.save_latest_freq

    model = create_model(opt) #选择模型pix2pixHD模型或者UIModel
    model = model.cuda()
    #visualizer = Visualizer(opt) #结果的显示方式的选择

    for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        if epoch != start_epoch:
            epoch_iter = epoch_iter % dataset_size
        for i, data in enumerate(dataset, start=epoch_iter):
            iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize

            # whether to collect output images
            save_fake = total_steps % opt.display_freq == display_delta

            ############## n Pass ######################
            losses, generated = model(Variable(data['label']), Variable(data['inst']),
                                      Variable(data['image']), Variable(data['segment']), 
                                      Variable(data['feat']), infer=save_fake)
            
            # sum per device losses
            losses = [torch.mean(x) if not isinstance(x, int) else x for x in losses]
            loss_dict = dict(zip(model.loss_names, losses))

            # calculate final loss scalar
            loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
            loss_G = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat', 0) + loss_dict.get('G_VGG', 0) + loss_dict['G_EDGE']

            ############### Backward Pass ####################
            # update generator weights
            model.optimizer_G.zero_grad()
            loss_G.backward()
            model.optimizer_G.step()

            # update discriminator weights
            model.optimizer_D.zero_grad()
            loss_D.backward()
            model.optimizer_D.step()


            ############## Display results and errors ##########
            ### print out errors
            if total_steps % opt.print_freq == print_delta:
                #errors = {k: v.data[0] if not isinstance(v, int) else v for k, v in loss_dict.items()}
                errors = {k: v.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}
                t = (time.time() - iter_start_time) / opt.batchSize
                # visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                # visualizer.plot_current_errors(errors, total_steps)

            ### display output images
            if save_fake:
                visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], 0)),
                                       ('synthesized_image', util.tensor2im(generated.data[0])),
                                       ('real_image', util.tensor2im(data['image'][0]))])
                #visualizer.display_current_results(visuals, epoch, total_steps)

            ### save latest model
            if total_steps % opt.save_latest_freq == save_delta:
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                model.save('latest')
                np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

            if epoch_iter >= dataset_size:
                break

        # end of epoch
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        ### save model for this epoch
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            model.save('latest')
            model.save(epoch)
            np.savetxt(iter_path, (epoch + 1, 0), delimiter=',', fmt='%d')

        ### instead of only training the local enhancer, train the entire network after certain iterations
        if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
            model.update_fixed_params()

        ### linearly decay learning rate after certain iterations
        if epoch > opt.niter:
            model.update_learning_rate()

    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
