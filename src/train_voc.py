import argparse
import torch
import torch.nn as nn
import numpy as np # type: ignore
import torch.optim as optim
import torch.backends.cudnn as cudnn
import os
import os.path as osp
import matplotlib.pyplot as plt # type: ignore

from src.config import *
from util import *
from util.typing.basic import *
from dataset.voc.dataset_voc import dataloader_voc
from model.vgg_voc import Our_Model


# from typing import List, Set, Dict, Tuple, Optional, Union, Callable, Iterator


def load_model(args) -> Tuple(Our_Model, int):
    if args.restore_from_where == "pretrained":
        return Our_Model(split), 0
    else:
        raise Exception('Restore model function has not been supported!')
        # restore_from = get_model_path(args.snapshot_dir)
        # model_restore_from = restore_from["model"]
        # i_iter = restore_from["step"]

        # model = Our_Model(split)
        # saved_state_dict = torch.load(model_restore_from)
        # model.load_state_dict(saved_state_dict)
    #end load_model



'''
Real work start!
@Auther: Ziqiang Y
'''
def main() -> None:
    """ 
    Main zero shot segmentation function 
    """
    args = get_arguments()
    device = args.device
    print_config(args)

    w, h = args.input_size.split(",")
    input_size = (int(w), int(h))
    model, i_iter = load_model(args)
    model.train()
    model.to(device)

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    train_loader = dataloader_voc(split = split)
    data_len = len(train_loader)
    num_steps = data_len * args.num_epochs

    '''
    optimizer = optim.SGD(
        model.optim_parameters_1x(args),
        lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    
    optimizer_10x = optim.SGD(
        model.optim_parameters_10x(args),
        lr=10 * args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    '''
    optimizer = optim.Adam(
        model.optim_parameters_1x(args),
        lr=args.learning_rate, weight_decay=args.weight_decay)
    optimizer.zero_grad()

    optimizer_10x = optim.Adam(
        model.optim_parameters_10x(args),
        lr=10 * args.learning_rate, weight_decay=args.weight_decay)
    optimizer_10x.zero_grad()

    seg_loss = nn.CrossEntropyLoss(ignore_index=255)
    #seg_loss = FocalLoss() # merely test if focal loss is useful...

    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode="bilinear", align_corners=True)

    with open(RESULT_DIR, "a") as f:
        f.write(SNAPSHOT_PATH.split("/")[-1] + "\n")
        f.write("lambda : " + str(lambdaa) + "\n")

    average_mIoUs_history = []

    blank_line(2)
    log('Training start ...')
    for epoch in range(args.num_epochs + 1):
        # blank_line()
        log(">> Epoch: {}".format(epoch))
        train_iter = enumerate(train_loader)
        model.train()
        hist = np.zeros((15, 15))
        average_mIoUs_per_epoch = 0
        for i in range(data_len):
            blank_line()
            log("> Epoch {}, loop {}".format(epoch, i))
            loss_pixel = 0
            loss_pixel_value = 0

            optimizer.zero_grad()
            adjust_learning_rate(optimizer, i_iter, num_steps, args, times=1)

            optimizer_10x.zero_grad()
            adjust_learning_rate(optimizer_10x, i_iter, num_steps, args, times=10)

            # train strong
            try:
                _, batch = train_iter.__next__()
            except StopIteration:
                train_strong_iter = enumerate(train_loader)
                _, batch = train_iter.__next__()

            images, masks = batch["image"], batch["label"]
            #print("mask: ", masks[0].min())
            images = images.to(device)
            masks = masks.long().to(device)
            pred = model(images, "all")
            pred = interp(pred)


            # Calculate mIoU
            debug(pred.shape, 'pred.shape')
            debug(masks.shape, 'mask.shape')
            #pred_IoU = pred[0].permute(1, 2, 0)
            #pred_0 = pred[0].clone().cpu()
            #pred_IoU = torch.max(pred_0, 0)[0].byte()

            max_ = torch.argmax(pred, 1)
            pred_IoU = max_[0].clone().detach().cpu().numpy()

            #print(pred_IoU.shape)
            #pred_cpu = pred_IoU.data.cpu().numpy()
            pred_cpu = pred_IoU
            mask_cpu = masks[0].cpu().numpy()

            pred_cpu[mask_cpu == 255] = 255
            m = confusion_matrix(mask_cpu.flatten(), pred_cpu.flatten(), 15)
            #hist += m
            hist += m
            mIoUs = per_class_iu(hist)
            average_mIoUs = sum(mIoUs) / len(mIoUs)
            average_mIoUs_per_epoch = average_mIoUs
            
            log("> mIoU: \n{}".format(per_class_iu(m)))
            log("> mIoUs: \n{}".format(mIoUs))
            log("> Average mIoUs: \n{}".format(average_mIoUs))

            '''
            m = confusion_matrix(np.array([1, 1, 1, 255]), np.array([1, 2, 0, 0]), 3)
            mIoUs = per_class_iu(m)
            print("test> mIoU: {}".format(mIoUs))
            '''

            #vis = to_color_img(pred.clone().detach())
            #print(vis.shape)
            #pyplot.imshow(vis)
            #pyplot.imshow(masks[0])
            #pyplot.show()
            loss_pixel = seg_loss(pred, masks)
            loss = loss_pixel# + loss_qfsl

            max_ = torch.argmax(pred, 1)
            #print(max_[0])
            if i % 10 == 0:
                ans = max_[0].clone().detach().cpu().numpy()
                #x = np.where(ans == 0, 255, ans)
                x = ans
                y = masks.cpu()[0]
                x[y == 255] = 255
                draw_sample(batch)
                # pyplot.figure()
                plt.imshow(x, cmap = 'tab20', vmin = 0, vmax = 21)
                plt.colorbar()
                # show_figure_nonblocking()
                save_figure('output/Epoch-{}-{}.pdf'.format(epoch, batch['name'][0]))
                
            debug('{} {}'.format(max_[0, 200, 200].data, masks[0, 200, 200].data))
            log("loss: {}".format(loss))
            loss.backward()
            optimizer.step()
            optimizer_10x.step()

        average_mIoUs_history.append(average_mIoUs_per_epoch)
        if epoch > 0 and epoch % SHOW_EPOCH == 0:
            plot(range(0, len(average_mIoUs_history)), average_mIoUs_history)
            # show_figure_nonblocking()
            if epoch >= SHOW_EPOCH:
                save_figure('output/mIoUs of Epoches {}-{}.pdf'.format(0, epoch))

    #end main






if __name__ == "__main__":
    main()
