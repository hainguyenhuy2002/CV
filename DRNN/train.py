import argparse
import copy
import os

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from datasets import load_dataset
from model import DRRN
from super_image.data import EvalDataset, TrainDataset, augment_five_crop
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms.functional import InterpolationMode, resize
from tqdm import tqdm
from utils import AverageMeter, EvalPrediction, compute_metrics, load_weights, Args

if __name__ == "main":

    parser = argparse.ArgumentParser()

    parser.add_argument("--B", default=1)
    parser.add_argument("--U", default=9)
    parser.add_argument("--num_features", default=32)
    parser.add_argument("--lr", default=0.1)
    parser.add_argument("--weights_file", default=None)
    parser.add_argument("--clip_grad", default=0.01)
    parser.add_argument("--num_epochs", default=30)
    parser.add_argument("--num_workers", default=2)
    parser.add_argument("--seed", default=32)
    parser.add_argument("--outputs_dir", default="/kaggle/working/")
    parser.add_argument("--batch_size", default=128)
    parser.add_argument("--eval_scale", default=2)
    parser.add_argument("--gpu", default=False)

    parse_args = parser.parse_args()

    B = parse_args.B
    U = parse_args.U
    lr = parse_args.lr
    weights_file = parse_args.weights_file
    clip_grad = parse_args.clip_grad
    num_epochs = parse_args.num_epochs
    num_workers = parse_args.num_workers
    seed = parse_args.seed
    outputs_dir = parse_args.outputs_dir
    batch_size = parse_args.batch_size
    eval_scale = parse_args.eval_scale
    gpu = parse_args.gpu
    num_features = parse_args.num_features


    # Define parameter for training
    args = Args(B = B, U = U, num_features = num_features,
                lr = lr, weights_file = weights_file, 
                clip_grad = clip_grad, num_epochs = num_epochs, 
                num_workers=num_workers, seed = seed, 
                outputs_dir = outputs_dir, 
                batch_size=batch_size, eval_scale = eval_scale)

    augmented_dataset = load_dataset('eugenesiow/Div2k', 'bicubic_x{}'.format(args.eval_scale), split='train')                          # download and augment the data with the five_crop method
    train_dataset = TrainDataset(augmented_dataset)                                                     # prepare the train dataset for loading PyTorch DataLoader
    eval_dataset = EvalDataset(load_dataset('eugenesiow/Div2k', 'bicubic_x{}'.format(args.eval_scale), split='validation'))      # prepare the eval dataset for the PyTorch DataLoader

    if parse_args.gpu == False:
        device = "cpu"
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if device == 'cpu':
            print("Setting flag 'gpu' = True but device doens't contain 'gpu'")

    model = DRRN(B=args.B, U=args.U, num_features=args.num_features, num_channels=3).to(device)

    if args.weights_file is not None:
        model = load_weights(model, args.weights_file)

    #     train_dataset = TrainDataset(args.train_file)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True)

    if args.eval == True:
    #     eval_dataset = EvalDataset(args.eval_file)
        eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0

    criterion = nn.MSELoss(reduction='sum')

    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.num_epochs):
        lr = args.lr * (0.5 ** ((epoch + 1) // 10))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        model.train()
        epoch_losses = AverageMeter()

        with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size), ncols=80) as t:
            t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))

            for data in train_dataloader:
                inputs, labels = data
                x = resize(inputs, size = (inputs.shape[2] * 2, inputs.shape[3] * 2), interpolation  =InterpolationMode.BICUBIC)
                x = x.to(device)
                
                labels = labels.to(device)

                preds = model(x)

                loss = criterion(preds, labels) / (2 * len(inputs))
                epoch_losses.update(loss.item(), len(inputs))

                optimizer.zero_grad()
                loss.backward()

                nn.utils.clip_grad.clip_grad_norm_(model.parameters(), args.clip_grad / lr)

                optimizer.step()

                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg), lr=lr)
                t.update(len(inputs))

        torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch)))

        if args.eval == True:
            model.eval()
            epoch_psnr = AverageMeter()
            epoch_ssim = AverageMeter()

            for data in eval_dataloader:
                inputs, labels = data
                x = resize(inputs, size = (inputs.shape[2] * args.eval_scale, inputs.shape[3] * args.eval_scale), 
                        interpolation = InterpolationMode.BICUBIC)
                x = x.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    preds = model(x)

    #             preds = denormalize(preds.squeeze(0).squeeze(0))
    #             labels = denormalize(labels.squeeze(0).squeeze(0))
                res = compute_metrics(EvalPrediction(predictions=preds, labels=labels), args.eval_scale)
                
    #             epoch_psnr.update(PSNR(preds, labels, shave_border=args.eval_scale), len(inputs))
                epoch_psnr.update(res['psnr'], len(inputs))
                epoch_ssim.update(res['ssim'], len(inputs))

            print('eval psnr: {:.2f}, eval ssim: {:.2f}'.format(epoch_psnr.avg, epoch_ssim.avg))

            if epoch_psnr.avg > best_psnr:
                best_epoch = epoch
                best_psnr = epoch_psnr.avg
                best_psnr_ssim = epoch_ssim.avg
                best_weights = copy.deepcopy(model.state_dict())

    if args.eval ==True:
        print('best epoch: {}, psnr: {:.2f}, ssim: {:.2f}'.format(best_epoch, best_psnr, best_psnr_ssim))
        torch.save(best_weights, os.path.join(args.outputs_dir, 'best.pth'))