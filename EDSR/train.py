from utils import AverageMeter, save_checkpoint
import time 
import torch
from super_image.trainer_utils import EvalPrediction
from super_image.utils.metrics import compute_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(train_loader,valid_loader, model, criterion, optimizer, epoch, scale, print_every = 5):
    
    global best_metric, best_epoch
    losses = AverageMeter()
    
    for i, (img, label) in enumerate(train_loader):
        
        start = time.time()

        img, label = img.to(device), label.to(device)
        pred = model(img)
        # print(pred.shape, label.shape)
        loss = criterion(pred, label)
        
        losses.update(loss.item(), img.size(0))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print status
        if i % print_every == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Training Time {3:.3f} \t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                  (time.time()-start)*print_every, loss=losses))

    with torch.no_grad():
        
        model.eval()
        val_losses = AverageMeter()
        epoch_psnr = AverageMeter()
        epoch_ssim = AverageMeter()
        
        for i, (val_inputs, val_labels) in enumerate(valid_loader):
            
            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
            val_preds = model(val_inputs)
            val_loss = criterion(val_preds, val_labels)
            
            val_losses.update(val_loss.item(), val_inputs.size(0))
            
            metrics = compute_metrics(EvalPrediction(predictions=val_preds, labels=val_labels), scale=scale)

            epoch_psnr.update(metrics['psnr'], val_inputs.size(0))
            epoch_ssim.update(metrics['ssim'], val_inputs.size(0))

        print(f'Validation Loss:{val_losses.avg:.2f}      eval psnr: {epoch_psnr.avg:.2f}     ssim: {epoch_ssim.avg:.4f}')

        if epoch_psnr.avg > best_metric:
            best_epoch = epoch
            best_metric = epoch_psnr.avg

            print(f'best epoch: {epoch}, psnr: {epoch_psnr.avg:.6f}, ssim: {epoch_ssim.avg:.6f}')
            
            # Save checkpoint
            print("Saving checkpoint epoch:", epoch)
            save_checkpoint(epoch, model, optimizer)

#         print('Epoch : {}/{}'.format(epoch_num, epochs))
#         print('Training Loss : {:.4f}'.format(losses.avg))
#         print('Validation Loss: {:.4f}'.format(val_losses.avg))