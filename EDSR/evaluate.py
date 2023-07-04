from utils import AverageMeter
import torch
from super_image.trainer_utils import EvalPrediction
from super_image.utils.metrics import compute_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(model, valid_loader, criterion, scale):
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
