import argparse
import torchvision
import time
from pathlib import Path
from datetime import datetime, date
import json
import sys

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from data_utils.wider_dataset import *
from data_utils.wider_eval import *
from data_utils.arraytools import *
from data_utils.data_read import *
from model.model_utils import *
from model.fasterrcnn import *
from model.advattack import *
from data_utils.utils import *

torch.cuda.empty_cache()

def arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--weight-decay", default=0.0005, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--workers", default=0, type=int)
    parser.add_argument("--start-epoch", default=0, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--save-every", default=5, type=int)
    parser.add_argument("--resume", default="")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--limit-images", default=None, type=int)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--freeze-rpn", action="store_true")
    parser.set_defaults(freeze_backbone=False)
    parser.set_defaults(freeze_rpn=False)

    return parser.parse_args()


def main():
    args = arguments()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_img_list, train_bboxes = wider_read(limit_images=args.limit_images)
    test_img_list, test_bboxes = wider_read(limit_images=args.limit_images, train=False)

    train_dataset = WiderDataset(train_img_list[:-1000], train_bboxes[:-1000])
    val_dataset = WiderDataset(train_img_list[-1000:], train_bboxes[-1000:])
    test_dataset = WiderDataset(test_img_list, test_bboxes)
    
    print('Batch Size being used: ',args.batch_size)

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.workers,
                              collate_fn=collate_fn
                              )
    val_loader = DataLoader(val_dataset,
                             batch_size=2,
                             shuffle=True,
                             num_workers=args.workers,
                             collate_fn=collate_fn
                             )
    test_loader = DataLoader(test_dataset,
                             batch_size=2,
                             shuffle=True,
                             num_workers=args.workers,
                             collate_fn=collate_fn
                             )

    model = load_Faster_RCNN(backbone='resnet18', freeze_backbone=args.freeze_backbone, freeze_rpn=args.freeze_rpn)

    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    # lr_scheduler = None

    print("Model built")

    if args.resume:
        checkpoint = torch.load(os.getcwd()+'/saved_models/'+args.resume)
        model.load_state_dict(checkpoint['model'])
#         if args.freeze_backbone is False and args.freeze_rpn is False:
#         optimizer.load_state_dict(checkpoint['optimizer'])
        # Set the start epoch if it has not been
        if not args.start_epoch:
            args.start_epoch = checkpoint['epoch']+1
        print('Model loaded from checkpoint ', args.resume)


    start_time = time.time()
    step_time = time.time()
    train_loss_hist = Averager()
    val_loss_hist = Averager()

    train_epoch_loss = {}

    model.train()
    print('Model training started at ',datetime.now())
    for epoch in range(args.start_epoch, args.epochs):
        
#         torch.cuda.empty_cache()
#         model.to(device)
        
        train_loss_hist.reset()
        val_loss_hist.reset()
        #     print(epoch)
        try:
            train_epoch(model, epoch, train_loader, train_loss_hist, optimizer)

        except Exception() as e:
            epoch_time = time.time()
            print("Error block")
            print(f"Epoch Time elapsed: {convert(epoch_time - step_time)}")
            save_checkpoint({
                'epoch': epoch,
                'batch_size': train_loader.batch_size,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, filename=f"fasterrcnn_resnet18_checkpoint_{date.today()}_{epoch}.pth")
            
            raise e
            # End program execution
            return

        val_epoch(model, val_loader, val_loss_hist)
        # update the learning rate
        if lr_scheduler is not None:
            lr_scheduler.step()

        if (epoch) % args.save_every == 0:
            save_checkpoint({
                'epoch': epoch,
                'batch_size': train_loader.batch_size,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, filename=f"fasterrcnn_resnet18_checkpoint_{date.today()}_{epoch}.pth")
            print(f"Saved Model - fasterrcnn_resnet18_checkpoint_{date.today()}_{epoch}.pth")

        train_epoch_loss.update({epoch:train_loss_hist.value})
        print(f"Epoch #{epoch} Train loss: {train_loss_hist.value} | Val loss: {val_loss_hist.value}")
        epoch_time = time.time()
        print(f"Epoch Time elapsed: {convert(epoch_time - step_time)}")
        step_time = epoch_time

    end_time = time.time()
    print(f"Total Time elapsed: {convert(end_time - start_time)}")

    with open(f"results/train losses {date.today()} final.json", "w") as outfile:
        json.dump(train_epoch_loss, outfile)

    save_checkpoint({
        'epoch': epoch,
        'batch_size': train_loader.batch_size,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }, filename=f"fasterrcnn_resnet18_{date.today()}_final.pth")
    print(f"Saved Final Model fasterrcnn_resnet18_{date.today()}_final.pth")


    # TEST SET METRICS
    start = time.time()

    prediction_info = []
    target_info = []
    model.eval()

    for images, targets in test_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            predictions = model(images)
            
        predictions = [{k: v.to('cpu').detach() for k, v in t.items()} for t in predictions]
        targets = [{k: v.to('cpu').detach() for k, v in t.items()} for t in targets]
        prediction_info.append(predictions)
        target_info.append(targets)
        torch.cuda.empty_cache()
    #     print(len(predictions[0]['scores']))

    end = time.time()
    print(f"Time elapsed in Predicting: {convert(end - start)}")
    prediction_info = list(itertools.chain(*prediction_info))
    target_info = list(itertools.chain(*target_info))

    r = evaluation(prediction_info, target_info, iou_thresh=0.5, interpolation_method='EveryPoint', disable_bar=True)
    print("The mAP for Test Set is:",r['AP'])

#     PlotPrecisionRecallCurve(r)


if __name__ == '__main__':
    main()
