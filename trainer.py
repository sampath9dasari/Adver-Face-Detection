import argparse
import torchvision
import time
from pathlib import Path
from datetime import datetime, date
import json

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from data_utils.wider_dataset import *
from data_utils.wider_eval import *
from data_utils.arraytools import *
from data_utils.data_read import *
from model.model_utils import *
from model.fasterrcnn import *
from lib.advattack import *
from lib.utils import *


def arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight-decay", default=0.0005, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--workers", default=6, type=int)
    parser.add_argument("--start-epoch", default=0, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--save-every", default=5, type=int)
    parser.add_argument("--resume", default="")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--limit-images", default=None, type=int)

    return parser.parse_args()


def main():
    args = arguments()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_img_list, train_bboxes = wider_read(limit_images=args.limit_images)
    val_img_list, val_bboxes = wider_read(limit_images=100, train=False)

    train_dataset = WiderDataset(train_img_list, train_bboxes)
    test_dataset = WiderDataset(val_img_list, val_bboxes)

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.workers,
                              collate_fn=collate_fn
                              )
    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             shuffle=True,
                             num_workers=args.workers,
                             collate_fn=collate_fn
                             )

    model = load_Faster_RCNN(backbone='resnet18')

    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    # lr_scheduler = None

    print("Model built")

    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # Set the start epoch if it has not been
        if not args.start_epoch:
            args.start_epoch = checkpoint['epoch']
        print('Model loaded from checkpoint')


    start_time = time.time()
    step_time = time.time()
    train_loss_hist = Averager()
    val_loss_hist = Averager()

    train_epoch_loss = {}

    model.train()
    print('Model training started')
    for epoch in range(args.start_epoch, args.epochs):
        train_loss_hist.reset()
        val_loss_hist.reset()
        #     print(epoch)
        try:
            train_epoch(model, epoch, train_loader, train_loss_hist, optimizer)

        except:
            epoch_time = time.time()
            print("Error block")
            print(f"Epoch Time elapsed: {(epoch_time - step_time) / 60:.2f} minutes")
            save_checkpoint({
                'epoch': epoch,
                'batch_size': train_loader.batch_size,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, filename=f"fasterrcnn_resnet18_checkpoint_{epoch+1}.pth")

            # End program execution
            return

        val_epoch(model, test_loader, val_loss_hist)
        # update the learning rate
        if lr_scheduler is not None:
            lr_scheduler.step()

        if epoch + 1 % args.save_every == 0:
            save_checkpoint({
                'epoch': epoch,
                'batch_size': train_loader.batch_size,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, filename=f"fasterrcnn_resnet18_checkpoint_{epoch+1}.pth")

        train_epoch_loss.update({epoch+1:train_loss_hist.value})
        print(f"Epoch #{epoch+1} Train loss: {train_loss_hist.value} | Val loss: {val_loss_hist.value}")
        epoch_time = time.time()
        print(f"Epoch Time elapsed: {(epoch_time - step_time) / 60:.2f} minutes")
        step_time = epoch_time

    end_time = time.time()
    print(f"Total Time elapsed: {(end_time - start_time) / 60:.2f} minutes")

    with open(f"results/train losses {datetime.now()}.json", "w") as outfile:
        json.dump(train_epoch_loss, outfile)

    save_checkpoint({
        'epoch': epoch,
        'batch_size': train_loader.batch_size,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }, filename=f"fasterrcnn_resnet18_{date.today()}.pth")


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
        prediction_info.append(predictions)
        target_info.append(targets)
    #     print(len(predictions[0]['scores']))

    end = time.time()
    print(f"Time elapsed in Predicting: {(end - start)/60:.2f} minutes")
    prediction_info = list(itertools.chain(*prediction_info))
    target_info = list(itertools.chain(*target_info))

    r = evaluation(prediction_info, target_info, iou_thresh=0.7, interpolation_method='EveryPoint')
    print(r['AP'])

    PlotPrecisionRecallCurve(r)


if __name__ == '__main__':
    main()
