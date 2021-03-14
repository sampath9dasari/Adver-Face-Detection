
import torchvision
import time
from pathlib import Path

from data_utils.wider_dataset import *
from data_utils.wider_eval import *
from data_utils.arraytools import *
from data_utils.data_read import *
from model.model_utils import *
from model.fasterrcnn import *
from lib.advattack import *
from lib.utils import *

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
train_img_list, train_bboxes = wider_read(10)
val_img_list, val_bboxes = wider_read(10, train=False)

train_dataset = WiderDataset(train_img_list, train_bboxes)
test_dataset = WiderDataset(val_img_list, val_bboxes)

train_loader = DataLoader(train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn
)
test_loader = DataLoader(test_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn
)


model = load_Faster_RCNN(backbone='resnet18')

model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
# lr_scheduler = None

num_epochs = 10

start_time = time.time()
step_time = time.time()
train_loss_hist = Averager()
val_loss_hist = Averager()
itr = 1
model.train()
# model.train()
for epoch in range(num_epochs):
    train_loss_hist.reset()
    val_loss_hist.reset()
    #     print(epoch)
    train_epoch(model, train_loader, train_loss_hist, optimizer)
    val_epoch(model, test_loader, val_loss_hist)
    # update the learning rate
    if lr_scheduler is not None:
        lr_scheduler.step()

    if epoch % 5 == 0:
        torch.save(model.state_dict(), 'saved_models/fasterrcnn_resnet18_checkpoint.pth')

    print(f"Epoch #{epoch} Train loss: {train_loss_hist.value} | Val loss: {val_loss_hist.value}")
    epoch_time = time.time()
    print(f"Epoch Time elapsed: {(epoch_time - step_time) / 60:.2f} minutes")
    step_time = epoch_time

end_time = time.time()
print(f"Total Time elapsed: {(end_time - start_time) / 60:.2f} minutes")
