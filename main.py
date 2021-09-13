import os
import random

import torch
import torch.nn as nn
import torchvision

from config import cfg
from src import pytorch_utils as ptu
from src import utils
from src import data_utils



if cfg.seed is not None:
    random.seed(cfg.seed)
    torch.random.manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True

cpu_device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('seed:', cfg.seed)
print('device:', device)
print('version:', cfg.version)
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True
    print(torch.cuda.get_device_properties(device))


dataset_a = data_utils.AndrewmvdDataset(os.path.join(cfg.data_path, cfg.dataset_a), transforms=cfg.train_transforms)
dataset_w = data_utils.WobotIntelligenceDataset(os.path.join(cfg.data_path, cfg.dataset_w), transforms=cfg.train_transforms)


if cfg.train_test:
    dataset_a, val_a = data_utils.train_test_split(data_utils.AndrewmvdDataset, dataset_a)
    dataset_w, val_w = data_utils.train_test_split(data_utils.WobotIntelligenceDataset, dataset_w)

    val_loader = torch.utils.data.DataLoader(data_utils.UnionDataset(val_a, val_w),
                                             batch_size=cfg.bs,
                                             num_workers=cfg.num_workers,
                                             shuffle=False,
                                             drop_last=True, collate_fn=data_utils.collate_fn)
else:
    val_loader = None

train_dataset = data_utils.UnionDataset(dataset_a, dataset_w)

if cfg.balanced_classes:
    train_dataset.get_inverse_weights(tqdm_bar=cfg.tqdm_bar)
    shuffle = False
    sampler = torch.utils.data.WeightedRandomSampler(train_dataset.item_weights,
                                                     num_samples=int(cfg.bs * cfg.num_batches),
                                                     replacement=True)
else:
    shuffle = True
    sampler = None

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=cfg.bs,
                                           num_workers=cfg.num_workers,
                                           sampler=sampler,
                                           shuffle=shuffle,
                                           drop_last=True, collate_fn=data_utils.collate_fn)


if cfg.load is not None and os.path.exists(os.path.join(cfg.models_dir, cfg.version, ptu.naming_scheme(cfg.version, epoch=cfg.load)) + '.pth'):
    checkpoint = ptu.load_model(cpu_device, version=cfg.version, models_dir=cfg.models_dir, epoch=cfg.load)
    for state in checkpoint.optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    if cfg.prints == 'display':
        display(checkpoint.log.sort_index(ascending=False).head(20))
    elif cfg.prints == 'print':
        print(checkpoint.log.sort_index(ascending=False).head(20))
else:
    model = vars(torchvision.models.detection)[cfg.model](pretrained=False,
                                                          num_classes=cfg.num_classes,
                                                          pretrained_backbone=True,
                                                          progress=False)

    if cfg.optimizer == 'sgd':
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                                    lr=cfg.lr,
                                    momentum=cfg.optimizer_momentum,
                                    weight_decay=cfg.wd)
    elif cfg.optimizer == 'adam':
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                      lr=cfg.lr,
                                      weight_decay=cfg.wd)
    else:
        raise NotImplementedError

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                              T_max=cfg.epochs,
                                                              eta_min=cfg.min_lr) if cfg.cos else None

    checkpoint = utils.MyCheckpoint(version=cfg.version,
                                    model=model,
                                    optimizer=optimizer,
                                    lr_scheduler=lr_scheduler,
                                    models_dir=cfg.models_dir,
                                    seed=cfg.seed,
                                    best_policy=cfg.best_policy,
                                    save=cfg.save,
                                   )
    if cfg.save:
        with open(os.path.join(checkpoint.version_dir, 'config.txt'), 'w') as f:
            f.writelines(str(cfg))

ptu.params(checkpoint.model)


print('checkpoint.train')
checkpoint.train(train_loader=train_loader,
                 val_loader=val_loader,
                 train_epochs=int(max(0, cfg.epochs - checkpoint.get_log())),
                 optimizer_params=cfg.optimizer_params,
                 prints=cfg.prints,
                 epochs_save=cfg.epochs_save,
                 epochs_evaluate_train=cfg.epochs_evaluate_train,
                 epochs_evaluate_validation=cfg.epochs_evaluate_validation,
                 max_iterations_train=cfg.max_iterations,
                 max_iterations_val=cfg.max_iterations,
                 device=device,
                 tqdm_bar=cfg.tqdm_bar,
                 save=cfg.save,
                 save_log=cfg.save_log,
                )

checkpoint = ptu.load_model(cpu_device, version=cfg.version, models_dir=cfg.models_dir, epoch='best')
checkpoint.model.eval()
for p in checkpoint.model.parameters():
    p.grad = None

torch.save(checkpoint.model, cfg.final_model_path)
model = torch.load(cfg.final_model_path, map_location=cpu_device)

print('training completed successfully')


