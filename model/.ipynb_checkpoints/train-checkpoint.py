import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, AUROC, F1Score, Precision, Recall

@dataclass
class EpochMetrics:
    """
    Dataclass to store the metrics for a single epoch.
    """
    epoch: int
    train: dict or None = None
    val: dict or None = None
    test: dict or None = None
        

def seed_torch(seed: int):
    """
    Seed all torch random number generators and set the deterministic flag.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def train_model(device, model, loader_dict, metric_collection, criterion,
                optimizer, n_epochs: int, save_dir: str or None = None,
                monitor_metric: str = "val_loss"):
    """
    Function to train a model.

    device: torch.device
        device to train on
    model: torch.nn.Module
        model to train
    loader_dict: dict
        dictionary of 'train', 'val', and 'test' dataloaders
    metric_collection: MetricCollection
        metric collection to store metrics
    criterion: torch.nn.Module
        loss function to use
    optimizer: torch.optim.Optimizer
        optimizer to use
    n_epochs: int
        number of epochs to train for
    """
    train_log_list = list()

    if save_dir is not None:
        # initialize the best metric based on what the monitor metric is
        # (and if it should be maximized or minimized)
        monitor_is_loss = monitor_metric.split('_')[-1] == 'loss'

        if monitor_is_loss:
            best_metric = np.inf
        else:
            best_metric = -np.inf

        # if save dir doesn't exist, make it
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        model_save_path = os.path.join(save_dir, 'best_model.pth')
    else:
        best_metric = None
        monitor_is_loss = None

    # iterate over epochs
    for epoch in range(n_epochs):
        # define an EpochMetrics object to store our metrics in
        epoch_metrics_object = EpochMetrics(epoch=epoch)

        print(f"\nEpoch {epoch} {'-' * 40}")

        # perform train/val phases
        for phase in ['train', 'val']:
            phase_metrics_dict = epoch_phase(
                phase=phase,
                device=device,
                model=model,
                loader_dict=loader_dict,
                metric_collection=metric_collection,
                criterion=criterion,
                optimizer=optimizer
            )

            # add current phase dict to the epoch metrics object
            setattr(epoch_metrics_object, phase, phase_metrics_dict)
            
            # if we want to save the model and the phase is val
            if (phase == "val") and (save_dir is not None):
                # extract the candidate for the new best metric
                best_candidate = phase_metrics_dict[monitor_metric]
                
                # check if the best candidate from this epoch is better
                metric_is_better = check_if_better(
                    best_candidate, 
                    best_metric,
                    monitor_is_loss
                )

                # save the model weights if the current val monitor metric is the best so far
                if (save_dir is not None) & metric_is_better:
                    best_metric = phase_metrics_dict[monitor_metric]
                    
                    print(f"saving model with best {monitor_metric} '{best_metric:.4f}'...")
                    
                    # get checkpoint
                    torch.save(model.state_dict(), model_save_path)

        # on last epoch, perform test phase
        if epoch == n_epochs - 1:
            phase_metrics_dict = epoch_phase(
                phase="test",
                device=device,
                model=model,
                loader_dict=loader_dict,
                metric_collection=metric_collection,
                criterion=criterion,
                optimizer=optimizer
            )

            # add test phase dict to the epoch metrics object
            setattr(epoch_metrics_object, "test", phase_metrics_dict)

        # add current epoch metrics object to the log list
        train_log_list.append(epoch_metrics_object)

    return train_log_list

def check_if_better(best_candidate, best_metric, monitor_is_loss: bool = False):
    """
    function to check if the best metric candidate is better than the previous
    best
    """
    if monitor_is_loss:
        return best_candidate < best_metric
    else:
        return best_candidate > best_metric

def epoch_phase(phase, device, model, loader_dict, metric_collection, criterion, optimizer):
    """
    Function to perform a single epoch phase.
    phase: str
        phase to perform (train, val, test)
    device: torch.device
        device to train on
    model: torch.nn.Module
        model to train
    loader_dict: dict
        dictionary of 'train', 'val', and 'test' dataloaders
    metric_collection: MetricCollection
        metric collection to store metrics
    criterion: torch.nn.Module
        loss function to use
    optimizer: torch.optim.Optimizer
        optimizer to use
    """
    # this function is a simplification of my normal torch loop
    # and was originally based on the implementation here:
    # https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

    # select current dataloader from the loader dict
    phase_loader = loader_dict[phase]
    phase_size = len(phase_loader)

    # if phase is training, set model to training mode
    # otherwise set model to eval mode
    if phase == 'train':
        model.train()
    else:
        model.eval()

    # init the running loss for the epoch
    running_loss = 0.0

    # iterate over data in current phase loader
    with tqdm(phase_loader, unit="batch", total=phase_size) as epoch_iter:
        for batch, data in enumerate(epoch_iter):
            # unpack data and send them to the device
            id_tensor = data['id_tensor'].to(device)
            mask_tensor = data['mask_tensor'].to(device)
            token_type_tensor = data['token_type_tensor'].to(device)
            label_tensor = data['label_tensor'].to(device)
            
            # X, y = data
            # X, y = X.to(device), y.to(device, dtype=torch.float)

            # zero the parameter gradients
            optimizer.zero_grad()

            # track history if in train
            with torch.set_grad_enabled(phase == 'train'):
                # get model outputs for the current batch
                preds = model(X)
                preds = preds.squeeze()

                # get the loss for the current batch preds and update
                # the running loss
                loss = criterion(preds, y)
                running_loss += loss.item()

                # update metric collection
                metric_collection.update(preds, y)

                # if train phase, backpropogate and step with the optimizer
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # update metrics in tqdm display after each 10% chunk
            # or if in val/test phase, update on the last batch
            if ((phase == 'train') & (batch % (max(phase_size // 10, 1)) == 0)) |\
            ((phase != 'train') & (batch == (phase_size - 1))):
                phase_metrics = metric_collection.compute()

                phase_metrics_dict = format_metrics_dict(
                    running_loss / (batch + 1),
                    phase_metrics,
                    phase
                )
                epoch_iter.set_postfix(phase_metrics_dict)

    # reset the metric collection at the end of the current phase
    metric_collection.reset()

    return phase_metrics_dict

def format_metrics_dict(loss, metrics_dict, set_name: str):
    """
    Function to format the metric dictionary output after each phase.
    loss: float
        the running loss (averaged per batch) divided by the current batch number
    metrics_dict: dict
        metrics dictionary output by MetricCollection.compute()
    set_name: str
        phase name (train, val, test)
    """
    # init out_metrics_dict and format the loss
    out_metrics_dict = {}
    out_metrics_dict[f'{set_name}_loss'] = loss

    # iterate over MetricCollection metrics and add them to the dict
    for k, v in metrics_dict.items():
        out_metrics_dict[f'{set_name}_{k}'] = v.item()

    return out_metrics_dict

def tuning_wrapper(dropout_proportion, arch_num, learning_rate, 
                   model_func, device, loader_dict, n_epochs: int, 
                   save_dir: str or None, monitor_metric: str, seed: int = 13):
    seed_torch(seed)
    
    # define model
    model = model_func(
        dropout_prop=dropout_proportion, 
        arch_num=str(int(arch_num))
    )
    model = model.to(device)

    # define metric collection
    TASK_TYPE = 'binary'
    NUM_CLASSES = 2
    metric_collection = MetricCollection({
        'acc': Accuracy(task=TASK_TYPE, num_classes=NUM_CLASSES),
        'auc': AUROC(task=TASK_TYPE, num_classes=NUM_CLASSES),
        'prec': Precision(task=TASK_TYPE, num_classes=NUM_CLASSES),
        'rec': Recall(task=TASK_TYPE, num_classes=NUM_CLASSES),
        'f1': F1Score(task=TASK_TYPE, num_classes=NUM_CLASSES, average='micro')
    })
    metric_collection.to(device)

    # define loss and optimizer
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # train the model and return the training history
    history = train_model(
        device=device, 
        model=model, 
        loader_dict=loader_dict, 
        metric_collection=metric_collection, 
        criterion=criterion,
        optimizer=optimizer, 
        n_epochs=n_epochs, 
        save_dir=save_dir,
        monitor_metric=monitor_metric
    )
    return history