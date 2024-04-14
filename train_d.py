import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import pandas as pd
import random
from dataclasses import dataclass
import numpy as np
import torch
import transformers
import json
import pickle
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizer
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, AUROC, F1Score, Precision, Recall
import medspacy
from medspacy.section_detection import SectionRule
from model.train import train_model


class TextDataset(Dataset):
    """
    this class is very closely based on the huggingface tutorial implementation
    """
    def __init__(self, dataframe, tokenizer, max_len, target_cols: list[str], id_col: str = 'row_id',
                 text_col: str = 'TEXT'):
        self.tokenizer = tokenizer
        # self.data = dataframe
        self.text_id_list = list(dataframe[id_col])
        self.text_list = list(dataframe[text_col])
        self.label_list = self._get_labels(dataframe, target_cols)
        self.max_len = max_len
        
    def _get_labels(self, dataframe, target_col_list):
        label_list_container = list()
        
        for target_col in target_col_list:
            label_list_container.append(
                list(dataframe[target_col].astype(float))
            )
            
        return list(zip(*label_list_container))

    def __len__(self):
        # get length of dataset (required for dataloader)
        return len(self.text_list)

    def __getitem__(self, idx):
        # extract text and label
        text = str(self.text_list[idx])
        label = self.label_list[idx]

        # tokenize text
        encoded_text = self.tokenizer.encode_plus(
            text,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True
        )

        # unpack encoded text
        ids = encoded_text['input_ids']
        attention_mask = encoded_text['attention_mask']
        token_type_ids = encoded_text["token_type_ids"]

        # wrap outputs in dict
        out_dict = {
            'text_id_list': self.text_id_list,
            'id_tensor': torch.tensor(ids, dtype=torch.long),
            'mask_tensor': torch.tensor(attention_mask, dtype=torch.long),
            'token_type_tensor': torch.tensor(token_type_ids, dtype=torch.long),
            'label_tensor': torch.tensor(label, dtype=torch.float)
        }

        return out_dict


def seed_script(seed: int):
    # set torch seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # set numpy seed
    np.random.seed(seed)
    print(f"seed set to {seed}...")

def get_dataloader(dataset, batch_size, shuffle: bool = True,
                   pin_memory: bool = True, num_workers: int = 0,
                   prefetch_factor: int or None = None):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor
    )
    return dataloader

class CustomRoberta(torch.nn.Module):
    """
    model subclass to define the RoBERTa architecture, also closely based on
    the huggingface tutorial implementation
    """
    def __init__(self, drop_percent, num_classes, pt_model_name: str = 'roberta-base'):
        super().__init__()
        self.base_model = RobertaModel.from_pretrained(pt_model_name)
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(drop_percent)
        self.classifier = torch.nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # get outputs from base model
        base_outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        # extract hidden state from roberta base outputs
        hidden_state = base_outputs[0]
        x = hidden_state[:, 0]

        # define the linear layer preceding the classifier
        # and apply ReLU activation to its outputs
        x = self.pre_classifier(x)
        x = torch.nn.ReLU()(x)

        # define the dropout layer and classifier
        x = self.dropout(x)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    print(os.environ["CUDA_VISIBLE_DEVICES"])

    MAX_LEN = 256
    BATCH_SIZE = 128
    SEED = 123129
    TASK_TYPE = 'multilabel'
    NUM_CLASSES = 2
    NUM_LABELS = 3
    AVERAGE_STRATEGY = 'global'
    LEARNING_RATE = 1e-7
    EPOCHS = 40
    MODEL_NAME = 'roberta-test-d'
    
    # get cpu, gpu or mps device for training.
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"using device: {device}...")

    train_set_df = pd.read_csv("data/train_sample.csv")
    from sklearn.model_selection import train_test_split

    # 70/15/15 split
    # split into train/val/test
    train_df, val_test_df = train_test_split(
        train_set_df,
        test_size=0.3,
        random_state=SEED
    )
    
    val_df, test_df = train_test_split(
        val_test_df,
        test_size=0.5,
        random_state=SEED
    )
    
    # load roberta base as a tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base', truncation=True, do_lower_case=True)
    target_col_list = ['environment_binary', 'community_binary', 'alcohol_binary']
    
    # load dataframes into dataset objects
    train_ds = TextDataset(
        dataframe=train_df, 
        tokenizer=tokenizer, 
        max_len=MAX_LEN, 
        target_cols=target_col_list,
        text_col='social_history'
    )
    val_ds = TextDataset(
        dataframe=val_df, 
        tokenizer=tokenizer, 
        max_len=MAX_LEN, 
        target_cols=target_col_list,
        text_col='social_history'
    )
    test_ds = TextDataset(
        dataframe=test_df, 
        tokenizer=tokenizer, 
        max_len=MAX_LEN, 
        target_cols=target_col_list,
        text_col='social_history'
    )
    
    # load datasets into loaders
    train_loader = get_dataloader(train_ds, BATCH_SIZE)
    val_loader = get_dataloader(val_ds, BATCH_SIZE)
    test_loader = get_dataloader(test_ds, BATCH_SIZE)
    
    # define metric collection
    metric_collection = MetricCollection({
        'acc': Accuracy(
            task=TASK_TYPE, 
            num_labels=NUM_LABELS, 
            num_classes=NUM_CLASSES, 
            multidim_average=AVERAGE_STRATEGY
        ),
        'auc': AUROC(
            task=TASK_TYPE, 
            num_labels=NUM_LABELS, 
            num_classes=NUM_CLASSES
        ),
        'prec': Precision(
            task=TASK_TYPE, 
            num_labels=NUM_LABELS, 
            num_classes=NUM_CLASSES, 
            multidim_average=AVERAGE_STRATEGY
        ),
        'rec': Recall(
            task=TASK_TYPE, 
            num_labels=NUM_LABELS, 
            num_classes=NUM_CLASSES, 
            multidim_average=AVERAGE_STRATEGY
        ),
        'f1': F1Score(
            task=TASK_TYPE, 
            num_labels=NUM_LABELS, 
            num_classes=NUM_CLASSES, 
            multidim_average=AVERAGE_STRATEGY
        )
    })
    metric_collection.to(device)

    # instantiate model
    model = CustomRoberta(0.4, 3)
    model.to(device)
    
    # define loss and optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    loader_dict = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    
    seed_script(SEED)
    
    training_history = train_model(
        device=device, 
        model=model, 
        loader_dict=loader_dict, 
        metric_collection=metric_collection, 
        criterion=criterion,
        optimizer=optimizer, 
        n_epochs=EPOCHS, 
        save_dir=MODEL_NAME, 
        monitor_metric="val_loss"
    )

    with open(f'{MODEL_NAME}-train-log.pickle', 'wb') as f:
        pickle.dump(training_history, f)