
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from pathlib import Path
import os 
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms
from PIL import Image
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchmetrics import F1Score, Recall, Precision, Accuracy


class InsectDataset(Dataset):
    def __init__(self, dataset_path=None, image_path=None, transform=None, dataframe=None, class_to_idx=None):
        super().__init__()
        self.transform = transform
        self.image_path = image_path
        
        
        if dataframe is not None:
            self.dataset = dataframe.copy()
            self.dataset.reset_index(drop=True, inplace=True)
        else:
            self.dataset_path = Path(dataset_path)
            dfs = []
            for f in self.dataset_path.iterdir():
                if f.is_file() and f.suffix == '.csv':
                    temp_df = pd.read_csv(f)
                    temp_df['Filename'] = f.with_suffix('.jpg').name
                    temp_df['image_path'] = os.path.join(image_path, f.with_suffix('.jpg').name)
                    dfs.append(temp_df)
                    
            if dfs:
                self.dataset = pd.concat(dfs, ignore_index=True)
                
            self.dataset['class_name'] = (self.dataset['class_name']
                                          .str.lower()               
                                          .str.replace(' ', '_')      )
            self.dataset['class_classification'] = self.dataset['class_name'].map(class_to_idx)

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, idx):
        item = self.dataset.iloc[idx]
        image_path, bbox_x, bbox_y, bbox_w, bbox_h = item['image_path'], item['bbox_x'], item['bbox_y'], item['bbox_w'], item['bbox_h']
        
        image = Image.open(image_path).convert('RGB')
        
        cropped_image = transforms.functional.crop(image, top=int(bbox_y), left=int(bbox_x), height=int(bbox_h), width=int(bbox_w))
               
        if self.transform:
            img = self.transform(cropped_image)
        else:
            composer = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((299, 299)),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            img = composer(cropped_image)
            
        return img, item['class_classification'], item['image_path']

class InsectDataModule(pl.LightningDataModule):
    def __init__(self, dataset_path, image_path, train_transform, val_transform, batch_size):
        super().__init__()
        self.dataset_path = dataset_path
        self.image_path = image_path
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.batch_size = batch_size

    def setup(self, stage=None):

        full_dataset = InsectDataset(dataset_path=self.dataset_path, image_path=self.image_path)
        df = full_dataset.dataset.copy()
        
        train_list, val_list, test_list = [], [], []
        
        for _, group in df.groupby('class_classification'):
            group = group.sample(frac=1, random_state=42).reset_index(drop=True)
            n_samples = len(group)
            
            if n_samples == 1:
                train_list.append(group)
                val_list.append(group)
                test_list.append(group)
            elif n_samples == 2:
                train_list.append(group.iloc[[0]])
                val_list.append(group.iloc[[1]])
                test_list.append(group.iloc[[1]])
            else:
                n_val_test = max(2, int(0.28 * n_samples))
                n_train = n_samples - n_val_test
                
                train_list.append(group.iloc[:n_train])
                val_list.append(group.iloc[n_train:])
        
        train_df = pd.concat(train_list).sample(frac=1, random_state=42).reset_index(drop=True)
        val_df = pd.concat(val_list).reset_index(drop=True)       
        resampled_dfs = []
        for class_idx, group in train_df.groupby('class_classification'):
            resampled_group = group.sample(n=100, replace=(len(group) < 100), random_state=42)
            resampled_dfs.append(resampled_group)
        
        train_df_resampled = pd.concat(resampled_dfs).sample(frac=1, random_state=42)

        self.train_dataset = InsectDataset(image_path=self.image_path, transform=self.train_transform, dataframe=train_df_resampled)
        self.val_dataset = InsectDataset(image_path=self.image_path, transform=self.val_transform, dataframe=val_df)
        self.test_dataset = torchvision.datasets.ImageFolder(root='/cluster/projects/nn10058k/SquidlePlusProject/insects_proj/official_test_images', transform=self.val_transform)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, batch_size=self.batch_size, num_workers=0)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False, batch_size=self.batch_size, num_workers=0)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, shuffle=False, batch_size=self.batch_size, num_workers=0)

def image_to_tb(self, batch, batch_idx, step_name):
    tensorboard = self.logger.experiment
    if batch_idx % 200 != 0:
        return
    images = []
    for index, image in enumerate(batch[0]):
        image = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )(image)
        images.append(image)
    if images:
        img_grid = torchvision.utils.make_grid(images)
        tensorboard.add_image('Batch images ' + step_name, img_grid, batch_idx)

        
def log_to_graph(self, value, var, name, global_step):
    self.logger.experiment.add_scalars(var, {name: value}, global_step)


class InsectsModel(pl.LightningModule):
    def __init__(self, task, num_classes, model, type, lr=1e-3, w2_decay=1e-5):
        super().__init__()
        self.accuracy_train_macro = Accuracy(task=task, num_classes=num_classes, average='macro')
        self.f1_score_train = F1Score(task=task, num_classes=num_classes, average = 'none')
        self.recall_train = Recall(task = task, num_classes = num_classes, average = 'none')
        self.precision_train = Precision(task = task, num_classes = num_classes, average = 'none')
        self.accuracy_train = Accuracy(task = task, num_classes=num_classes, average='none')
        self.model_to_use = model
        
        self.accuracy_val_macro = Accuracy(task=task, num_classes=num_classes, average='macro')
        self.f1_score_val = F1Score(task=task, num_classes=num_classes, average = 'none')
        self.recall_val = Recall(task = task, num_classes = num_classes, average = 'none')
        self.precision_val = Precision(task = task, num_classes = num_classes, average = 'none')
        self.accuracy_val = Accuracy(task = task, num_classes=num_classes, average='none')
        
        self.accuracy_test_macro = Accuracy(task=task, num_classes=num_classes, average='macro')
        self.f1_score_test = F1Score(task=task, num_classes=num_classes, average = 'none')
        self.recall_test = Recall(task = task, num_classes = num_classes, average = 'none')
        self.precision_test = Precision(task = task, num_classes = num_classes, average = 'none')
        self.accuracy_test = Accuracy(task = task, num_classes=num_classes, average='none') 
        self.save_hyperparameters()
        self.loss = nn.CrossEntropyLoss()
        self.train_loss = []
        self.validation_loss = []
        self.test_loss = []

        if type == 'CNN_based':
            try:
                self.model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
            except Exception as e:
                print(f"Warning: Could not load from hub ({e}). Using alternative model...")
                from torchvision.models import inception_v3
                self.model = inception_v3(pretrained=False)
            self.model.AuxLogits = None 
            hidden_layers = self.model.fc.in_features
            self.model.fc = nn.Identity()
            self.classifier_head = nn.Linear(hidden_layers, num_classes)

    def forward(self, x):
        x = self.model(x)
        if hasattr(x, 'logits'):
            x = x.logits
        x = self.classifier_head(x)
        return x
        
    def training_step(self, batch):
        features, target, img_path = batch
        pred = self(features)
        loss = self.loss(pred, target)

        self.f1_score_train(pred, target)
        self.recall_train(pred, target)
        self.precision_train(pred, target)
        self.accuracy_train(pred, target)

        self.train_loss.append(loss.item())

        image_to_tb(self, batch, self.current_epoch, 'train')

        return loss
        
    def on_train_epoch_end(self):
        name = 'train'

        log_to_graph(self, np.mean(self.train_loss),           'loss',      name, self.current_epoch)

        accuracy_per_class = self.accuracy_train.compute()
        for i, class_accuracy in enumerate(accuracy_per_class):
            log_to_graph(self, class_accuracy.item(), f'accuracy_score_class_{i}', name, self.current_epoch)

        precision_per_class = self.precision_train.compute()
        for i, class_precision in enumerate(precision_per_class):
            log_to_graph(self, class_precision.item(), f'precision_score_class_{i}', name, self.current_epoch)

        f1_per_class = self.f1_score_train.compute()
        for i, class_f1 in enumerate(f1_per_class):
            log_to_graph(self, class_f1.item(), f'f1_score_class_{i}', name, self.current_epoch)


        recall_per_class = self.recall_train.compute()
                
        for i, class_recall in enumerate(recall_per_class):
            log_to_graph(self, class_recall.item(), f'recall_score_class_{i}', name, self.current_epoch)

        train_accuracy_macro = self.accuracy_train_macro.compute()
        log_to_graph(self, train_accuracy_macro.item(), 'accuracy_macro', name, self.current_epoch)


        self.train_loss = []
        self.accuracy_train.reset()
        self.f1_score_train.reset()
        self.precision_train.reset()
        self.recall_train.reset()
        self.accuracy_train_macro.reset()

    def validation_step(self, batch):
        features, target, img_path = batch
        pred = self(features)
        loss = self.loss(pred, target)

        self.f1_score_val(pred, target)
        self.recall_val(pred, target)
        self.precision_val(pred, target)
        self.accuracy_val(pred, target)

        self.validation_loss.append(loss.item())

        image_to_tb(self, batch, self.current_epoch, 'val')

        return loss
    
    def on_validation_epoch_end(self):
        name = 'val'

        log_to_graph(self, np.mean(self.validation_loss), 'loss', name, self.current_epoch)

        accuracy_per_class = self.accuracy_val.compute()
        for i, class_accuracy in enumerate(accuracy_per_class):
            log_to_graph(self, class_accuracy.item(), f'accuracy_class_{i}', name, self.current_epoch)

        precision_per_class = self.precision_val.compute()
        for i, class_precision in enumerate(precision_per_class):
            log_to_graph(self, class_precision.item(), f'precision_class_{i}', name, self.current_epoch)

        f1_per_class = self.f1_score_val.compute()
        for i, class_f1 in enumerate(f1_per_class):
            log_to_graph(self, class_f1.item(), f'f1_score_class_{i}', name, self.current_epoch)

        recall_per_class = self.recall_val.compute()
        for i, class_recall in enumerate(recall_per_class):
            log_to_graph(self, class_recall.item(), f'recall_class_{i}', name, self.current_epoch)

        val_macro_accuracy = self.accuracy_val_macro.compute()
        log_to_graph(self, val_macro_accuracy.item(), 'accuracy_macro', name, self.current_epoch)

        self.validation_loss = []
        self.accuracy_val.reset()
        self.f1_score_val.reset()
        self.precision_val.reset()
        self.recall_val.reset()
        self.accuracy_val_macro.reset()
    def test_step(self, batch):
        features, target, img_path = batch
        pred = self(features)
        loss = self.loss(pred, target)

        self.f1_score_test(pred, target)
        self.recall_test(pred, target)
        self.precision_test(pred, target)
        self.accuracy_test(pred, target)

        self.test_loss.append(loss.item())

        image_to_tb(self, batch, self.current_epoch, 'test')

        return loss
    
    def on_test_epoch_end(self):
        name = 'test'

        log_to_graph(self, np.mean(self.test_loss), 'loss', name, self.current_epoch)

        accuracy_per_class = self.accuracy_test.compute()
        for i, class_accuracy in enumerate(accuracy_per_class):
            log_to_graph(self, class_accuracy.item(), f'accuracy_class_{i}', name, self.current_epoch)

        precision_per_class = self.precision_test.compute()
        for i, class_precision in enumerate(precision_per_class):
            log_to_graph(self, class_precision.item(), f'precision_class_{i}', name, self.current_epoch)

        f1_per_class = self.f1_score_test.compute()
        for i, class_f1 in enumerate(f1_per_class):
            log_to_graph(self, class_f1.item(), f'f1_score_class_{i}', name, self.current_epoch)

        recall_per_class = self.recall_test.compute()
        for i, class_recall in enumerate(recall_per_class):
            log_to_graph(self, class_recall.item(), f'recall_class_{i}', name, self.current_epoch)

        test_macro_accuracy = self.accuracy_test_macro.compute()
        log_to_graph(self, test_macro_accuracy.item(), 'accuracy_macro', name, self.current_epoch)
        self.test_loss = []
        self.accuracy_test.reset()
        self.f1_score_test.reset()
        self.precision_test.reset()
        self.recall_test.reset()
        self.accuracy_test_macro.reset()
    def configure_optimizers(self):
        parameters_model = [p for p in self.model.parameters() if p.requires_grad is True]
        parameters_clh = [p for p in self.classifier_head.parameters() if p.requires_grad is True]
        parameters = parameters_model + parameters_clh
        optimizer = torch.optim.AdamW(parameters, lr=self.hparams.lr, weight_decay=self.hparams.w2_decay)
        return optimizer

       
if __name__ == '__main__':
    tb_logger = TensorBoardLogger(save_dir=r'/cluster/projects/nn10058k/SquidlePlusProject/insects_proj/ShadowGraph/ShadowGraph/output_dir', name='lightning_logs')

    trainer = pl.Trainer(
            accelerator='auto',
            logger=tb_logger,
            default_root_dir=r'/cluster/projects/nn10058k/SquidlePlusProject/insects_proj/ShadowGraph/ShadowGraph/output_dir',
            enable_checkpointing=True,
            max_epochs=10,
            precision='16-mixed',
            log_every_n_steps=50,
        )
        
    img_path = r'/cluster/projects/nn10058k/SquidlePlusProject/insects_proj/ShadowGraph/ShadowGraph/data/morphocluster-free-number-of-clusters-200imgs/200samples/'
    dataset_path = r'/cluster/projects/nn10058k/SquidlePlusProject/insects_proj/ShadowGraph/ShadowGraph/data/200imgsResults_annotation_results/200imgsResults_annotation_results/200imgsResults_annotations_per_image/'

    train_transform = transforms.Compose([
        transforms.RandomRotation(degrees=45), 
        transforms.RandomHorizontalFlip(),     
        transforms.ToTensor(),
        transforms.Resize((299, 299)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((299, 299)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    datamodule = InsectDataModule(
        dataset_path, 
        img_path, 
        train_transform=train_transform, 
        val_transform=val_transform, 
        batch_size=64
    )
    
    num_classes = 6
    class_to_idx = {
        'cladoceramorpha': 0,
        'copepoda': 1,
        'leptodora_kindtii': 2,
        'plant_or_algae': 3,
        'rotifera': 4,
        'unidentified_organism': 5
    }
    model = InsectsModel(task='multiclass', num_classes=num_classes, type='CNN_based', model='VGG', lr=0.001, w2_decay=0.01, class_to_idx = class_to_idx)
    
    trainer.fit(datamodule=datamodule, model=model)
