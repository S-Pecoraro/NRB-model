import torch
import numpy as np
import copy
from tqdm import tqdm
from torch.utils.data import DataLoader
import time
from customDataset import TrashDataset
from torch.utils.data.sampler import WeightedRandomSampler
import matplotlib as mlp

mlp.use('Agg')
import matplotlib.pyplot as plt
import itertools


def get_labels_and_class_counts(labels_list):
    """
    Calculates the counts of all unique classes.
    """
    labels = np.array(labels_list)
    _, class_counts = np.unique(labels, return_counts=True)
    return labels, class_counts


def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.plasma)
    plt.title("Confusion matrix")
    cbar = plt.colorbar()
    cbar.set_label('# of observations', rotation=90)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.min() + (cm.max() - cm.min()) / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] < threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure


class classModel:
    def __init__(self, device, data_dir, writer, workers, epochs, batch_size):
        self.writer = writer
        self.data_dir = data_dir
        self.device = device
        self.workers = workers
        self.num_epochs = epochs
        self.batch_size = batch_size
        self.image_datasets = None
        self.dataset_sizes = None
        self.ref_labels = None
        self.sampler = None
        self.nb_classes = None
        self.dataloaders = None

    def load_data(self, imgs_root_dir, train_data, val_data, ref_labels, data_transforms):
        self.ref_labels = ref_labels
        self.nb_classes = len(self.ref_labels)
        self.image_datasets = {
            'train': TrashDataset(pandas_df=train_data, root_dir=imgs_root_dir + 'train',
                                  transform=data_transforms['train']),
            'val': TrashDataset(pandas_df=val_data, root_dir=imgs_root_dir + 'val', transform=data_transforms['val'])}

        self.dataset_sizes = {x: len(self.image_datasets[x]) for x in ['train', 'val']}

    def get_datasets(self):
        return self.image_datasets

    def imbalanced_dataset(self):
        # Compute the dataset distribution
        train_targets, train_class_counts = get_labels_and_class_counts(
            self.image_datasets['train'].get_column_obs('cat_index'))
        weights = 1. / torch.tensor(train_class_counts, dtype=torch.float)

        samples_weights = weights[train_targets]
        print("\nTrain dataset distribution\n")
        for idx, (count, weight) in enumerate(zip(train_class_counts, weights)):
            print('Class {}: {} samples, {:.5} weight'.format(self.ref_labels.loc[idx, 'label'], count, weight))

        self.sampler = WeightedRandomSampler(
            weights=samples_weights,
            num_samples=len(samples_weights),
            replacement=True)

    def __set_dataloaders(self):
        if self.sampler:
            self.dataloaders = {
                'train': DataLoader(self.image_datasets['train'], batch_size=self.batch_size, sampler=self.sampler,
                                    num_workers=self.workers),
                'val': DataLoader(self.image_datasets['val'], batch_size=self.batch_size, shuffle=False,
                                  num_workers=self.workers)}
        else:
            self.dataloaders = {
                'train': DataLoader(self.image_datasets['train'], batch_size=self.batch_size, shuffle=True,
                                    num_workers=self.workers),
                'val': DataLoader(self.image_datasets['val'], batch_size=self.batch_size, shuffle=False,
                                  num_workers=self.workers)}

    def __draw_per_class_accuracy(self, confusion_matrix, epoch):
        print(f'\nConfusion matrix validation set epoch {epoch}\n')
        print(confusion_matrix)
        print('\n')
        accuracies = confusion_matrix.diag() / confusion_matrix.sum(1)
        for i, acc in enumerate(accuracies):
            print(f'Accuracy for class {self.ref_labels.label.loc[i]} is {acc}')

    def train_model(self, model, criterion, optimizer, scheduler):
        self.__set_dataloaders()
        since = time.time()
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        pbar = tqdm(self.num_epochs, total=self.num_epochs)  # progress bar
        for epoch in range(self.num_epochs):
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode
                    confusion_matrix = torch.zeros(self.nb_classes, self.nb_classes)

                running_loss = 0.0
                running_corrects = 0
                batches = len(self.dataloaders[phase])
                current_batch = 0
                obs = 0
                # Iterate over data.
                for inputs, labels in self.dataloaders[phase]:
                    obs += inputs.size(0)
                    current_batch += 1
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                        if phase == 'val':
                            for t, p in zip(labels.view(-1), preds.view(-1)):
                                confusion_matrix[t.long(), p.long()] += 1
                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    # print
                    mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
                    s = f'Phase: {phase} --' \
                        f' Loss: {running_loss / obs};' \
                        f' Accuracy: {running_corrects.double() / obs};' \
                        f' Batches: {current_batch}/{batches},' \
                        f' GPU usage: {mem}'
                    pbar.set_description(s)

                if phase == 'val':
                    self.__draw_per_class_accuracy(confusion_matrix, epoch)
                    cm_fig = plot_confusion_matrix(confusion_matrix.numpy(), self.ref_labels['label'])
                    self.writer.add_figure("confusion matrix on validation set", cm_fig, global_step=epoch)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset_sizes[phase]

                print('\n--{}-- Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                # Save metrics
                if phase == 'train':
                    self.writer.add_scalar('Loss/train', epoch_loss, epoch)
                    self.writer.add_scalar('Accuracy/train', epoch_acc, epoch)
                else:
                    self.writer.add_scalar('Loss/test', epoch_loss, epoch)
                    self.writer.add_scalar('Accuracy/test', epoch_acc, epoch)

                # deep copy the model and save checkpoints
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_model_wts, 'best.pt')
                if phase == 'val':
                    torch.save(model.state_dict(), 'last.pt')

            pbar.update(1)
            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model
