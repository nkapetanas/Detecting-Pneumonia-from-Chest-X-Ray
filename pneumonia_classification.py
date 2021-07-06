import copy
import glob
import os
import random
import time

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from PIL import Image
from matplotlib import pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from tqdm.auto import tqdm

TRAIN_NORMAL_PATH = "./data/chest_xray/train/NORMAL/"
TRAIN_PNEUMONIA_PATH = "./data/chest_xray/train/PNEUMONIA/"

TEST_NORMAL_PATH = "./data/chest_xray/test/NORMAL/"
TEST_PNEUMONIA_PATH = "./data/chest_xray/test/PNEUMONIA/"

TRAIN_NORMAL_DATA_PATH = glob.glob("./data/chest_xray/train/NORMAL/*.jpeg")
TRAIN_PNEUMONIA_DATA_PATH = glob.glob("./data/chest_xray/train/PNEUMONIA/*.jpeg")

TEST_NORMAL_DATA_PATH = glob.glob("./data/chest_xray/test/NORMAL/*.jpeg")
TEST_PNEUMONIA_DATA_PATH = glob.glob("./data/chest_xray/test/PNEUMONIA/*.jpeg")

EPOCHS = 5
LEARNING_RATE = 3e-3
BATCH_SIZE = 15
IMAGE_DIMENSIONS = (500, 500)

train_paths = TRAIN_NORMAL_DATA_PATH + TRAIN_PNEUMONIA_DATA_PATH
test_paths = TEST_NORMAL_DATA_PATH + TEST_PNEUMONIA_DATA_PATH

train_labels = [0] * len(TRAIN_NORMAL_DATA_PATH) + [1] * len(TRAIN_PNEUMONIA_DATA_PATH)
test_targets = [0] * len(TEST_NORMAL_DATA_PATH) + [1] * len(TEST_PNEUMONIA_DATA_PATH)


def dataset_stats(path_class1, path_class2, class_label1, class_label2, description):
    sample_size = pd.DataFrame.from_dict(
        {class_label1: [len([os.path.join('%s' % path_class1, filename)
                             for filename in os.listdir('%s' % path_class1)])],

         class_label2: [len([os.path.join('%s' % path_class2, filename)
                             for filename in os.listdir('%s' % path_class2)])]})

    sns.barplot(data=sample_size).set_title(description, fontsize=20)
    plt.show()


def data_preprocess():
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=IMAGE_DIMENSIONS),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomGrayscale(),
        transforms.RandomAffine(translate=(0.05, 0.05), degrees=0),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=IMAGE_DIMENSIONS),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomGrayscale(),
        transforms.RandomAffine(translate=(0.05, 0.05), degrees=0),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return train_transform, test_transform


def compute_performance_metrics(cm):
    tn, fp, fn, tp = cm.ravel()
    accuracy = (np.array(predictions) == np.array(y_true)).sum() / len(y_predictions)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * ((precision * recall) / (precision + recall))
    print("Accuracy of the model is {:.2f}".format(accuracy))
    print("Recall of the model is {:.2f}".format(recall))
    print("Precision of the model is {:.2f}".format(precision))
    print("F1 Score of the model is {:.2f}".format(f1))


def show_random_images():
    path_random_normal = random.choice(TRAIN_NORMAL_DATA_PATH)
    path_random_abnormal = random.choice(TRAIN_PNEUMONIA_DATA_PATH)

    plt.figure(figsize=(10, 10))

    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(Image.open(path_random_normal).convert("LA"))
    ax1.set_title("Normal X-ray")

    ax2 = plt.subplot(1, 2, 2)
    ax2.imshow(Image.open(path_random_abnormal).convert("LA"))
    ax2.set_title("Abnormal X-ray")

    plt.show()


class ChestXRayDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.labels[index]
        label = torch.tensor([label])

        return image, label


class Resnet_18_Model(nn.Module):
    def __init__(self, pretrained=True):
        super(Resnet_18_Model, self).__init__()
        self.backbone = resnet18(pretrained=pretrained)
        self.fc = nn.Linear(in_features=512, out_features=1)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)

        x = x.view(x.size(0), 512)
        x = self.fc(x)

        return x


def compute_confusion_matrix(y_true, y_predictions):
    cm = confusion_matrix(y_true, y_predictions)
    plt.figure()
    plot_confusion_matrix(cm, figsize=(12, 8), cmap=plt.cm.Blues)
    plt.xticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)
    plt.yticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)
    plt.xlabel('Predicted Label', fontsize=18)
    plt.ylabel('True Label', fontsize=18)
    plt.show()

    return cm


def train(model, criterion, optimizer, num_epochs, device="cpu"):
    time_started = time.time()

    highest_accuracy = 0

    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in tqdm(range(num_epochs), leave=False):
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            loss = 0.0
            correct_predictions = 0

            for i, (inputs, targets) in tqdm(enumerate(dataloaders[phase]), leave=False, total=len(dataloaders[phase])):

                inputs = inputs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)

                    predictions = outputs.sigmoid() > 0.5

                    loss = criterion(outputs, targets.float())

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                loss += loss.item() * inputs.size(0)

                correct_predictions += torch.sum(predictions == targets.data)

                if (i % logging_steps[phase] == 0) & (i > 0):
                    avg_loss = loss / ((i + 1) * batch_sizes[phase])
                    avg_acc = correct_predictions / ((i + 1) * batch_sizes[phase])

                    print(f"[{phase}]: {epoch + 1} / {num_epochs} | loss : {avg_loss} | acc : {avg_acc}")

            epoch_loss = loss / dataset_sizes[phase]
            epoch_accuracy = correct_predictions.double() / dataset_sizes[phase]

            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_accuracy))

            if phase == "val" and epoch_accuracy > highest_accuracy:
                highest_accuracy = epoch_accuracy
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - time_started

    print(f"training took {time_elapsed} seconds")

    model.load_state_dict(best_model_wts)

    return model


if __name__ == '__main__':

    dataset_stats(TRAIN_NORMAL_PATH, TRAIN_PNEUMONIA_PATH, 'Normal', 'Pneumonia', 'Training Set Data Stats')
    dataset_stats(TEST_NORMAL_PATH, TEST_PNEUMONIA_PATH, 'Normal', 'Pneumonia', 'Test Set Data Stats')

    train_paths, valid_paths, train_labels, valid_labels = train_test_split(train_paths, train_labels,
                                                                            stratify=train_labels)

    train_transform, test_transform = data_preprocess()

    train_dataset = ChestXRayDataset(train_paths, train_labels, train_transform)
    valid_dataset = ChestXRayDataset(valid_paths, valid_labels, test_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=5, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, num_workers=5, shuffle=False)

    dataloaders = {
        "train": train_dataloader,
        "val": valid_dataloader
    }

    logging_steps = {
        "train": len(dataloaders["train"]) // 10,
        "val": len(dataloaders["val"]) // 10
    }

    dataset_sizes = {
        "train": len(train_dataset),
        "val": len(valid_dataset)
    }

    batch_sizes = {
        "train": BATCH_SIZE,
        "val": BATCH_SIZE
    }

    model = Resnet_18_Model(pretrained=True)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    model = train(model, criterion, optimizer, EPOCHS)

    test_paths = glob.glob("./data/chest_xray/test/NORMAL/*.jpeg") + glob.glob(
        "./data/chest_xray/test/PNEUMONIA/*.jpeg")

    test_targets = [0] * len(glob.glob("./data/chest_xray/test/NORMAL/*.jpeg")) + [1] * len(
        glob.glob("./data/chest_xray/test/PNEUMONIA/*.jpeg"))

    print("The test data lengths are: Data: " + str(len(test_paths)) + " Targets: " + str(len(test_targets)))

    test_dataset = ChestXRayDataset(test_paths, test_targets, test_transform)

    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, drop_last=False)

    y_predictions = list()
    y_true = list()

    for i, (tensors, targets) in tqdm(enumerate(test_dataloader), leave=False, total=len(test_dataloader)):
        with torch.no_grad():
            predictions = model(tensors)
            predictions = predictions.sigmoid()
            predictions = predictions > 0.5

            y_predictions.append(predictions)

            y_true.append(targets)

    y_predictions = torch.cat(y_predictions)
    y_true = torch.cat(y_true)

    y_predictions = y_predictions.cpu().numpy()
    y_true = y_true.numpy()

    y_predictions = y_predictions.astype(np.int64)
    y_true = y_true.astype(np.int64)

    y_predictions = y_predictions.reshape(-1)
    y_true = y_true.reshape(-1)

    accuracy_score(y_true, y_predictions)

    confusion_matrix_metrics = compute_confusion_matrix(y_true, y_predictions)

    compute_performance_metrics(confusion_matrix_metrics)
