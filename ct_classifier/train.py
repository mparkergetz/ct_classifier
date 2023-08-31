'''
    Training script. Here, we load the training and validation datasets (and
    data loaders) and the model and train and validate the model accordingly.

    2022 Benjamin Kellenberger
'''

import os
import argparse
import yaml
import glob
from tqdm import trange

import wandb
from sklearn.metrics import confusion_matrix, average_precision_score, precision_recall_curve, ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.optim import SGD

from util import init_seed
from dataset import CTDataset 
from model import CustomResNet18
import matplotlib.pyplot as plt
import numpy as np

def create_dataloader(cfg, split='train'):
    '''
        Loads a dataset according to the provided split and wraps it in a
        PyTorch DataLoader object.
    '''
    dataset_instance = CTDataset(cfg, split)       
    #print ("Print the length of the dataset")
    #print(len(dataset_instance))
    
    dataLoader = DataLoader(
            dataset=dataset_instance,
            batch_size=cfg['batch_size'],
            shuffle=True,
            num_workers=cfg['num_workers']
        )
    return dataLoader


def load_model(cfg,load_latest_version=False):
    '''
        Creates a model instance and loads the latest model state weights.

        Default is to start from 0 for the epochs. If you want to load an existing model
        you can call load_model(cfg, load_latest_version = True), which will load an already trained-ish model
    '''
    model_instance = CustomResNet18(cfg['num_classes'])         

    # # load latest model state - how it used to be traded
    # model_states = glob.glob('model_states/*.pt')

    # load latest model state
    experiment_folder = os.path.join('all_model_states', cfg["experiment_name"])
    model_states = glob.glob(os.path.join(experiment_folder, '*.pt'))

    #model_states = [] # Sets it to 0, and not see any checkpoint files: Hey this has been changed during training since we are not ready to resume
    if len(model_states) > 0 and load_latest_version==True:
        # # at least one save state found; get latest
        # model_epochs = [int(m.replace('modetl_states/','').replace('.pt','')) for m in model_states]
        # start_epoch = max(model_epochs) # The highest or latest epoch that the model hsa run to
        # # load state dict and apply weights to model
        # print(f'Resuming from epoch {start_epoch}')
        # state = torch.load(open(f'model_states/{start_epoch}.pt', 'rb'), map_location='cpu')
        # model_instance.load_state_dict(state['model'])
           # at least one save state found; get latest
        # these lines have now been replaced with the below code
        model_epochs = [int(os.path.basename(m).replace('.pt', '')) for m in model_states]
        start_epoch = max(model_epochs)  # The highest or latest epoch that the model has run to
        # load state dict and apply weights to model
        print(f'Resuming from epoch {start_epoch}')
        state = torch.load(open(os.path.join(experiment_folder, f'{start_epoch}.pt'), 'rb'), map_location='cpu')
        model_instance.load_state_dict(state['model'])


    else:
        # no save state found; start anew
        print('Starting new model')
        start_epoch = 0

    return model_instance, start_epoch

def save_model(cfg, epoch, model, stats):
    # make sure save directory exists; create if not
    main_folder = 'all_model_states'

    # Create a subfolder for the current experiment
    experiment_folder = os.path.join(main_folder, cfg["experiment_name"])
    os.makedirs(experiment_folder, exist_ok=True)
    # os.makedirs('model_states', exist_ok=True)

    # get model parameters and add to stats...
    stats['model'] = model.state_dict()

    # ...and save
    # torch.save(stats, open(f'model_states/{epoch}.pt', 'wb'))
    stats_file = os.path.join(experiment_folder, f'{epoch}.pt')
    torch.save(stats, open(stats_file, 'wb'))
    
    # also save config file if not present
    # Construct the full path using os.path.join
    # cfpath = 'model_states/config.yaml'
    # cfpath = os.path.join(cfpath, cfg["experiment_name"])
    # cfpath = f'{"config"}_{cfg["experiment_name"]}.yaml'
    # if not os.path.exists(cfpath):
    #     with open(cfpath, 'w') as f:
    #         yaml.dump(cfg, f)

     # Save config file if not present
    config_file = os.path.join(experiment_folder, f'config_{cfg["experiment_name"]}.yaml')
    if not os.path.exists(config_file):
        with open(config_file, 'w') as f:
            yaml.dump(cfg, f)

def setup_optimizer(cfg, model):
    '''
        The optimizer is what applies the gradients to the parameters and makes
        the model learn on the dataset.
    '''
    optimizer = SGD(model.parameters(),
                    lr=cfg['learning_rate'],
                    weight_decay=cfg['weight_decay'])

   # Define the learning rate scheduler based on the provided step_size and gamma
    scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg['scheduler_step_size'], gamma=cfg['scheduler_gamma'])

    return optimizer, scheduler



def train(cfg, dataLoader, model, optimizer):
    '''
        Our actual training function.
    '''

    device = cfg['device']

    # put model on device
    model.to(device)
    
    # put the model into training mode
    # this is required for some layers that behave differently during training
    # and validation (examples: Batch Normalization, Dropout, etc.)
    model.train()

    # needs to have a softmax function : # Apply softmax to convert logits to probabilities
    # softmax = nn.Softmax(dim=1)
    # probabilities = softmax(logits)
    # print(probabilities)

    # loss function
    criterion = nn.CrossEntropyLoss() # the softmax is internal to this function: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

    # running averages
    loss_total, oa_total = 0.0, 0.0                         # for now, we just log the loss and overall accuracy (OA)

    # iterate over dataLoader
    progressBar = trange(len(dataLoader))

    # initialise predictions and labels lists
    labels_list = []
    pred_list = []

    for idx, (data, labels) in enumerate(dataLoader):       # see the last line of file "dataset.py" where we return the image tensor (data) and label

        # put data and labels on device
        data, labels = data.to(device), labels.to(device)

        # forward pass
        prediction = model(data)

        # reset gradients to zero
        optimizer.zero_grad()

        # loss
        loss = criterion(prediction, labels)

        # backward pass (calculate gradients of current batch)
        loss.backward()

        # apply gradients to model parameters
        optimizer.step()

        # log statistics
        loss_total += loss.item()                       # the .item() command retrieves the value of a single-valued tensor, regardless of its data type and device of tensor

        pred_label = torch.argmax(prediction, dim=1)    # the predicted label is the one at position (class index) with highest predicted value
        oa = torch.mean((pred_label == labels).float()) # OA: number of correct predictions divided by batch size (i.e., average/mean)
        oa_total += oa.item()

        # auprc preparation
        pred_list.extend(list(pred_label.detach().cpu().numpy()))
        labels_list.extend(list(labels.detach().cpu().numpy()))

        progressBar.set_description(
            '[Train] Loss: {:.2f}; OA: {:.2f}%'.format(
                loss_total/(idx+1),
                100*oa_total/(idx+1)
            )
        )
        progressBar.update(1)

    # Use label_binarize to be multi-label like settings
    one_hot_labels = label_binarize(labels_list, classes=list(range(cfg['num_classes']))) # this will index from 0-28 for 29 classes
    n_classes = one_hot_labels.shape[1]
    #print('nclasses labels', n_classes, one_hot_labels)
    one_hot_preds = label_binarize(pred_list, classes=list(range(cfg['num_classes']))) # this will index from 0-28 for 29 classes
    n_classes = one_hot_preds.shape[1]
    #print('nclasses preds', n_classes, one_hot_preds)
    # auprc = average_precision_score(labels.detach().cpu().numpy() , prediction.detach().cpu().numpy(),average=None)
    auprc = average_precision_score(one_hot_labels, one_hot_preds,average=None)
    mAP_train = np.mean(auprc)


    # end of epoch; finalize
    progressBar.close()
    loss_total /= len(dataLoader)           # shorthand notation for: loss_total = loss_total / len(dataLoader)
    oa_total /= len(dataLoader)

    return loss_total, oa_total, mAP_train


def validate(cfg, dataLoader, model):
    '''
        Validation function. Note that this looks almost the same as the training
        function, except that we don't use any optimizer or gradient steps.
    '''
    
    device = cfg['device']
    model.to(device)

    # put the model into evaluation mode
    # see lines 103-106 above
    model.eval()
    
    criterion = nn.CrossEntropyLoss()   # we still need a criterion to calculate the validation loss

    # running averages
    loss_total, oa_total = 0.0, 0.0     # for now, we just log the loss and overall accuracy (OA)

    # iterate over dataLoader
    progressBar = trange(len(dataLoader))
    
    # initialise predictions and labels lists
    labels_list = []
    pred_list = []

    with torch.no_grad():               # don't calculate intermediate gradient steps: we don't need them, so this saves memory and is faster
        for idx, (data, labels) in enumerate(dataLoader):

            # put data and labels on device
            data, labels = data.to(device), labels.to(device)

            # forward pass
            prediction = model(data)

            # loss
            loss = criterion(prediction, labels)

            # log statistics
            loss_total += loss.item()

            pred_label = torch.argmax(prediction, dim=1)
            oa = torch.mean((pred_label == labels).float())
            oa_total += oa.item()
            
            # auprc preparation
            pred_list.extend(list(pred_label.detach().cpu().numpy()))
            labels_list.extend(list(labels.detach().cpu().numpy()))

            progressBar.set_description(
                '[Val ] Loss: {:.2f}; OA: {:.2f}%'.format(
                    loss_total/(idx+1),
                    100*oa_total/(idx+1)
                )
            )
            progressBar.update(1)

        # Use label_binarize to be multi-label like settings
        one_hot_labels = label_binarize(labels_list, classes=list(range(cfg['num_classes']))) # this will index from 0-28 for 29 classes
        n_classes = one_hot_labels.shape[1]
        #print('val_nclasses labels', n_classes)

        one_hot_preds = label_binarize(pred_list, classes=list(range(cfg['num_classes']))) # this will index from 0-28 for 29 classes
        n_classes = one_hot_preds.shape[1]
        #print(one_hot_preds)
        #print('val_nclasses preds', n_classes)
        # auprc = average_precision_score(labels.detach().cpu().numpy() , prediction.detach().cpu().numpy(),average=None)
        auprc = average_precision_score(one_hot_labels, one_hot_preds,average=None)
        mAP_val = np.mean(auprc)

    # end of epoch; finalize
    progressBar.close()
    loss_total /= len(dataLoader)
    oa_total /= len(dataLoader)

    return loss_total, oa_total, mAP_val



def main():


    # architecture name: 
    # dataset type:_high
    # batch size:
    # number of epochs: 
    # resume = True # to update an existing experiment... or not

    # Argument parser for command-line arguments:
    # python ct_classifier/train.py --config configs/exp_resnet18.yaml
    parser = argparse.ArgumentParser(description='Train deep learning model.')
    parser.add_argument('--config', help='Path to config file', default='configs/exp_resnet18.yaml')
    args = parser.parse_args()

    # load config
    print(f'Using config "{args.config}"')
    cfg = yaml.safe_load(open(args.config, 'r'))

    run = wandb.init(
        project=cfg['project_name'],
        config=cfg)
    
    artifact = wandb.Artifact(
    name = "pytorch-data",
    type = "dataset")

    artifact.add_dir("media/")
    wandb.log_artifact(artifact)

    
    resume = False  # Set this to True if you want to resume an existing experiment
    # if resume:
    #     with open("experiment_key.txt", "r") as file:
    #             experiment_key = file.read().strip()
    #     # experiment = comet_ml.ExistingExperiment(
    #     #     api_key=cfg["api_key"],
    #     #     project_name=cfg["project_name"],
    #     #     previous_experiment=experiment_key,
    #     # )

    # else:
    #     experiment = comet_ml.Experiment(
    #         api_key=cfg["api_key"],
    #         project_name=cfg["project_name"],
    #     )
    #     experiment.set_name(cfg["experiment_name"])

    #     # Get the experiment key
    #     experiment_key = experiment.get_key()


    # init random number generator seed (set at the start)
    init_seed(cfg.get('seed', None))

    # check if GPU is available
    device = cfg['device']
    if device != 'cpu' and not torch.cuda.is_available():
        print(f'WARNING: device set to "{device}" but CUDA not available; falling back to CPU...')
        cfg['device'] = 'cpu'

    # initialize data loaders for training and validation set
    dl_train = create_dataloader(cfg, split='train') # was just train before
    sample_batch = next(iter(dl_train))
    inputs, labels = sample_batch
    #print (labels)


    # Display the images
    fig = plt.figure(figsize=(12, 8))
    for idx in range(24):
        ax = fig.add_subplot(4, 6, idx + 1, xticks=[], yticks=[])
        # The imshow function is used to display the images, and the loop displays a sample of 12 images along with their corresponding labels
        ax.imshow(inputs[idx].permute(1, 2, 0))
        ax.set_title(f"Label: {labels[idx]}")

        # print("Print the image paths for the images that are being plotted")
        # print (image_path(idx))
        # image_name, _ = dataset.data[idx]  # Assuming dataset is the instance of your CustomDataset
        # image_path = os.path.join(dl_train.data_root, 'high', image_name)
        # print(f"Image Path: {image_path}")
        # ax.set_title(f"Label: {labels[idx]}\nPath: {image_paths[idx]}") 

   
    plt.tight_layout()
    plt.savefig("media/sanity_check_train.png")

    dl_val = create_dataloader(cfg, split='val')
    sample_batch = next(iter(dl_val))
    inputs, labels = sample_batch
    #print (labels)


    # Display the images
    fig = plt.figure(figsize=(12, 8))
    for idx in range(24):
        ax = fig.add_subplot(4, 6, idx + 1, xticks=[], yticks=[])
        # The imshow function is used to display the images, and the loop displays a sample of 12 images along with their corresponding labels
        ax.imshow(inputs[idx].permute(1, 2, 0))
        ax.set_title(f"Label: {labels[idx]}")

        # print("Print the image paths for the images that are being plotted")
        # print (image_path(idx))
        # image_name, _ = dataset.data[idx]  # Assuming dataset is the instance of your CustomDataset
        # image_path = os.path.join(dl_train.data_root, 'high', image_name)
        # print(f"Image Path: {image_path}")
        # ax.set_title(f"Label: {labels[idx]}\nPath: {image_paths[idx]}") 

   
    plt.tight_layout()
    plt.savefig("media/sanity_check_val.png")

    #print ("Length of training dataloader")

    # Number of training samples divided by batch size
    #print(len(dl_train))
    # print ("Length of validation dataloader")
    #print(len(dl_val))

    # initialize model
    model, current_epoch = load_model(cfg) # load_latest_version=True

    # set up model optimizer
    optim, scheduler = setup_optimizer(cfg, model)

    # we have everything now: data loaders, model, optimizer; let's do the epochs!
    numEpochs = cfg['num_epochs']
    while current_epoch < numEpochs:
        current_epoch += 1
        print(f'Epoch {current_epoch}/{numEpochs}')

        loss_train, oa_train, mAP_train = train(cfg, dl_train, model, optim)
        loss_val, oa_val, mAP_val = validate(cfg, dl_val, model)


        # combine stats and save
        stats = {
            'loss_train': loss_train,
            'loss_val': loss_val,
            'oa_train': oa_train,
            'oa_val': oa_val,
            'mAP_train': mAP_train,
            'mAP_val': mAP_val
        }

        # # Log loss metrics in the same plot
        wandb.log({'Training loss': loss_train, 'Validation loss': loss_val})

        # # Group accuracy metrics in a plot
        wandb.log({'Training OA accuracy': oa_train, 'Validation OA accuracy': oa_val})

        # # Group mAP metrics in a plot
        wandb.log({'Training mAP': mAP_train, 'Validation mAP': mAP_val})

        #wandb.log(stats)


        #  # Log hyperparameters like the learning rate and the batch size
        # for param_name, param_value in cfg.items():
        #     experiment.log_parameter(param_name, param_value)

        # # Log the last_lr value as a hyperparameter, which you might only need if you are scheduling
        # experiment.log_parameter("last_lr", last_lr)

        save_model(cfg, current_epoch, model, stats)
        
        # # Scheduler step to save, which is at the end of the training loop basically
        scheduler.step()
        last_lr = scheduler.get_last_lr()

        #  # Log learning rate, if you want to see where the steps are for example during training
        wandb.log({"learning_rate": last_lr[0]})


        # # Print the experiment key to load in the evaluation file
        # print("Experiment Key:", experiment_key)

        # # Save the experiment key to a file
        # with open("experiment_key.txt", "w") as file:
        #     file.write(experiment_key)

        # # experiment.end()


if __name__ == '__main__':
    # This block only gets executed if you call the "train.py" script directly
    # (i.e., "python ct_classifier/train.py").
    main()

# CTDataset.__getitem__()
# CTDataset.len ()
# dataLoader.len()