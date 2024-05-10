import os

# Training Hyperparameters
NUM_CLASSES         = 200
BATCH_SIZE          = 256
VAL_EVERY_N_EPOCH   = 1

NUM_EPOCHS          = 40
 
#OPTIMIZER_PARAMS    = {'type': 'SGD', 'lr': 0.002, 'momentum': 0.95}
OPTIMIZER_PARAMS    = {'type': 'Adam', 'lr': 0.001,'weight_decay': 0.0001}

#SCHEDULER_PARAMS    = {'type': 'MultiStepLR', 'milestones': [30, 38], 'gamma': 0.5}
#SCHEDULER_PARAMS    = {'type': 'ExponentialLR', 'gamma': 0.95}
#SCHEDULER_PARAMS    = {'type': 'CosineAnnealingLR', 'T_max': 12 , 'eta_min': 0.001}
#SCHEDULER_PARAMS    = {'type': 'CosineAnnealingWarmRestarts', 'T_0': 24,'T_mult':1 }
SCHEDULER_PARAMS    = {'type': 'StepLR', 'step_size': 12, 'gamma': 0.1 }

# Dataaset
DATASET_ROOT_PATH   = 'datasets/'
NUM_WORKERS         = 8

# Augmentation
IMAGE_ROTATION      = 20
IMAGE_FLIP_PROB     = 0.5
IMAGE_NUM_CROPS     = 64
IMAGE_PAD_CROPS     = 4
IMAGE_MEAN          = [0.4802, 0.4481, 0.3975]
IMAGE_STD           = [0.2302, 0.2265, 0.2262]

# Network
#MODEL_NAME          = 'resnet18'
MODEL_NAME          = 'resnext50_32x4d'
#MODEL_NAME          = 'vgg16'

# Compute related
ACCELERATOR         = 'gpu'
DEVICES             = [0]
PRECISION_STR       = '32-true'

# Logging
WANDB_PROJECT       = 'aue8088-pa1'
WANDB_ENTITY        = os.environ.get('WANDB_ENTITY')
WANDB_SAVE_DIR      = 'wandb/'
WANDB_IMG_LOG_FREQ  = 50
WANDB_NAME          = f'{MODEL_NAME}-B{BATCH_SIZE}-{OPTIMIZER_PARAMS["type"]}'
WANDB_NAME         += f'-{SCHEDULER_PARAMS["type"]}{OPTIMIZER_PARAMS["lr"]:.1E}'
