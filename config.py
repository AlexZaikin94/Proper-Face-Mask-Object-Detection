import os
from src import utils
from src import data_utils



cfg = utils.Config()

###########################################################################
########################### general hyperparams ###########################
###########################################################################
cfg.seed = None
# cfg.data_path = os.path.join('/mnt', 'd', 'data')
cfg.data_path = 'data'
cfg.models_dir = 'models'
# cfg.final_model_path = os.path.join(cfg.models_dir, 'model.pth')
cfg.final_model_path = 'model.pth'
cfg.save_log = True                    # additionally save log and training loss logs to a .csv file
cfg.epochs_evaluate_train = 1          # evaluate train (in eval mode with no_grad) every epochs_evaluate_train epochs
cfg.epochs_evaluate_validation = 1     # evaluate validation (in eval mode with no_grad) every epochs_evaluate_validation epochs
cfg.num_workers = 2                    # num_workers for data loader
cfg.epochs_save = 5                    # save a checkpoint (additionally to last and best) every epochs_save epochs

cfg.save = False                       # save model checkpoints (best, last and epoch) True
cfg.tqdm_bar = False                   # using a tqdm bar for loading data and epoch progression, should be False if not using a jupyter notebook
# cfg.tqdm_bar = True                   # using a tqdm bar for loading data and epoch progression, should be False if not using a jupyter notebook
cfg.prints = 'print'                   # should be 'display' if using a jupyter notebook, else 'print'
# cfg.prints = 'display'                   # should be 'display' if using a jupyter notebook, else 'print'
cfg.load = -1                          # epoch to load `-1` to load last, `best` to load best epoch, or `NUM` to load NUM epoch
cfg.max_iterations = None

cfg.train_test = False
cfg.balanced_classes = True
cfg.dataset_a = 'andrewmvd'
cfg.dataset_w = 'wobotintelligence'

###########################################################################
############################ model hyperparams ############################
###########################################################################
# cfg.model = 'fasterrcnn_mobilenet_v3_large_320_fpn'
cfg.model = 'ssdlite320_mobilenet_v3_large'

cfg.wd = 0.0  # 1e-5
cfg.bs = 32  # 32 96 64
cfg.epochs = 300  # 600 800 1000
cfg.num_batches = 100

cfg.num_classes = 4

cfg.optimizer = 'adam'  # adam sgd
cfg.optimizer_params = {}
cfg.optimizer_momentum = 0.9
cfg.lr = 1e-3  # 3e-4 1e-3
cfg.min_lr = 5e-8
cfg.cos = False
cfg.best_policy = 'train_loss'
cfg.bias = True
cfg.version = f'{cfg.model}_{cfg.optimizer}_bs{cfg.bs}_wd{cfg.wd}{"_cos" if cfg.cos else ""}'

cfg.train_transforms = data_utils.train_transforms



