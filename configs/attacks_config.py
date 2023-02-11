import os
import yaml
import sys
from pathlib import Path
import torch
import datetime
import time

from utils.general import init_seeds

# init seed cuda
init_seeds()


FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))



class BaseConfigVoice:
    def __init__(self):
        self.root_path = ROOT
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # print('device is {}'.format(device), flush=True)

        model_name = 'ECPA_TDNN'  # see configs/model_config.yaml for other options


        dataset_name = 'LIBRI'
        # self._set_dataset(dataset_name) #$$$

        self.loss_func_params = {
            'MSE': {},
            # 'L1': {},
            # 'LPIPS': {
            #     'net': 'squeeze',  # for LPIPS, one of: ['alex','vgg','squeeze']
            #     'device': self.device
            # },  # for LPIPS
        }
        self.loss_params = {
            'weights': [1]
        }

        self.attack_name = 'PGD'
        self.attack_params = {
            'norm': 2,
            'eps': 10,
            'eps_step': 0.2,
            'decay': None,
            'max_iter': 50,
            'targeted': True,
            'num_random_init': 1,
            'device': self.device
        }

        if 'active_output_indices' in self.dataset_config:
            mask = torch.zeros((1, 3, self.dataset_config['height'], self.dataset_config['width']))
            mask[:, :, -self.dataset_config['active_output_indices']['x2']-1:-self.dataset_config['active_output_indices']['x1']-1,  # using x1 and x2 this way because of updown flip
                self.dataset_config['active_output_indices']['y1']:self.dataset_config['active_output_indices']['y2']] = 1
            mask = mask.to(self.device)
            self.loss_params.update({'mask': mask})

        self.demo_mode = 'predict_one'  # 'predict_one' or 'predict_many'
        self.record_embedding_layer = True
        self.layers_to_record = ["down5_encode_0"]
        self.reduction_method = 'PCA'
        self.output_path_for_reconstructions = '/dt/shabtaia/dt-fujitsu/Lensless_imaging/Datasets/DiffuserCam/reconstructions/test_set'
        self.save_reconstructions_as_jpg = True
        self.save_reconstructions_as_np = False

        self.loader_params = {
            'batch_size': 8,
            'num_workers': 4
        }

        self._set_model(model_name)
        self._set_losses(self.loss_func_params)

        with open(ROOT / 'configs/attack_config.yaml', 'r') as stream:
            self.attack_config = yaml.safe_load(stream)[self.attack_name]
        with open(ROOT / 'configs/classifier_config.yaml', 'r') as stream:
            self.estimator_config = yaml.safe_load(stream)[self.estimator_name]
        self.estimator_config.update({'device': self.device, 'mode': 'eval'})

        self._update_current_dir()

    def __getitem__(self, item):
        return getattr(self, item)

    def _set_model(self, model_name):
        self.model_name = model_name
        with open(ROOT / 'configs/model_config.yaml', 'r') as stream:
            self.model_config = yaml.safe_load(stream)[self.model_name]

    def _set_losses(self, loss_func_params):
        self.loss_func_params = loss_func_params
        with open(ROOT / 'configs/losses_config.yaml', 'r') as stream:
            yaml_file = yaml.safe_load(stream)
            self.losses_config = []
            for loss_name in self.loss_func_params.keys():
                self.losses_config.append(yaml_file[loss_name])
            del yaml_file

    def _update_current_dir(self):
        my_date = datetime.datetime.now()
        month_name = my_date.strftime("%B")
        self.current_dir = os.path.join("experiments", month_name, time.strftime("%d-%m-%Y") + '_' + time.strftime("%H%M%S"))
        if 'SLURM_JOB_ID' in os.environ.keys():
            self.current_dir += '_' + os.environ['SLURM_JOB_ID']
        Path(self.current_dir).mkdir(parents=True, exist_ok=True)

    def _set_dataset(self, dataset_name):
        with open(ROOT / 'configs/dataset_config.yaml', 'r') as stream:
            self.dataset_config = yaml.safe_load(stream)[dataset_name]
        self.dataset_config['dataset_name'] = dataset_name


class BaseConfig:
    def __init__(self):
        self.root_path = ROOT
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model_name = 'Le_ADMM_U'  # see configs/model_config.yaml for other options
        self.estimator_name = 'ResNet18'

        dataset_name = 'DiffuserCam'
        self._set_dataset(dataset_name)

        self.loss_func_params = {
            'MSE': {},
            # 'L1': {},
            # 'LPIPS': {
            #     'net': 'squeeze',  # for LPIPS, one of: ['alex','vgg','squeeze']
            #     'device': self.device
            # },  # for LPIPS
        }
        self.loss_params = {
            'weights': [1]
        }

        self.attack_name = 'PGD'
        self.attack_params = {
            'norm': 2,
            'eps': 10,
            'eps_step': 0.2,
            'decay': None,
            'max_iter': 50,
            'targeted': True,
            'num_random_init': 1,
            'device': self.device
        }

        if 'active_output_indices' in self.dataset_config:
            mask = torch.zeros((1, 3, self.dataset_config['height'], self.dataset_config['width']))
            mask[:, :, -self.dataset_config['active_output_indices']['x2']-1:-self.dataset_config['active_output_indices']['x1']-1,  # using x1 and x2 this way because of updown flip
                self.dataset_config['active_output_indices']['y1']:self.dataset_config['active_output_indices']['y2']] = 1
            mask = mask.to(self.device)
            self.loss_params.update({'mask': mask})

        self.demo_mode = 'predict_one'  # 'predict_one' or 'predict_many'
        self.record_embedding_layer = True
        self.layers_to_record = ["down5_encode_0"]
        self.reduction_method = 'PCA'
        self.output_path_for_reconstructions = '/dt/shabtaia/dt-fujitsu/Lensless_imaging/Datasets/DiffuserCam/reconstructions/test_set'
        self.save_reconstructions_as_jpg = True
        self.save_reconstructions_as_np = False
        self.loader_params = {
            'batch_size': 8,
            'num_workers': 4
        }

        self._set_model(model_name)
        self._set_losses(self.loss_func_params)

        with open(ROOT / 'configs/attack_config.yaml', 'r') as stream:
            self.attack_config = yaml.safe_load(stream)[self.attack_name]

        with open(ROOT / 'configs/classifier_config.yaml', 'r') as stream:
            self.estimator_config = yaml.safe_load(stream)[self.estimator_name]
        self.estimator_config.update({'device': self.device, 'mode': 'eval'})

        self._update_current_dir()

    def __getitem__(self, item):
        return getattr(self, item)

    def _set_model(self, model_name):
        self.model_name = model_name
        with open(ROOT / 'configs/model_config.yaml', 'r') as stream:
            self.model_config = yaml.safe_load(stream)[self.model_name]

    def _set_losses(self, loss_func_params):
        self.loss_func_params = loss_func_params
        with open(ROOT / 'configs/losses_config.yaml', 'r') as stream:
            yaml_file = yaml.safe_load(stream)
            self.losses_config = []
            for loss_name in self.loss_func_params.keys():
                self.losses_config.append(yaml_file[loss_name])
            del yaml_file

    def _update_current_dir(self):
        my_date = datetime.datetime.now()
        month_name = my_date.strftime("%B")
        self.current_dir = os.path.join("experiments", month_name, time.strftime("%d-%m-%Y") + '_' + time.strftime("%H%M%S"))
        if 'SLURM_JOB_ID' in os.environ.keys():
            self.current_dir += '_' + os.environ['SLURM_JOB_ID']
        Path(self.current_dir).mkdir(parents=True, exist_ok=True)

    def _set_dataset(self, dataset_name):
        with open(ROOT / 'configs/dataset_config.yaml', 'r') as stream:
            self.dataset_config = yaml.safe_load(stream)[dataset_name]
        self.dataset_config['dataset_name'] = dataset_name


class OneToOneAttackConfig(BaseConfig):
    def __init__(self):
        super(OneToOneAttackConfig, self).__init__()

        dataset_path = '/dt/shabtaia/dt-fujitsu/Lensless_imaging/Datasets/DiffuserCam'
        attack_image_name = 'im157.npy'  # 157
        self.attack_img_diff_path = os.path.join(dataset_path, 'diffuser', attack_image_name)
        target_image_name = 'im326.npy'  # 326
        self.target_img_orig_path = os.path.join(dataset_path, 'diffuser', target_image_name)
        self.loss_params.update({'images_save_path': os.path.join(self.current_dir, 'outputs')})


class ManyToManyAttackConfig(BaseConfig):
    def __init__(self):
        super(ManyToManyAttackConfig, self).__init__()

        self.loader_params = {
            'batch_size': 8,
            'num_workers': 4
        }


class UniversalAttackConfig(BaseConfig):
    def __init__(self):
        super(UniversalAttackConfig, self).__init__()


config_dict = {
    'BaseVoice':BaseConfigVoice,
    'Base': BaseConfig,
    'OneToOne': OneToOneAttackConfig,
    'ManyToMany': ManyToManyAttackConfig,
    'Universal': UniversalAttackConfig
}
