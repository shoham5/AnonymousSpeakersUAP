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


class BaseConfig:
    def __init__(self):
        self.root_path = ROOT
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model_source = 'SpeechBrain'  # see configs/model_config.yaml for other options
        model_name = 'spkrec-ecapa-voxceleb'  # see configs/model_config.yaml for other options
        self._set_model(model_source, model_name)

        dataset_name = 'LIBRI'
        self._set_dataset(dataset_name)

        self.loss_func_params = {
            'CosineSimilarity': {'dim': 0},
        }
        self._set_losses(self.loss_func_params)

        self.loss_params = {
            'weights': [1]
        }

        self.attack_name = 'PGD'
        self.attack_params = {
            'norm': 2,
            'eps': 0.1,
            'eps_step': 0.01,
            'decay': None,
            'max_iter': 100,
            'targeted': True,
            'num_random_init': 1,
            'clip_values': (-2 ** 15, 2 ** 15 - 1),
            'device': self.device
        }

        self.loader_params = {
            'batch_size': 8,
            'num_workers': 4
        }

        self.fs = 16000
        with open(ROOT / 'configs/attack_config.yaml', 'r') as stream:
            self.attack_config = yaml.safe_load(stream)[self.attack_name]

        self._update_current_dir()

    def __getitem__(self, item):
        return getattr(self, item)

    def _set_model(self, model_source, model_name):
        self.model_source = model_source
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

    def _set_dataset(self, dataset_name):
        self.dataset_name = dataset_name
        with open(ROOT / 'configs/dataset_config.yaml', 'r') as stream:
            self.dataset_config = yaml.safe_load(stream)[dataset_name]
        # self.dataset_config['dataset_name'] = dataset_name

    def _update_current_dir(self):
        my_date = datetime.datetime.now()
        month_name = my_date.strftime("%B")
        self.current_dir = os.path.join("experiments", month_name, time.strftime("%d-%m-%Y") + '_' + time.strftime("%H%M%S"))
        if 'SLURM_JOB_ID' in os.environ.keys():
            self.current_dir += '_' + os.environ['SLURM_JOB_ID']
        Path(self.current_dir).mkdir(parents=True, exist_ok=True)


class SingleWAVConfig(BaseConfig):
    def __init__(self):
        super(SingleWAVConfig, self).__init__()
        self.speaker_id = '19'
        self.num_wavs_for_emb = 3
        self.wav_path = 'data/LIBRI/d3/19/19-198-0009.wav'


class SingleSpeakerConfig(BaseConfig):
    def __init__(self):
        super(SingleSpeakerConfig, self).__init__()
        self.speaker_id = '19'
        self.num_wavs_for_emb = 3


class UniversalAttackConfig(BaseConfig):
    def __init__(self):
        super(UniversalAttackConfig, self).__init__()


config_dict = {
    'Base': BaseConfig,
    'SingleWAV': SingleWAVConfig,
    'SingleSpeaker': SingleSpeakerConfig,
    'Universal': UniversalAttackConfig
}
