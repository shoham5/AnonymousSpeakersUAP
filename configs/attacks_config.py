import os
import yaml
import sys
from pathlib import Path
import torch
import datetime
import time
from utils.data_utils import create_speakers_list
from utils.general import init_seeds
# import nemo.collections.asr as nemo_asr


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
        model_name ="spkrec-ecapa-voxceleb"

        print("model_name: ", model_name)
        self._set_model(model_source, model_name) # update self.model_config

        dataset_name = 'VOX-CLOSE-SET-TRAIN-50'
        print("dataset_name: ", dataset_name)
        self._set_dataset(dataset_name)

        self.loss_func_params = {
            'CosineSimilarity': {'dim': 1},
        }
        self._set_losses(self.loss_func_params)

        self.loss_params = {
            'weights': [1]
        }



        self.loader_params = {
            'batch_size': 64,
            'num_workers': 4
        }

        self.num_wavs_for_emb = 5
        self.num_of_seconds = 3
        self.fs = 16000

        self._update_current_dir()

    def __getitem__(self, item):
        return getattr(self, item)

    def _set_models_list(self, models_name):
        self.model_name_list = models_name
        self.model_config_dict = {}
        for model_name in models_name:
            with open(ROOT / 'configs/model_config.yaml', 'r') as stream:
                self.model_config_dict[model_name] = yaml.safe_load(stream)[model_name]

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

    def _update_current_dir(self):
        my_date = datetime.datetime.now()
        month_name = my_date.strftime("%B")
        self.current_dir = os.path.join("experiments", month_name, time.strftime("%d-%m-%Y") + '_' + time.strftime("%H%M%S"))
        if 'SLURM_JOB_ID' in os.environ.keys():
            self.current_dir += '_' + os.environ['SLURM_JOB_ID']
        Path(self.current_dir).mkdir(parents=True, exist_ok=True)


class UniversalAttackConfig(BaseConfig):
    def __init__(self):
        super(UniversalAttackConfig, self).__init__()
        self.init_pert_type = "random"
        self.start_learning_rate = 5e-3
        self.es_patience = 7
        self.sc_patience = 2
        self.sc_min_lr = 1e-8
        self.epochs = 100
        self.scheduler_factory = lambda optimizer: torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                                              patience=self.sc_patience,
                                                                                              min_lr=self.sc_min_lr,
                                                                                              mode='min')
        number_of_speakers = 50
        print(f"epochs: {self.epochs} , number_of_speakers: {number_of_speakers}")
        speaker_labels = os.listdir(self.dataset_config['root_path'])
        speaker_labels.sort()
        speaker_labels = speaker_labels[:number_of_speakers]

        speaker_labels_mapper = {i: lab for i, lab in enumerate(speaker_labels)}
        num_wavs_for_emb = 5
        self.dataset_config.update({
            'number_of_speakers': number_of_speakers,
            'speaker_labels': speaker_labels,
            'speaker_labels_mapper': speaker_labels_mapper,
            'num_wavs_for_emb': num_wavs_for_emb,
        })



config_dict = {
    'Universal': UniversalAttackConfig
    }
