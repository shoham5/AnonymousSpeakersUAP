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
        model_name = "hubert" # hubert "spkrec-ecapa-voxceleb" # "titanet" #"wavlm"# "spkrec-xvect-voxceleb" #"spkrec-ecapa-voxceleb"#

        # 'spkrec-ecapa-voxceleb'  # see configs/model_config.yaml for other options

        print("model_name: ",model_name)
        self._set_model(model_source, model_name)

        dataset_name = 'LIBRI-CLOSE-SET-TRAIN' # 'VOXCELEB1-CLOSE-SET-TRAIN' # 'VOXCELEB1-CLOSE-SET-TRAIN'#'LIBRI-CLOSE-SET-TRAIN' # 'LIBRIALL'
        print("dataset_name: ", dataset_name)
        self._set_dataset(dataset_name)

        # # TODO: split to two classes. one for eval, other for attack. delete similarity from BaseConfig
        # self.similarity_params = {'dim': 2}
        # similarity_func_name = 'CosineSimilarity'
        # self._set_similarity_one_func(similarity_func_name)

        # self.loss_func_params = {
        #     'CosineSimilarity': {'dim': 0},
        # }
        # self._set_losses(self.loss_func_params)
        #
        # self.loss_params = {
        #     'weights': [1]
        # }

        self.loss_func_params = {
            'CosineSimilarity': {'dim': 1},
        }
        self._set_losses(self.loss_func_params)

        self.loss_params = {
            'weights': [1]
        }

        self.attack_name = 'PGD'
        self.attack_params = {
            'norm': 2,
            'eps': 0.5,
            'eps_step': 0.3,
            'decay': None,
            'max_iter': 18,
            'targeted': True,
            'num_random_init': 1,
            'clip_values': (-1, 1),#(-2 ** 15, 2 ** 15 - 1),
            'device': self.device
        }

        self.loader_params = {
            'batch_size': 64,
            'num_workers': 4
        }

        self.num_wavs_for_emb = 5
        self.num_of_seconds = 3
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

    # TODO: delete after split classes to eval and test
    def _set_similarity_one_func(self, similarity_func_name):
        # self.similarity_func_params = similarity_func_params
        with open(ROOT / 'configs/similarities_config.yaml', 'r') as stream:
            yaml_file = yaml.safe_load(stream)
            self.similarity_config_one = yaml_file[similarity_func_name]

    def _update_current_dir(self):
        my_date = datetime.datetime.now()
        month_name = my_date.strftime("%B")
        self.current_dir = os.path.join("experiments", month_name, time.strftime("%d-%m-%Y") + '_' + time.strftime("%H%M%S"))
        if 'SLURM_JOB_ID' in os.environ.keys():
            self.current_dir += '_' + os.environ['SLURM_JOB_ID']
        Path(self.current_dir).mkdir(parents=True, exist_ok=True)


class BaseEvalConfig:
    def __init__(self):
        self.root_path = ROOT
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model_source = 'SpeechBrain'  # see configs/model_config.yaml for other options
        model_name = 'spkrec-ecapa-voxceleb'  # see configs/model_config.yaml for other options
        self._set_model(model_source, model_name)

        dataset_name = 'LIBRIALL'#'LIBRI'
        self._set_dataset(dataset_name)

        self.similarity_params = {'dim': 2}
        similarity_func_name = 'CosineSimilarity'
        self._set_similarity_one_func(similarity_func_name)

        self.similarity_func_params = {
            'CosineSimilarity': {'dim': 2},
        }
        # self.similarity_func_params = 'CosineSimilarity'
        self._set_similarity(self.similarity_func_params)

        self.loader_params = {
            'batch_size': 4,
            'num_workers': 4
        }

        self.fs = 16000

        # output directory
        # self._update_current_dir()

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

    # TODO # doesn't need similarity_config using dict
    def _set_similarity(self, similarity_func_params):
        self.similarity_func_params = similarity_func_params
        with open(ROOT / 'configs/similarities_config.yaml', 'r') as stream:
            # self.sims_config = yaml.safe_load(stream)[similarity_func_params]
            yaml_file = yaml.safe_load(stream)
            self.similarity_config = []
            for sims_name in self.similarity_func_params.keys():
                self.similarity_config.append(yaml_file[sims_name])
            del yaml_file

    def _set_similarity_one_func(self, similarity_func_name):
        # self.similarity_func_params = similarity_func_params
        with open(ROOT / 'configs/similarities_config.yaml', 'r') as stream:
            yaml_file = yaml.safe_load(stream)
            self.similarity_config_one = yaml_file[similarity_func_name]

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


class SingleWAVEvalConfig(BaseEvalConfig):
    def __init__(self):
        super(SingleWAVEvalConfig, self).__init__()
        self.wav_path = 'data/LIBRI/d3/19/19-198-0009.wav'
        self.eval_wav_path = 'data/LIBRI/d3/19/19-198-0033.wav'
        self.speaker_id = '19'
        self.num_wavs_for_emb = 3


class SingleWAVConfig(BaseConfig):
    def __init__(self):
        super(SingleWAVConfig, self).__init__()
        self.speaker_id = '19'
        self.wav_path = 'data/LIBRI/d3/19/19-198-0009.wav'
        self.eval_wav_path = 'data/LIBRI/d3/19/19-198-0033.wav'
        self.speaker_id = '19'
        self.num_wavs_for_emb = 3


class SingleSpeakerConfig(BaseConfig):
    def __init__(self):
        super(SingleSpeakerConfig, self).__init__()
        self.data_path = 'data/LIBRI/d3/19'
        self.speaker_id = '19'
        self.wav_path = 'data/LIBRI/d3/19t/19-198-0036.wav'
        self.wav_path2 = 'data/LIBRI/d3/19t/19-198-0021.wav'
        self.num_wavs_for_emb = 3
        self.loss_func_params = {
            'CosineSimilarity': {'dim': 1},
        }
        self._set_losses(self.loss_func_params)

class MultiSpeakersConfig(BaseConfig):
    def __init__(self):
        super(MultiSpeakersConfig, self).__init__()
        self.data_path = 'data/LIBRI/d3'
        self.speakers_id = create_speakers_list('data/LIBRI/d3')
        self.wav_path = 'data/LIBRI/d3t/19t/19-198-0036.wav'
        self.wav_path2 = 'data/LIBRI/d3t/2843t/2843-152918-0026.wav'
        self.wav_path3 = 'data/LIBRI/d3t/8747t/8747-293952-0094.wav'
        self.num_wavs_for_emb = 3
        self.loss_func_params = {
            'CosineSimilarity': {'dim': 1},
        }
        self._set_losses(self.loss_func_params)


class UniversalAttackConfig(BaseConfig):
    def __init__(self):
        super(UniversalAttackConfig, self).__init__()
        self.init_pert_type = "random" ##  'zeros' #, "random" "prev"
        self.start_learning_rate = 5e-3 # 10 #5e-3 #5e-4#5e-4 # 5e-1 #5e-2 #5e-3
        self.es_patience = 7
        self.sc_patience = 2
        self.sc_min_lr = 1e-8 # 1e-8
        self.epochs = 100 # 50 #36 # 20 for eps 0.1
        self.scheduler_factory = lambda optimizer: torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                                              patience=self.sc_patience,
                                                                                              min_lr=self.sc_min_lr,
                                                                                              mode='min')
        number_of_speakers = 100
        speaker_labels = os.listdir(self.dataset_config['root_path'])[:number_of_speakers]
        speaker_labels_mapper = {i: lab for i, lab in enumerate(speaker_labels)}
        num_wavs_for_emb = 5
        self.dataset_config.update({
            'number_of_speakers': number_of_speakers,
            'speaker_labels': speaker_labels,
            'speaker_labels_mapper': speaker_labels_mapper,
            'num_wavs_for_emb': num_wavs_for_emb,
        })

        self.data_path = 'data/libri_train-clean-100/'
        self.test_path = 'data/libri_test/'
        self.is_test = False
        # self.speakers_id = create_speakers_list('data/LIBRI/d3')
        # self.speakers_number = len(self.speakers_id)
        self.wav_path = 'data/LIBRI/d3t/19t/19-198-0036.wav'
        self.wav_path2 = 'data/LIBRI/d3t/2843t/2843-152918-0026.wav'
        self.wav_path3 = 'data/LIBRI/d3t/8747t/8747-293952-0094.wav'



config_dict = {
    'Base': BaseConfig,
    'SingleWAV': SingleWAVConfig,
    'SingleSpeaker': SingleSpeakerConfig,
    "MultiSpeakers": MultiSpeakersConfig,
    'UniversalAttack': UniversalAttackConfig,
    'Universal': UniversalAttackConfig,
    'UniversalEval':SingleWAVEvalConfig,
    'SingleWavEval': SingleWAVEvalConfig
}
