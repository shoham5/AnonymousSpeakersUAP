from configs.attacks_config import BaseConfig
from utils.general import init_seeds
import yaml
import sys
from pathlib import Path
import torch
import datetime
import time
import os
from utils.data_utils import load_from_npy

init_seeds()


FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))


class PlotEmbeddings:
    pass


class BaseConfiguration:
    def __init__(self):
        self.seed = 42
        self.patch_name = 'base'
        # Train dataset options
        self.is_real_person = False
        self.train_dataset_name = 'LIBRIALL'
      #################################################
        # TODO: CHECK DATASET
        self.test_data_set_name = 'LIBRI-CLOSE-SET-TEST' # "VOXCELEB1-CLOSE-SET-TEST" # "VOXCELEB1-CLOSE-SET-TEST" #'LIBRI-TEST' # 'LIBRI-CLOSE-SET-TEST'  #'LIBRI-TEST' # 'LIBRI-CLOSE-SET-TEST'# 'LIBRI-TEST'#  'LIBRI-CLOSE-SET-TEST'# "LIBRI-TEST"
        self._set_dataset(self.test_data_set_name)
#######################################################
        # self.train_img_dir = os.path.join('..', 'datasets', self.train_dataset_name)
        # self.train_number_of_people = 100
        # self.celeb_lab = os.listdir(self.train_img_dir)[:self.train_number_of_people]
        # self.celeb_lab_mapper = {i: lab for i, lab in enumerate(self.celeb_lab)}
        # self.num_of_train_images = 5
        #
        # self.shuffle = True
        # self.img_size = (112, 112)
        # self.train_batch_size = 4

        self.test_batch_size = 32
        self.magnification_ratio = 35

        # # Attack options
        # self.mask_aug = True
        # self.patch_size = (112, 112)  # height, width
        # self.initial_patch = 'white'  # body, white, random, stripes, l_stripes
        # self.epochs = 100
        # self.start_learning_rate = 1e-2
        # self.es_patience = 7
        # self.sc_patience = 2
        # self.sc_min_lr = 1e-6
        # self.scheduler_factory = lambda optimizer: optim.lr_scheduler.ReduceLROnPlateau(optimizer,
        #                                                                                 patience=self.sc_patience,
        #                                                                                 min_lr=self.sc_min_lr,
        #                                                                                 mode='min')
        #
        # # Landmark detection options
        # self.landmark_detector_type = 'mobilefacenet'  # face_alignment, mobilefacenet


        # Embedder options
        self.train_embedder_names = ['spkrec-ecapa-voxceleb' ]
        self.test_embedder_source = ['SpeechBrain']
        self.test_embedder_names = ['wavlm' ,'spkrec-ecapa-voxceleb', 'spkrec-xvect-voxceleb'] # ['spkrec-xvect-voxceleb'] #['spkrec-ecapa-voxceleb']#, 'spkrec-xvect-voxceleb']
        self.test_embedder_classes = {}
        print(self.test_embedder_names)
        # self.datasets_name = ['LIBRIALL']
        # self.datasets_configs = []


        for model_name in (self.test_embedder_names):
            self._set_model(model_name)

        # for dataset_name in (self.datasets_name):
        #     self._set_dataset(dataset_name)


        # Loss options
        self.dist_loss_type = 'cossim'
        self.dist_weight = 1
        self.tv_weight = 0.1

        self.similarity_params = {'dim': 2}
        similarity_func_name = 'CosineSimilarity'
        self._set_similarity_one_func(similarity_func_name)

        self.update_current_dir()

    def set_attribute(self, name, value):
        setattr(self, name, value)

    def _set_similarity_one_func(self, similarity_func_name):
        # self.similarity_func_params = similarity_func_params
        with open(ROOT / 'configs/similarities_config.yaml', 'r') as stream:
            yaml_file = yaml.safe_load(stream)
            self.similarity_config_one = yaml_file[similarity_func_name]

    def _set_dataset(self, dataset_name):
        with open(ROOT / 'configs/dataset_config.yaml', 'r') as stream:
            self.dataset_config = yaml.safe_load(stream)[dataset_name]
            # self.datasets_configs(dataset_config)

    def _set_model(self, model_name):
        with open(ROOT / 'configs/model_config.yaml', 'r') as stream:
            curr_model_class = yaml.safe_load(stream)[model_name]
            self.test_embedder_classes[model_name] = curr_model_class

    def update_current_dir(self):
        my_date = datetime.datetime.now()
        month_name = my_date.strftime("%B")
        self.current_dir = os.path.join("experiments", month_name, time.strftime("%d-%m-%Y") + '_' + time.strftime("%H-%M-%S"))
        if 'SLURM_JOBID' in os.environ.keys():
            self.current_dir += '_' + os.environ['SLURM_JOBID']





class BaseEvalConfig:
    def __init__(self):
        self.root_path = ROOT
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model_source = 'SpeechBrain'  # see configs/model_config.yaml for other options
        model_name = 'spkrec-ecapa-voxceleb'  # see configs/model_config.yaml for other options
        self._set_model(model_source, model_name)

        dataset_name = 'LIBRIALL'
        self._set_dataset(dataset_name)

        # # TODO: split to two classes. one for eval, other for attack. delete similarity from BaseConfig
        # self.similarity_params = {'dim': 2}
        # similarity_func_name = 'CosineSimilarity'
        # self._set_similarity_one_func(similarity_func_name)

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
            'batch_size': 128,
            'num_workers': 4
        }

        self.num_wavs_for_emb = 3
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


class UniversalAttackEval(BaseConfiguration):
    def __init__(self):
        super(UniversalAttackEval, self).__init__()
        self.patch_name = 'universal'
        # self.num_of_train_images = 5
        # self.train_batch_size = 4
        self.test_batch_size = 32

        self.masks_path = os.path.join('..', 'data', 'masks')
        self.random_mask_path = os.path.join(self.masks_path, 'random.png')

        # Test dataset options
        self.test_num_of_images_for_emb = 5
        self.test_dataset_names = ['LIBRI-CLOSE-SET-TEST',"LIBRI-TEST" ]#"VOXCELEB1-CLOSE-SET-TEST"] #"VOXCELEB1-OPEN-SET-TEST"]#, "VOXCELEB1-CLOSE-SET-TEST", "LIBRI-TEST",'LIBRI-CLOSE-SET-TEST'] #'LIBRI-TEST'] # , 'LIBRI-CLOSE-SET-TEST']
        self.test_img_dir = {name: self.set_dataset(name) for name in self.test_dataset_names}
        # self.test_img_dir = {name: os.path.join('..', 'datasets', name) for name in self.test_dataset_names}
        self.test_number_of_people = 40
        self.test_celeb_lab = {}

        for dataset_name, img_dir in self.test_img_dir.items():
            label_list = os.listdir(img_dir['root_path'])[:self.test_number_of_people]
            if dataset_name == self.train_dataset_name:
                label_list = os.listdir(img_dir['root_path'])[-self.test_number_of_people:]
            self.test_celeb_lab[dataset_name] = label_list
        self.test_celeb_lab_mapper = {dataset_name: {i: lab for i, lab in enumerate(self.test_celeb_lab[dataset_name])}
                                      for dataset_name in self.test_dataset_names}

        # number_of_speakers = 100
        # speaker_labels = os.listdir(self.dataset_config['root_path'])[:number_of_speakers]
        # speaker_labels_mapper = {i: lab for i, lab in enumerate(speaker_labels)}


    def set_dataset(self, dataset_name):
        with open(ROOT / 'configs/dataset_config.yaml', 'r') as stream:
            return yaml.safe_load(stream)[dataset_name]