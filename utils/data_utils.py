import torch
import random
import torchaudio
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
import glob
# import cv2
import random
import os
import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from pathlib import Path
import yaml
import collections
import itertools
import math
from collections import Counter
from pathlib import Path
from models.speechbrain_ecpa import AttackSRModel

class LITDataset(Dataset):
    def __init__(self, root_path, split):
        self.input_size = 1600
        self.output_size = 500
        self.split = split
        self.num_channels = 1

        self.patterns = np.load(os.path.join(root_path, f'{split}_patterns.npy'))
        self.targets = np.load(os.path.join(root_path, f'{split}_targets.npy'))

    def __len__(self):
        return len(self.patterns)

    def __getitem__(self, idx):
        # img = cv2.resize(img, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        pattern_ = np.load(self.patterns[idx])
        # r, g, b = cv2.split(pattern_)
        # pattern_ = np.dstack((b, g, r))
        # target_ = cv2.imread(self.targets[idx])

        c = random.randint(0, 2)
        pattern = pattern_[:, :, c]
        # target = target_[:, :, c]

        pattern = (pattern - pattern.mean()) / pattern.std()
        # target = cv2.resize(target, (self.output_size, self.output_size), interpolation=cv2.INTER_AREA)
        # target = cv2.normalize(target, None, 0, 255, cv2.NORM_MINMAX)
        # return np.reshape(pattern, (self.num_channels, self.input_size, self.input_size)), \
        #        np.reshape(target, (self.num_channels, self.output_size, self.output_size))


class DiffuserDataset(Dataset):
    def __init__(self, root_path, labels_dict, load_lensed=False, random_pair=False, diff_transform=None, lensed_transform=None, cropping=False, load_reconstruct=False, **kwargs):
        # sorted_labels_dict = collections.OrderedDict(sorted(labels_dict.items()))
        files_idx = map(str, labels_dict.keys())
        files_names = ['im' + sub + '.npy' for sub in files_idx]
        self.diffuser_list = [os.path.join(root_path, 'diffuser', file) for file in files_names]
        self.lensed_list = [os.path.join(root_path, 'original', file) for file in files_names]
        if load_reconstruct:
            files_idx = map(str, labels_dict.keys())
            # files_names = ['im' + sub + '.jpg' for sub in files_idx]
            files_names = ['im' + sub + '.npy' for sub in files_idx]
            self.lensed_list = [os.path.join(root_path, 'reconstructions/dataset', file) for file in files_names]
        self.labels = np.array(list(map(int, labels_dict.values())))
        self.cropping = cropping

        self.random_pairs = None
        if random_pair:
            self.random_pairs = []
            indices = list(range(len(self.diffuser_list)))
            for i in range(len(indices)):
                prob = np.where(self.labels[i] == self.labels[indices], 0, 1)
                self.random_pairs.append(np.random.choice(indices, p=prob/sum(prob)))

        # diffuser_files = sorted(os.listdir(os.path.join(root_path, 'diffuser')), key=lambda x: int(x[2:-4]))
        # self.diffuser_list = [os.path.join(root_path, 'diffuser', file) for file in diffuser_files if file.endswith(('.npy'))]
        # original_files = sorted(os.listdir(os.path.join(root_path, 'original')), key=lambda x: int(x[2:-4]))
        # self.original_list = [os.path.join(root_path, 'original', file) for file in original_files if file.endswith(('.npy'))]
        # self.load_orig = load_orig

        # self.labels = None
        # if labels_file:
        #     with open(os.path.join(labels_file), 'rb') as f:
        #         self.labels = pickle.load(f)

        # if class_to_load is not None:
        #     keep = np.zeros(len(self.labels), dtype=np.bool)
        #     for cls in class_to_load:
        #         keep |= ((self.labels[:, cls] == 1) & (self.labels.sum(axis=1) == 1))
        #     self.labels = self.labels[keep]
        #     self.diffuser_list = self.diffuser_list[keep]
        #     self.original_list = self.original_list[keep]

        self.diff_transform = diff_transform
        self.lensed_transform = lensed_transform
        self.load_lensed = load_lensed
        self.load_reconstruct = load_reconstruct

    def __len__(self):
        return len(self.diffuser_list)

    def __getitem__(self, idx):
        diffuser_source_img = np.load(self.diffuser_list[idx]).transpose((2, 0, 1))
        lensed_source_img, lensed_target_img = -1, -1
        if self.load_lensed:
            if not self.load_reconstruct:  # replaced original image with reconstruct
                lensed_source_img = self.load_image_from_list_in_numpy(self.lensed_list[idx])
            else:
                lensed_source_img = self.load_image_from_list_in_numpy(self.lensed_list[idx])

        target_idx, diffuser_target_img = -1, -1
        if self.random_pairs:
            target_idx = self.random_pairs[idx]
            diffuser_target_img = np.load(self.diffuser_list[target_idx]).transpose((2, 0, 1))

            if self.load_lensed:
                lensed_target_img_name = self.lensed_list[target_idx]
                lensed_target_img = self.load_image_from_list_in_numpy(lensed_target_img_name)

        if self.diff_transform:
            diffuser_source_img = self.diff_transform(diffuser_source_img)
            if diffuser_target_img != -1:
                diffuser_target_img = self.diff_transform(diffuser_target_img)

        if self.lensed_transform:
            lensed_source_img = self.lensed_transform(lensed_source_img)
            if lensed_target_img != -1:
                lensed_target_img = self.lensed_transform(lensed_target_img)

        source_label, target_label = -1, -1
        source_label = self.labels[idx]
        if self.random_pairs is not None:
            target_label = self.labels[target_idx]

        diffuser_source_img_name = self.diffuser_list[idx]
        return [diffuser_source_img, lensed_source_img, source_label,
                diffuser_target_img, lensed_target_img, target_label,
                diffuser_source_img_name]

    def load_image_from_list_in_numpy(self, img_path):
        source_img = np.clip(np.load(img_path, allow_pickle=True), 0, 1)
        source_img = np.flipud(source_img).copy()
        if self.cropping:
            source_img = source_img[60:, 62:-38, :]
        source_img = source_img.transpose((2, 0, 1))
        source_img = source_img[[2, 1, 0]]  # BGR to RGB
        return source_img

    def load_image_from_list_in_jpg(self, img_path):
        # source_img = cv2.imread(img_path) / 255
        # source_img = cv2.copyMakeBorder(source_img, 60, 0, 62, 38, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        # source_img = source_img.transpose((2, 0, 1))
        # source_img = source_img[[2, 1, 0]]  # BGR to RGB
        # return source_img
        pass


def get_dataset(dataset_name):
    if dataset_name == 'LIT':
        return LITDataset
    elif dataset_name == 'DiffuserCam':
        return DiffuserDataset


def get_loaders(loader_params, dataset_config, splits_to_load, **kwargs):
    dataset_name = dataset_config['dataset_name']
    print('Loading {} dataset...'.format(dataset_name))
    train_loader, val_loader, test_loader = None, None, None
    dataset = get_dataset(dataset_name)
    if 'train' in splits_to_load:
        train_set = load_pickle_file(os.path.join(dataset_config['root_path'],
                                                  'split_balanced_annotations',
                                                  'train.pickle'))
        train_data = dataset(root_path=dataset_config['root_path'],
                             labels_dict=train_set,
                             **kwargs)
        train_loader = DataLoader(train_data,
                                  batch_size=loader_params['batch_size'],
                                  num_workers=loader_params['num_workers'],
                                  shuffle=True,
                                  pin_memory=True)
    if 'validation' in splits_to_load:
        val_set = load_pickle_file(os.path.join(dataset_config['root_path'],
                                                'split_balanced_annotations',
                                                'validation.pickle'))
        val_data = dataset(root_path=dataset_config['root_path'],
                           labels_dict=val_set,
                           **kwargs)
        val_loader = DataLoader(val_data,
                                batch_size=loader_params['batch_size'],
                                num_workers=loader_params['num_workers'] // 2,
                                shuffle=False,
                                pin_memory=True)
    if 'test' in splits_to_load:
        test_set = load_pickle_file(os.path.join(dataset_config['root_path'],
                                                 'split_balanced_annotations',
                                                 'test.pickle'))
        test_data = dataset(root_path=dataset_config['root_path'],
                            labels_dict=test_set,
                            **kwargs)
        test_loader = DataLoader(test_data,
                                 batch_size=loader_params['batch_size'],
                                 num_workers=loader_params['num_workers'] // 2,
                                 shuffle=False,
                                 pin_memory=True)


    train_data = ConcatDataset([train_data, val_data])
    train_loader = DataLoader(train_data,
                              batch_size=loader_params['batch_size'],
                              num_workers=loader_params['num_workers'],
                              shuffle=True,
                              pin_memory=True)
    return train_loader, val_loader, test_loader


def load_pickle_file(path):
    with open(path, "rb") as f:
        pickled_file = pickle.load(f)
    return pickled_file


MIRFLICKER_classes = [
    'people_r1',
    'dog_r1',
    'car_r1',
    'tree_r1',
    'flower_r1'
]


def get_loaders_demo():
    dataset_config = yaml.safe_load(Path('../configs/dataset_config.yaml').read_text())
    loader_config = yaml.safe_load(Path('../configs/loader_config.yaml').read_text())
    splits_to_load = ['train', 'validation', 'test']
    get_loaders(loader_params=loader_config, dataset_config=dataset_config, splits_to_load=splits_to_load)


class CreateAnnotationsFiles:
    def __init__(self,dataset_path):
        self.dataset_path = dataset_path
        self.annotations_path = os.path.join(dataset_path,'annotations')
        self.annotations = ['people_r1', 'dog_r1', 'car_r1', 'tree_r1', 'flower_r1']

    def create_annotation(self):
        idx_list = np.arange(0, 25001, 1)
        labels_list = np.ones(25001)*-1
        labels_dict = dict(zip(idx_list, labels_list))
        for file in tqdm(self.annotations, desc ="Loading annotations"):
            file_path = os.path.join(self.annotations_path, f'{file}.txt')
            with open(file_path) as f:
                lines = f.read().splitlines()
                for line in lines:
                    if labels_dict[int(line)] != -1:
                        labels_dict[int(line)] = -1
                        continue
                    labels_dict[int(line)] = self.annotations.index(file)

        labels_dict = self.balance_data1(labels_dict)
        train,val,test = self.train_val_test_split(labels_dict)
        self.dump_files((train, val, test))

    def balance_data(self, labels_dict):
        labels_dict = {key: value for (key, value) in labels_dict.items() if value != -1}
        dict_without_people_class = {key: value for (key, value) in labels_dict.items() if value != 0}
        dict_with_only_people_class = {key: value for (key, value) in labels_dict.items() if value == 0}
        dict_with_sliced_people_class = dict(itertools.islice(dict_with_only_people_class.items(), 650))
        merged_dicts = {**dict_without_people_class, **dict_with_sliced_people_class}
        return merged_dicts

    def balance_data1(self, labels_dict):
        labels_dict = {key: value for (key, value) in labels_dict.items() if value != -1}
        class_count = Counter(labels_dict.values())
        max_samples = class_count.most_common()[-1][1]
        merged_dicts = {}
        for i in range(len(MIRFLICKER_classes)):
        # dict_without_people_class = {key: value for (key, value) in labels_dict.items() if value != 0}
            dict_with_only_specific_class = {key: value for (key, value) in labels_dict.items() if value == i}
            dict_with_only_specific_class = list(dict_with_only_specific_class.items())
            random.shuffle(dict_with_only_specific_class)
            dict_with_only_specific_class = dict(dict_with_only_specific_class)
            dict_with_sliced_specific_class = dict(itertools.islice(dict_with_only_specific_class.items(), max_samples))
            merged_dicts = {**merged_dicts, **dict_with_sliced_specific_class}
        return merged_dicts

    def train_val_test_split(self, labels_dict):
        X_train, X_test, y_train, y_test = train_test_split(list(labels_dict.keys()), list(labels_dict.values()), test_size = 0.2, random_state = 42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state = 42)
        return dict(zip(X_train, y_train)), dict(zip(X_val, y_val)),dict(zip(X_test, y_test))

    def dump_files(self,datasets):
        names = ['train', 'validation', 'test']
        for dataset,name in zip(datasets,names):
            with open(f'{name}.pickle', 'wb') as file:
                pickle.dump(dataset, file, protocol=pickle.HIGHEST_PROTOCOL)


def create_annotations_file_demo():
    dt = CreateAnnotationsFiles("/dt/shabtaia/dt-fujitsu/Lensless_imaging/Datasets/DiffuserCam")
    dt.create_annotation()
# create_annotations_file_demo()


def explore_images():
    dataset_path = "/dt/shabtaia/dt-fujitsu/Lensless_imaging/Datasets/DiffuserCam/original"
    for i in range(20):
        img_path = os.path.join(dataset_path,f"im{i}.npy")
        original_source_img = np.load(img_path)*255
        original_source_img = np.rot90(np.rot90(original_source_img))
        original_source_img = original_source_img[60:, 62:-60, :]
        # cv2.imwrite(f"/dt/shabtaia/dt-fujitsu/Lensless_imaging/Datasets/test_check/check_{i}.jpg", original_source_img)


def create_dataset_labels(labels_path, output_path, dataset_size):
    files = os.listdir(labels_path)
    labels = np.zeros((dataset_size, len(files)))
    labels_map = {}
    for cls_num, file in enumerate(files):
        labels_map[cls_num] = file.split('_')[0]
        with open(os.path.join(labels_path, file), 'r') as f:
            for line in f.readlines():
                labels[int(line.strip())-1, cls_num] = 1

    with open(os.path.join(output_path, 'labels_mat.pkl'), 'wb') as f_out:  # Pickling
        pickle.dump(labels, f_out)
    with open(os.path.join(output_path, 'labels_dict.txt'), 'w') as f_out:  # Pickling
        f_out.write(str(labels_map))





def get_anc_wav(all_rec_path):
    for try_rec_idx in range(0, 8):
        all_neg_rec_path = [f for f in glob.glob(f"{all_rec_path}/*.wav")]
        # print(all_neg_rec_path)
        # print("size: ",len(all_neg_rec_path))
        neg_rec_idx = random.sample(range(0, len(all_neg_rec_path)), 1)[0]
        neg_res_path = f'{all_neg_rec_path[neg_rec_idx]}'
        signal2, fs = torchaudio.load(neg_res_path)

        if signal2.shape[1] >= 48000:
            return signal2, fs, neg_res_path
    return None, None, None


def get_neg_wav(all_neg_id_path):
    for try_idx in range(0, 10):
        neg_id_idx = random.sample(range(0, len(all_neg_id_path)), 1)[0]
        signal2, fs, res_path = get_anc_wav(f"{all_neg_id_path[neg_id_idx]}")
        if signal2 == None:
            continue
        if signal2.shape[1] >= 48000:
            return signal2, fs, res_path
    return None, None, None


def get_anc_wav_from_json(all_speaker_paths, perturb_size=48000, sample_size=1):
    for choose_path_idx in range(0, sample_size):
        anc_idx = random.sample(range(0, len(all_speaker_paths)), 1)[0]
        # print("anc idx: ",anc_idx)
        anc_path = all_speaker_paths[anc_idx]
        # print("anc_path: ",anc_path)
        signal, fs = torchaudio.load(all_speaker_paths[anc_idx])

        if signal.shape[1] >= perturb_size:
            return signal, fs, anc_path

        else:
            cat_size = math.ceil(perturb_size / signal.shape[1])
            cat_size = [signal for i in range(cat_size)]
            # print('***NONE***')
            # print("signal before: " ,signal.shape[1])
            signal = torch.cat(cat_size, 1)
            # print(signal.shape[1])
            return signal, fs, anc_path


def get_neg_wav_from_json(negative_id_speakers, train_data):
    neg_speaker_idx = random.sample(range(0, len(negative_id_speakers)), 1)[0]
    for try_idx in range(0, 1):
        signal, fs, neg_path = get_anc_wav_from_json(train_data[negative_id_speakers[neg_speaker_idx]])
    return signal, fs, neg_path


def prepare_wav_from_json(all_speaker_paths, perturb_size=48000, sample_size=1):
    signal, fs, anc_path = get_anc_wav_from_json(all_speaker_paths, perturb_size=48000, sample_size=1)

    signal_len = signal.shape[1]
    # idx_sec = random.sample(range(0, int(signal.shape[1]/(trainind_model.fs * 3))), 1)[0]
    start_idx = random.randint(0, signal_len - (perturb_size + 1))
    cropped_signal = signal[0][start_idx: (start_idx) + (perturb_size)]
    return cropped_signal, anc_path, start_idx



import json
def read_json(file_path):
  with open(file_path, 'r') as f:
    data = json.load(f)
  return data

def write_json(file_name, data):
  with open(f'{file_name}.json', 'w') as f:
      json.dump(data, f)

def read_pickle(file_name):
  with open(f'{file_name}.pickle', 'rb') as f:
      return pickle.load(f)

def write_pickle(file_name, data):
  with open(f'{file_name}.pickle', 'wb') as f:
      pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def write_npy(file_name, data):
    np.save(f'{file_name}.npy',data)

def read_npy(file_name):
    return np.load(f'{file_name}.npy')



def split_data_to_train_test(path):
    # TIMIT_PATH = "/content/drive/MyDrive/voice_attack/timit_train/dr3"
    # TIMIT_PATH = "/content/drive/MyDrive/voice_attack/libri/dr3"

    path_obj = Path(path)
    filename = path_obj.stem


    Path(f'{path_obj.parent}/data_lists').mkdir(parents=True, exist_ok=True)

    id_dir_to_train = os.listdir(path)

    train_files = defaultdict()
    test_files =  defaultdict()


    for speaker_id in tqdm(id_dir_to_train, desc="Speaker_id:"):
        all_speaker_path = [f for f in glob.glob(f"{path}/{speaker_id}/*.wav")]
        x_train ,x_test = train_test_split(all_speaker_path,test_size=0.2,shuffle=False) # $$$ change to true in eval
        train_files[speaker_id] = x_train
        test_files[speaker_id] = x_test



    write_json(f'{Path(path).parent}/data_lists/{filename}/train_files',train_files)
    write_json(f'{Path(path).parent}/data_lists/{filename}/test_files',test_files)

from collections import defaultdict

def create_enroll_from_json(train_path,model_instance):
    ### create enroll from json ###

    trainind_model = model_instance # AttackSRModel()
    train_data = read_json(train_path)
    # train_data = read_json("train_files.json")

    enroll_emb_dict = defaultdict()
    enroll_files_dict = defaultdict()


    for speaker_id in tqdm(train_data, desc="Speaker_id:"):
      signal, fs, anc_path = get_anc_wav_from_json(train_data[speaker_id])
      if signal == None:
        print(f'SKIP anchor: {speaker_id}')
        continue
      signal_len = signal.shape[1]
      start_idx = random.randint(0, signal_len -(trainind_model.fs * 3 + 1))
      enroll_files_dict[speaker_id] = {'anc':anc_path,'start_idx':start_idx}
      cropped_signal = signal[0][start_idx: (start_idx+1) + (trainind_model.fs * 3)]
      embd = trainind_model.forward_clean(cropped_signal,1)
      embd_np = embd.detach().numpy()[0]
      ### spk_id = anc_path.split('/')[-2]

      if speaker_id not in enroll_emb_dict.keys():
          enroll_emb_dict[speaker_id] = embd_np
    write_json(f'{Path(train_path).parent}/enroll_files_dict',enroll_files_dict)
    write_pickle(f'{Path(train_path).parent}/enroll_emb_dict',enroll_emb_dict)


def read_enroll_files():
    enroll_files_dict = read_json('enroll_files_dict.json')
    enroll_emb_dict = read_pickle('enroll_emb_dict')

def get_uap_perturb_npy(path):
    temp = [f for f in glob.glob(f"{path}/*.npy")]
    return temp
    # yield (f for f in glob.glob(f"{path}/*.npy"))