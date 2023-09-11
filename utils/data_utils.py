import fnmatch
import glob
import json
import math
import os
import pickle
import random
import sys
from pathlib import Path
from warnings import warn
from visualization.plots import plot_waveform
from utils.general import calculator_snr_direct, calculate_snr_github_direct
import matplotlib.pyplot as plt
import datetime
import numpy as np
import pandas as pd
import torch
import torchaudio
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchaudio.datasets import LIBRISPEECH
from tqdm import tqdm

torchaudio.set_audio_backend("sox_io")
global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

URL = "train-clean-100"
FOLDER_IN_ARCHIVE = "LibriSpeech"

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

class BasicLibriSpeechDatasetTest(Dataset):
    def __init__(self, root_path, speaker_labels_mapper, indices, transform=None):
        # self.files_path , self.labels_df = create_labels_from_path(data_root_dir)
        self.root_path = root_path
        self.speaker_labels_mapper = {lab: i for i, lab in speaker_labels_mapper.items()}

        def extend_speakers_utterances():
            perturb_size = 48000 # current perturb size
            wav_names = self.get_wav_names(indices)
            update_names = []
            update_indices = []
            start_indices = []
            for idx, wav_path in zip(indices,wav_names):
                signal, fs = get_signal_and_fs_from_wav(wav_path, perturb_size=perturb_size) # torchaudio.load(wav_name)
                signal_len = signal.shape[1]
                ceil_num = math.floor(signal_len / perturb_size) # 3 is the uap size
                update_names.extend([wav_path] * ceil_num)
                update_indices.extend([idx] * ceil_num)
                start_indices.extend(list(range(0, ceil_num )))


            return update_names, update_indices, start_indices

        self.wav_names, self.update_indices, self.start_indices = extend_speakers_utterances()

        self.orig_wav_names = self.get_wav_names(indices)
        self.orig_indices = indices
        self.mode = 'Test'
        self.transform = transform # signal, fs = torchaudio.load(wav_path)

    def __len__(self):
        return len(self.wav_names)

    def __getitem__(self, idx):
        wav_path = self.wav_names[idx]
        st_idx = self.start_indices[idx]
        if not wav_path:
            print("wav path: ", wav_path)
        cropped_signal = get_signal_from_wav_test(wav_path, start_idx=st_idx)
        label = wav_path.split(os.path.sep)[-2]
        if self.transform:
            cropped_signal = self.transform(cropped_signal)
        return cropped_signal, self.speaker_labels_mapper[label]

# TODO: change to .flac (libri) instead .wav(voxceleb) to support libri-train-100
    def get_wav_names(self, indices):
        files_in_folder = get_nested_dataset_files(self.root_path, self.speaker_labels_mapper.keys())
        files_in_folder = [item for sublist in files_in_folder for item in sublist]
        if indices is not None:
            files_in_folder = [files_in_folder[i] for i in indices]
        wavs = fnmatch.filter(files_in_folder, '*.wav')
        return wavs



class BasicLibriSpeechDataset(Dataset):
    def __init__(self, root_path, speaker_labels_mapper, indices, transform=None):
        self.root_path = root_path
        self.speaker_labels_mapper = {lab: i for i, lab in speaker_labels_mapper.items()}
        self.wav_names = self.get_wav_names(indices)
        self.transform = transform

    def __len__(self):
        return len(self.wav_names)

    def __getitem__(self, idx):
        wav_path = self.wav_names[idx]
        if not wav_path:
            print("wav path: ", wav_path)
        cropped_signal = get_signal_from_wav_random(wav_path)

        label = wav_path.split(os.path.sep)[-2]
        if self.transform:
            cropped_signal = self.transform(cropped_signal)
        return cropped_signal, self.speaker_labels_mapper[label]

# TODO: change to .flac instead .wav to support libri-train-100
    def get_wav_names(self, indices):
        files_in_folder = get_nested_dataset_files(self.root_path, self.speaker_labels_mapper.keys())
        files_in_folder = [item for sublist in files_in_folder for item in sublist]
        if indices is not None:
            files_in_folder = [files_in_folder[i] for i in indices]
        wavs = fnmatch.filter(files_in_folder, '*.wav')
        return wavs


#TODO: changed from .wav(voxceleb) to .flac(libri) ('**/*.') to support libri-train-100, change to
def get_nested_dataset_files(img_dir, labels):
    files_in_folder = [glob.glob(os.path.join(img_dir, lab, '**/*.wav'), recursive=True) for lab in labels]
    return files_in_folder

# TODO: train index depend on num_of_samples which depend on samples to create embbedings from. changed to fit 70% train
def get_split_indices(img_dir, labels, test_num_of_images_for_emb,split_rate=0.8,istrain=False):
    dataset_nested_files = get_nested_dataset_files(img_dir, labels)

    nested_indices = [np.array(range(len(arr))) for i, arr in enumerate(dataset_nested_files)]
    nested_indices_continuous = [nested_indices[0]]
    for i, arr in enumerate(nested_indices[1:]):
        nested_indices_continuous.append(arr + nested_indices_continuous[i][-1] + 1)

    num1 = random.randint(0, 100)
    if istrain:
        np.random.seed(0)
    train_indices = np.concatenate([np.random.choice(arr_idx, size=math.floor(len(arr_idx) * split_rate), replace=False) for arr_idx in nested_indices_continuous])

    val_indices = list(set(list(range(nested_indices_continuous[-1][-1]))) - set(train_indices))

    return train_indices, val_indices

def apply_perturbation(src_file, perturb, device,eps=1.0):
    adv_signal = src_file + eps * perturb
    return adv_signal

@torch.no_grad()
def get_person_embedding(config, loader, person_ids, embedders, device, include_others=False):
    print('Calculating persons embeddings {}...'.format('with mask' if include_others else 'without mask'), flush=True)
    embeddings_by_embedder = {}
    for embedder_name, embedder in embedders.items():
        person_embeddings = {i: torch.empty(0, device=device) for i in range(len(person_ids))}
        for img_batch, person_indices in tqdm(loader):
            img_batch = img_batch.to(device)
            embedding = embedder.encode_batch(img_batch).squeeze().detach()# embedder.encode(img_batch)
            for idx in person_indices.unique():
                relevant_indices = torch.nonzero(person_indices == idx, as_tuple=True)
                emb = embedding[relevant_indices]
                person_embeddings[idx.item()] = torch.cat([person_embeddings[idx.item()], emb], dim=0)
        final_embeddings_small = [get_constant_size_emb(person_emb,test=True) for person_emb in
                                  person_embeddings.values()]  # TODO: add constrain on embeding size
        final_embeddings = torch.cat(final_embeddings_small,dim=0)
        embeddings_by_embedder[embedder_name] = final_embeddings
    return embeddings_by_embedder




def get_dataset(dataset_name,test_mode=False):
    if 'LIBRI' in dataset_name and test_mode:
        return BasicLibriSpeechDatasetTest
    elif 'VOX' in dataset_name and test_mode:
        return BasicLibriSpeechDatasetTest
    elif 'LIBRI' in dataset_name:
        return BasicLibriSpeechDataset
    elif 'VOX' in dataset_name:
        return BasicLibriSpeechDataset



def get_loaders(loader_params, dataset_config, splits_to_load, **kwargs):
    dataset_name = dataset_config['dataset_name']
    train_indices, val_indices = get_split_indices(dataset_config['root_path'],
                                                   dataset_config['speaker_labels'],
                                                   dataset_config['num_wavs_for_emb'], istrain=True)
    train_loader, val_loader, test_loader = None, None, None
    dataset = get_dataset(dataset_name)
    if 'train' in splits_to_load:
        train_data = dataset(root_path=dataset_config['root_path'],
                             speaker_labels_mapper=dataset_config['speaker_labels_mapper'],
                             indices=train_indices,
                             **kwargs)
        train_loader = DataLoader(train_data,
                                  batch_size=loader_params['batch_size'],
                                  num_workers=loader_params['num_workers'],
                                  shuffle=True,
                                  pin_memory=True)
    if 'validation' in splits_to_load:
        val_data = dataset(root_path=dataset_config['root_path'],
                           speaker_labels_mapper=dataset_config['speaker_labels_mapper'],
                           indices=val_indices,
                           **kwargs)
        val_loader = DataLoader(val_data,
                                batch_size=loader_params['batch_size'],
                                num_workers=loader_params['num_workers'] // 2,
                                shuffle=False,
                                pin_memory=True)
    if 'test' in splits_to_load:
        test_data = dataset(root_path=dataset_config['root_path'],
                            **kwargs)
        test_loader = DataLoader(test_data,
                                 batch_size=loader_params['batch_size'],
                                 num_workers=loader_params['num_workers'] // 2,
                                 shuffle=False,
                                 pin_memory=True)

    return train_loader, val_loader, test_loader

def update_loader_indices(indices,loader, perturb_size = 48000):
    update_indices = []
    for wav_path in loader.wav_names:
        signal, fs = get_signal_and_fs_from_wav(wav_path, perturb_size=perturb_size)  # torchaudio.load(wav_name)
        signal_len = signal.shape[1]
        ceil_num = math.floor(signal_len / perturb_size)  # 3 is the uap size
        update_indices.extend([wav_path] * ceil_num)

    return update_indices

def get_test_loaders(config, dataset_names):
    emb_loaders = {}
    test_loaders = {}
    for dataset_name in dataset_names:
        test_indices, emb_indices = get_split_indices(config.test_img_dir[dataset_name]['root_path'],
                                                      config.test_celeb_lab[dataset_name],
                                                      config.test_num_of_images_for_emb,split_rate=0.5)

        dataset = get_dataset (dataset_name, test_mode=True) # (dataset_name, test_mode=True)# (dataset_name)
        emb_dataset = dataset(root_path=config.test_img_dir[dataset_name]['root_path'],
                                     speaker_labels_mapper=config.test_celeb_lab_mapper[dataset_name],
                                     indices=emb_indices,
                              )
        emb_loader = DataLoader(emb_dataset, batch_size=config.test_batch_size)
        emb_loaders[dataset_name] = emb_loader

        t_dataset = get_dataset(dataset_name, test_mode=True)
        test_dataset = t_dataset(root_path=config.test_img_dir[dataset_name]['root_path'],
                                      speaker_labels_mapper=config.test_celeb_lab_mapper[dataset_name],
                                      indices=test_indices
                                      )
        test_loader = DataLoader(test_dataset, batch_size=config.test_batch_size)
        test_loaders[dataset_name] = test_loader

    return emb_loaders, test_loaders

def get_anc_wav_from_json(all_speaker_paths, perturb_size=48000, sample_size=1):
    for choose_path_idx in range(0, sample_size):
        anc_idx = random.sample(range(0, len(all_speaker_paths)), 1)[0]
        anc_path = all_speaker_paths[anc_idx]
        signal, fs = torchaudio.load(all_speaker_paths[anc_idx])

        if signal.shape[1] >= perturb_size:
            return signal, fs, anc_path

        else:
            cat_size = math.ceil(perturb_size / signal.shape[1])
            cat_size = [signal for i in range(cat_size)]
            signal = torch.cat(cat_size, 1)
            return signal, fs, anc_path


def create_dirs_not_exist(current_path, directories_list):
    for dir_name in directories_list:
        dir_path = os.path.join(current_path,dir_name)
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)


def save_to_pickle(root_path, pickle_data, audio_type, file_name):
    file_path = os.path.join(root_path, audio_type, Path(file_name).stem)
    with open(f'{file_path}.pickle', 'wb') as f:
        pickle.dump(pickle_data, f, protocol=pickle.HIGHEST_PROTOCOL)


def save_audio_as_wav(root_path, waveform, audio_type,file_name, sample_rate=16000, src_voice=""):
    file_path = os.path.join(root_path, audio_type, Path(file_name).name)
    if '.flac' in file_name:
        file_path = os.path.join(root_path, audio_type, f'{Path(file_name).stem}{src_voice}.wav')
    torchaudio.save(file_path, waveform, sample_rate,encoding="PCM_F", bits_per_sample=32)


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


def save_emb_as_npy(root_path, np_data, audio_type, file_name):
    file_path = os.path.join(root_path, audio_type, Path(file_name).stem)
    np.save(f'{file_path}.npy', np_data)


def load_from_npy(root_path, audio_type, file_name):
    file_path = os.path.join(root_path, audio_type, Path(file_name).stem)
    return np.load(f'{file_path}.npy')


def create_labels_from_path(dir_path, suffix_p):
    all_spk_paths = glob.glob(f"{ROOT}/{dir_path}{suffix_p}.wav") # in case of parent dir use /*/*.wav instead
    labels = [int(os.path.basename(p).split('-')[0]) for p in all_spk_paths]
    labels_df = pd.DataFrame(labels)
    return all_spk_paths, labels_df


def create_speakers_list(root_data_path):

    return os.listdir(root_data_path)

def load_labels():
    data = np.load('./data/TIMIT/speaker/processed/TIMIT_labels.npy', allow_pickle=True)

def read_npy(file_name):
    return np.load(f'{file_name}.npy')


def split_data_to_train_test(path):
    path_obj = Path(path)
    filename = path_obj.stem
    Path(f'{path_obj.parent}/data_lists').mkdir(parents=True, exist_ok=True)
    id_dir_to_train = os.listdir(path)
    train_files = defaultdict()
    test_files = defaultdict()
    for speaker_id in tqdm(id_dir_to_train, desc="Speaker_id:"):
        all_speaker_path = [f for f in glob.glob(f"{path}/{speaker_id}/*.wav")]
        x_train ,x_test = train_test_split(all_speaker_path,test_size=0.2,shuffle=False) # $$$ change to true in eval
        train_files[speaker_id] = x_train
        test_files[speaker_id] = x_test

    write_json(f'{Path(path).parent}/data_lists/{filename}/train_files',train_files)
    write_json(f'{Path(path).parent}/data_lists/{filename}/test_files',test_files)

from collections import defaultdict

def create_enroll_from_json(train_path,model_instance):

    trainind_model = model_instance
    train_data = read_json(train_path)

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

      if speaker_id not in enroll_emb_dict.keys():
          enroll_emb_dict[speaker_id] = embd_np
    write_json(f'{Path(train_path).parent}/enroll_files_dict',enroll_files_dict)
    write_pickle(f'{Path(train_path).parent}/enroll_emb_dict',enroll_emb_dict)


def get_signal_and_fs_from_wav(wav_path, perturb_size=48000):
    signal, fs = torchaudio.load(wav_path)
    if signal.shape[1] > perturb_size + 1:
        return signal, fs

    else:
        c_size = math.ceil(perturb_size / signal.shape[1]) +1
        cat_size = [signal for i in range(c_size)]
        signal = torch.cat(cat_size, 1)
    return signal, fs


def get_signal_from_wav_random(wav_path, perturb_size=48000):
    signal, _ = get_signal_and_fs_from_wav(wav_path, perturb_size=perturb_size)
    signal_len = signal.shape[1]
    if (signal_len - (perturb_size + 1)) <= 0:
        print("temp <=0: ",wav_path )
    start_idx = np.random.randint(low=0, high=(signal_len - (perturb_size + 1)), dtype=np.int64())
    cropped_signal = signal[0][start_idx: start_idx + perturb_size]

    return cropped_signal

def get_signal_from_wav_test(wav_path, perturb_size=48000,start_idx=0):
    signal, _ = get_signal_and_fs_from_wav(wav_path, perturb_size=perturb_size)
    signal_len = signal.shape[1]
    cropped_signal = signal[0][start_idx: start_idx + perturb_size]

    return cropped_signal

@torch.no_grad()
def get_speaker_embedding(model, data_path, speaker_id, num_samples, fs, device):
    speaker_emb = torch.empty(0, device=device)
    speaker_wavs = os.listdir(os.path.join(data_path, speaker_id))
    wavs_for_embedding = random.sample(speaker_wavs, num_samples)
    for speaker_wav in wavs_for_embedding:
        signal, _ = get_signal_and_fs_from_wav(os.path.join(data_path, speaker_id, speaker_wav))
        print("utter_emb: ",  speaker_wav)
        if signal == None:
            print(f'SKIP anchor: {speaker_id}')
            continue
        signal_len = signal.shape[1]
        start_idx = random.randint(0, signal_len - (fs * 3 + 1))
        cropped_signal = signal[0][start_idx: start_idx + (fs * 3)]
        cropped_signal = cropped_signal.to(device)
        embedding = model.encode_batch(cropped_signal)
        embedding = embedding.detach()
        speaker_emb = torch.cat([speaker_emb, embedding], dim=0)

    return speaker_emb.mean(dim=0)


def get_constant_size_emb(person_emb, emb_size=5,test=False):
    indices = [5,15,20,25,30]
    if test:
        indices = [0,1,2,3,4]
    speaker_emb = person_emb[indices]
    speaker_emb_mean = torch.mean(speaker_emb,dim=0)
    return speaker_emb_mean.unsqueeze(0)

@torch.no_grad()
def get_embeddings(model, loader, person_ids, device):
    embeddings = {id: torch.empty(0, device=device) for id in person_ids}
    for cropped_signal_batch, person_ids_batch in tqdm(loader):
        cropped_signal_batch = cropped_signal_batch.to(device)
        embedding = model.encode_batch(cropped_signal_batch).detach()
        for idx in person_ids_batch.unique():
            relevant_indices = torch.nonzero(person_ids_batch == idx, as_tuple=True)
            emb = embedding[relevant_indices]
            embeddings[idx.item()] = torch.cat([embeddings[idx.item()], emb], dim=0)
    # final_embeddings = [person_emb.mean(dim=0).unsqueeze(0) for person_emb in embeddings.values()] # TODO: add constrain on embeding size
    final_embeddings_small = [ get_constant_size_emb(person_emb) for person_emb in embeddings.values()]  # TODO: add constrain on embeding size
    final_embeddings = torch.cat(final_embeddings_small, dim=0)
    return final_embeddings


def save_perturbation(waveform, name):
    save_emb_as_npy("..", waveform, "output_files", name)
    save_audio_as_wav("..", waveform.float(), "output_files", f'{name}.wav')

    plot_waveform(waveform, 16000)




