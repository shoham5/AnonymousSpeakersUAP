import fnmatch
import glob
import itertools
# import cv2
import json
import math
import os
import pickle
import random
import sys
from collections import Counter
from pathlib import Path
from warnings import warn

import numpy as np
import pandas as pd
import torch
import torchaudio
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchaudio.datasets import LIBRISPEECH
from tqdm import tqdm

torchaudio.set_audio_backend("sox_io")

URL = "train-clean-100"
FOLDER_IN_ARCHIVE = "LibriSpeech"

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))


def get_nested_dataset_files(img_dir, labels):
    files_in_folder = [glob.glob(os.path.join(img_dir, lab, '**/*.wav'), recursive=True) for lab in labels]
    return files_in_folder


def get_split_indices(img_dir, labels, num_of_samples):
    dataset_nested_files = get_nested_dataset_files(img_dir, labels)

    nested_indices = [np.array(range(len(arr))) for i, arr in enumerate(dataset_nested_files)]
    nested_indices_continuous = [nested_indices[0]]
    for i, arr in enumerate(nested_indices[1:]):
        nested_indices_continuous.append(arr + nested_indices_continuous[i][-1] + 1)
    train_indices = np.array([np.random.choice(arr_idx, size=num_of_samples, replace=False) for arr_idx in
                              nested_indices_continuous]).ravel()
    val_indices = list(set(list(range(nested_indices_continuous[-1][-1]))) - set(train_indices))

    return train_indices, val_indices


def get_test_loaders(config, dataset_names):
    emb_loaders = {}
    test_loaders = {}
    for dataset_name in dataset_names:
        emb_indices, test_indices = get_split_indices(config.test_img_dir[dataset_name],
                                                      config.test_celeb_lab[dataset_name],
                                                      config.test_num_of_images_for_emb)
        emb_dataset = CustomDataset1(img_dir=config.test_img_dir[dataset_name],
                                     celeb_lab_mapper=config.test_celeb_lab_mapper[dataset_name],
                                     img_size=config.img_size,
                                     indices=emb_indices,
                                     transform=transforms.Compose(
                                         [transforms.Resize(config.img_size),
                                          transforms.ToTensor()]))
        emb_loader = DataLoader(emb_dataset, batch_size=config.test_batch_size)
        emb_loaders[dataset_name] = emb_loader
        test_dataset = CustomDataset1(img_dir=config.test_img_dir[dataset_name],
                                      celeb_lab_mapper=config.test_celeb_lab_mapper[dataset_name],
                                      img_size=config.img_size,
                                      indices=test_indices,
                                      transform=transforms.Compose(
                                          [transforms.Resize(config.img_size),
                                           transforms.ToTensor()]))
        test_loader = DataLoader(test_dataset, batch_size=config.test_batch_size)
        test_loaders[dataset_name] = test_loader

    return emb_loaders, test_loaders


class BasicLibriSpeechDataset(Dataset):
    def __init__(self, root_path, speaker_labels_mapper, indices, transform=None):
        # self.files_path , self.labels_df = create_labels_from_path(data_root_dir)
        self.root_path = root_path
        self.speaker_labels_mapper = {lab: i for i, lab in speaker_labels_mapper.items()}
        self.wav_names = self.get_wav_names(indices)
        # self.files_path, self.labels_df = create_labels_from_path(root_path)
        # self.labels_dict_len = len(self.labels_df)

        self.transform = transform

    def __len__(self):
        return len(self.wav_names)

    def __getitem__(self, idx):
        # img_path = os.path.join(self.data_root_dir, self.labels.iloc[idx, 0])
        # audio_path = os.path.join(self.data_root_dir, self.files_path[idx])
        # print("self.files_path[idx]: ", self.wav_names[idx])
        wav_path = self.wav_names[idx]
        cropped_signal = get_signal_from_wav_random(wav_path)

        label = wav_path.split(os.path.sep)[-2]
        if self.transform:
            cropped_signal = self.transform(cropped_signal)
        return cropped_signal, self.speaker_labels_mapper[label]

    def get_wav_names(self, indices):
        files_in_folder = get_nested_dataset_files(self.root_path, self.speaker_labels_mapper.keys())
        files_in_folder = [item for sublist in files_in_folder for item in sublist]
        if indices is not None:
            files_in_folder = [files_in_folder[i] for i in indices]
        wavs = fnmatch.filter(files_in_folder, '*.wav')
        return wavs

# https://github.com/usc-sail/gard-adversarial-speaker-id/blob/master/dev/loaders/librispeech.py

class LibriSpeech4Speakers(LIBRISPEECH):
    def __init__(
            self, root, project_fs, subset, wav_length=None, url=URL,
            folder_in_archive=FOLDER_IN_ARCHIVE, download=False,
            train_speaker_ratio=1, train_utterance_ratio=0.8,
            return_file_name=False
    ):
        super().__init__(root, url=url, folder_in_archive=folder_in_archive, download=download)
        self._split(subset, train_speaker_ratio, train_utterance_ratio)
        self.project_fs = project_fs
        self.wav_length = wav_length
        self.return_file_name = return_file_name

    def _split(self, subset, train_speaker_ratio, train_utterance_ratio):
        """ Splits into training and testing sets """

        def _parse(name_string):
            speaker_id, chapter_id, utterance_id = name_string.split("-")
            return speaker_id, chapter_id, utterance_id

        n_total_utterance = len(self._walker)

        self._walker.sort()

        utt_per_speaker = {}
        for filename in self._walker:
            speaker_id, chapter_id, utterance_id = _parse(filename)
            if utt_per_speaker.get(speaker_id, None) is None:
                utt_per_speaker[speaker_id] = [utterance_id]
            else:
                utt_per_speaker[speaker_id].append(utterance_id)

        speakers = list(utt_per_speaker.keys())
        speakers.sort()
        num_train_speaker = int(len(speakers) * train_speaker_ratio)
        speakers = {
            "train": speakers[:num_train_speaker],
            "test": speakers[num_train_speaker:]
        }
        self.speakers = {
            "train": [int(s) for s in speakers["train"]],
            "test": [int(s) for s in speakers["test"]],
        }

        for spk in speakers["train"]:
            utt_per_speaker[spk].sort()
            num_train_utterance = int(len(utt_per_speaker[spk]) * train_utterance_ratio)
            utt_per_speaker[spk] = {
                "train": utt_per_speaker[spk][:num_train_utterance],
                "test": utt_per_speaker[spk][num_train_utterance:],
            }

        trn_walker = []
        test_walker = []
        outsiders = []
        for filename in self._walker:
            speaker_id, chapter_id, utterance_id = _parse(filename)
            if speaker_id in speakers["train"]:
                if utterance_id in utt_per_speaker[speaker_id]["train"]:
                    trn_walker.append(filename)
                else:
                    test_walker.append(filename)
            else:
                outsiders.append(outsiders)

        if subset == "train":
            self._walker = trn_walker
        elif subset == "test":
            self._walker = test_walker
        else:
            self._walker = outsiders

        order = "first" if subset == "training" else "last"
        warn(
            f"Deterministic split: {len(self._walker)} out of the {order}"
            f" {n_total_utterance} utterances are taken as {subset} set.",
            UserWarning
        )

    def __getitem__(self, n):
        if not self.return_file_name:
            waveform, sample_rate, _, speaker_id, _, _ = super().__getitem__(n)
        else:
            waveform, sample_rate, _, speaker_id, chapter_id, utt_id = super().__getitem__(n)

        n_channel, duration = waveform.shape
        if self.wav_length is None:
            pass
        elif duration > self.wav_length:
            i = torch.randint(0, duration - self.wav_length, []).long()
            waveform = waveform[:, i: i + self.wav_length]
        else:
            waveform = torch.cat(
                [
                    waveform,
                    torch.zeros(n_channel, self.wav_length - duration)
                ],
                1
            )

        # waveform = librosa.core.resample(waveform, sample_rate, self.project_fs)
        if not self.return_file_name:
            return waveform, self.speakers["train"].index(speaker_id)
        else:
            return waveform, self.speakers["train"].index(speaker_id), str(speaker_id) + "-" + str(
                chapter_id) + "-" + str(utt_id).zfill(4)


def get_dataset(dataset_name):
    if dataset_name == 'LIBRI':
        return BasicLibriSpeechDataset


def get_loaders(loader_params, dataset_config, splits_to_load, **kwargs):
    dataset_name = dataset_config['dataset_name']
    train_indices, val_indices = get_split_indices(dataset_config['root_path'],
                                                   dataset_config['speaker_labels'],
                                                   dataset_config['num_wavs_for_emb'])
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

    # train_data = ConcatDataset([train_data, val_data])
    # train_loader = DataLoader(train_data,
    #                           batch_size=loader_params['batch_size'],
    #                           num_workers=loader_params['num_workers'],
    #                           shuffle=True,
    #                           pin_memory=True)
    return train_loader, val_loader, test_loader


def load_pickle_file(path):
    with open(path, "rb") as f:
        pickled_file = pickle.load(f)
    return pickled_file


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


def create_dirs_not_exist(current_path, directories_list):
    for dir_name in directories_list:
        dir_path = os.path.join(current_path,dir_name)
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)


def load_drom_pickle(root_path,audio_type,file_name):
    file_path = os.path.join(root_path, audio_type, Path(file_name).stem)
    with open(f'{file_path}.pickle', 'rb') as f:
        return pickle.load(f)


def save_to_pickle(root_path, pickle_data, audio_type, file_name):
    file_path = os.path.join(root_path, audio_type, Path(file_name).stem)
    with open(f'{file_path}.pickle', 'wb') as f:
        pickle.dump(pickle_data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_audio_from_wav(root_path,audio_type,file_name, sample_rate=16000):
    file_path = os.path.join(root_path, audio_type, Path(file_name).name)
    signal , _ =  torchaudio.load(file_path, sample_rate)
    return signal


def save_audio_as_wav(root_path, waveform, audio_type,file_name, sample_rate=16000):
    file_path = os.path.join(root_path, audio_type, Path(file_name).name)

    # downsample_rate = 16000
    # downsample_resample = torchaudio.transforms.Resample(
    #     sample_rate, downsample_rate, resampling_method='sinc_interpolation')
    # down_sampled = downsample_resample(waveform)
    # # print(down_sampled)

    # torchaudio.save(
    #     file_path, torch.clamp(down_sampled, -1, 1), downsample_rate, precision=32)

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

# def load_emb_from_npy(root_path, np_data, audio_type, file_name):
#     file_path = os.path.join(root_path, audio_type, Path(file_name).stem)
#     np.save(f'{file_path}.npy', np_data)


def save_emb_as_npy(root_path, np_data, audio_type, file_name):
    file_path = os.path.join(root_path, audio_type, Path(file_name).stem)
    np.save(f'{file_path}.npy', np_data)


def load_from_npy(root_path, audio_type, file_name):
    file_path = os.path.join(root_path, audio_type, Path(file_name).stem)
    return np.load(f'{file_path}.npy')


def get_pretrained_speaker_embedding(root_path, audio_type, speaker_id): # model
    speaker_emb = glob.glob(f'{root_path}/{audio_type}/{speaker_id}-*')[0]
    return np.load(f'{speaker_emb}')


def write_npy(file_name, data):
    np.save(f'{file_name}.npy',data)


def create_labels_from_path(dir_path, suffix_p):

    all_spk_paths = glob.glob(f"{ROOT}/{dir_path}{suffix_p}.wav") # in case of parent dir use /*/*.wav instead
    labels = [int(os.path.basename(p).split('-')[0]) for p in all_spk_paths]

    # TODO WHAT'S BETTER? USING INT? or LABELENCODER?
    # labels_as_int = map(int,labels)
    # label_encoder = preprocessing.LabelEncoder()
    # label_encoder.fit_transform(labels)

    labels_df = pd.DataFrame(labels)

    # labels_dict = {k: v for v, k in enumerate(labels)}
    return all_spk_paths, labels_df


def create_speakers_list(root_data_path):
    # temp=os.walk(root_data_path)
    return os.listdir(root_data_path)
    # list_temp=[x[0] for x in os.walk(root_data_path)]
    # print("end create speakers")
    # all_spk_paths = glob.glob(f"{ROOT}/{dir_path}/*.wav")  # in case of parent dir use /*/*.wav instead
    # labels = [int(os.path.basename(p).split('-')[0]) for p in all_spk_paths]


def load_labels():
    data = np.load('./data/TIMIT/speaker/processed/TIMIT_labels.npy', allow_pickle=True)

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


def get_signal_and_fs_from_wav(wav_path, perturb_size=48000):
    signal, fs = torchaudio.load(wav_path)

    if signal.shape[1] >= perturb_size:
        return signal, fs

    else:
        cat_size = math.ceil(perturb_size / signal.shape[1])
        cat_size = [signal for i in range(cat_size)]
        signal = torch.cat(cat_size, 1)
        return signal, fs


def get_signal_from_wav_random(wav_path, perturb_size=48000):
    signal, _ = get_signal_and_fs_from_wav(wav_path, perturb_size=perturb_size)
    signal_len = signal.shape[1]
    start_idx = np.random.randint(0, signal_len - (perturb_size + 1))
    # print(start_idx)
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


@torch.no_grad()
def get_embeddings(model, loader, person_ids, device):
    embeddings = {id: torch.empty(0, device=device) for id in person_ids}
    for cropped_signal_batch, person_ids_batch in tqdm(loader):
        embedding = model.encode_batch(cropped_signal_batch).detach()
        for idx in person_ids_batch.unique():
            relevant_indices = torch.nonzero(person_ids_batch == idx, as_tuple=True)
            emb = embedding[relevant_indices]
            embeddings[idx.item()] = torch.cat([embeddings[idx.item()], emb], dim=0)
    final_embeddings = [person_emb.mean(dim=0).unsqueeze(0) for person_emb in embeddings.values()]
    final_embeddings = torch.cat(final_embeddings, dim=0)
    return final_embeddings


if __name__ == '__main__':
    training_path = 'data/LIBRI/d1'
    basic_libri_dataset = BasicLibriSpeechDataset(training_path)
    train_dataloader = DataLoader(basic_libri_dataset, batch_size=4, shuffle=True)
    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")

