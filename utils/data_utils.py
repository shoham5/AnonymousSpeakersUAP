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
from visualization.plots import plot_waveform
from utils.general import calculator_snr_direct, calculate_snr_github_direct
import matplotlib.pyplot as plt
import datetime
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
global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

URL = "train-clean-100"
FOLDER_IN_ARCHIVE = "LibriSpeech"

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

# TODO: send from config
TRAIN_SPLIT = 0.8



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
        wavs = fnmatch.filter(files_in_folder, '*.flac')
        return wavs


#TODO: changed from .wav to .flac ('**/*.flac') to support libri-train-100, change to
def get_nested_dataset_files(img_dir, labels):
    files_in_folder = [glob.glob(os.path.join(img_dir, lab, '**/*.flac'), recursive=True) for lab in labels]
    return files_in_folder

# TODO: train index depend on num_of_samples which depend on samples to create embbedings from. changed to fit 70% train
def get_split_indices(img_dir, labels, num_of_samples,split_rate=0.3):
    dataset_nested_files = get_nested_dataset_files(img_dir, labels)

    nested_indices = [np.array(range(len(arr))) for i, arr in enumerate(dataset_nested_files)]
    nested_indices_continuous = [nested_indices[0]]
    # min_samples_in = min(map(len, nested_indices))
    for i, arr in enumerate(nested_indices[1:]):
        nested_indices_continuous.append(arr + nested_indices_continuous[i][-1] + 1)

    train_indices = np.concatenate([np.random.choice(arr_idx, size=math.floor(len(arr_idx) * split_rate), replace=False) for arr_idx in nested_indices_continuous])

    val_indices = list(set(list(range(nested_indices_continuous[-1][-1]))) - set(train_indices))
    small_val_indices = list(np.random.choice(val_indices, size=math.floor(len(val_indices) * split_rate),replace=False))
    # train_indices_ravel = np.array([np.random.choice(arr_idx, size=num_of_samples, replace=False) for arr_idx in
    #                           nested_indices_continuous]).ravel()
    # val_indices = list(set(list(range(nested_indices_continuous[-1][-1]))) - set(train_indices_ravel))

    return train_indices, small_val_indices # val_indices


@torch.no_grad()
def load_mask(config, mask_path, device):
    transform = transforms.Compose([transforms.Resize(config.patch_size), transforms.ToTensor()])
    img = Image.open(mask_path)
    img_t = transform(img).unsqueeze(0).to(device)
    return img_t

def apply_perturbation(src_file, perturb, device,eps=1.0):
    # perturb = perturb.to(device)
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
        final_embeddings = [person_emb.mean(dim=0).unsqueeze(0) for person_emb in person_embeddings.values()]
        final_embeddings = torch.cat(final_embeddings,dim=0)
        embeddings_by_embedder[embedder_name] = final_embeddings
    return embeddings_by_embedder


# @torch.no_grad()
# def get_person_embedding_src(config, loader, celeb_lab, embedders, device, include_others=False):
#     print('Calculating persons embeddings {}...'.format('with mask' if include_others else 'without mask'), flush=True)
#     embeddings_by_embedder = {}
#     for embedder_name, embedder in embedders.items():
#         person_embeddings = {i: torch.empty(0, device=device) for i in range(len(celeb_lab))}
#         masks_path = [config.blue_mask_path, config.black_mask_path, config.white_mask_path]
#         for img_batch, person_indices in tqdm(loader):
#             img_batch = img_batch.to(device)
#             if include_others:
#                 mask_path = masks_path[random.randint(0, 2)]
#                 mask_t = load_mask(config, mask_path, device)
#                 applied_batch = apply_mask(location_extractor, fxz_projector, img_batch, mask_t[:, :3], mask_t[:, 3], is_3d=True)
#                 img_batch = torch.cat([img_batch, applied_batch], dim=0)
#                 person_indices = person_indices.repeat(2)
#             embedding = embedder(img_batch)
#             for idx in person_indices.unique():
#                 relevant_indices = torch.nonzero(person_indices == idx, as_tuple=True)
#                 emb = embedding[relevant_indices]
#                 person_embeddings[idx.item()] = torch.cat([person_embeddings[idx.item()], emb], dim=0)
#         final_embeddings = [person_emb.mean(dim=0).unsqueeze(0) for person_emb in person_embeddings.values()]
#         final_embeddings = torch.stack(final_embeddings)
#         embeddings_by_embedder[embedder_name] = final_embeddings
#     return embeddings_by_embedder



def get_dataset(dataset_name):
    if dataset_name == 'LIBRI' or dataset_name == 'LIBRI-TEST' or dataset_name == 'LIBRIALL':
        return BasicLibriSpeechDataset
    elif dataset_name == 'VOX1':
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


def get_test_loaders(config, dataset_names):
    emb_loaders = {}
    test_loaders = {}
    for dataset_name in dataset_names:
        test_indices, emb_indices = get_split_indices(config.test_img_dir[dataset_name]['root_path'],
                                                      config.test_celeb_lab[dataset_name],
                                                      config.test_num_of_images_for_emb)
        dataset = get_dataset(dataset_name)
        emb_dataset = dataset(root_path=config.test_img_dir[dataset_name]['root_path'],
                                     speaker_labels_mapper=config.test_celeb_lab_mapper[dataset_name],
                                     indices=emb_indices,
                              )
        emb_loader = DataLoader(emb_dataset, batch_size=config.test_batch_size)
        emb_loaders[dataset_name] = emb_loader

        # self.speaker_embeddings = get_embeddings(self.model,
        #                                          self.train_loader,
        #                                          self.cfg['dataset_config']['speaker_labels_mapper'].keys(),
        #                                          self.cfg['device'])



        test_dataset = dataset(root_path=config.test_img_dir[dataset_name]['root_path'],
                                      speaker_labels_mapper=config.test_celeb_lab_mapper[dataset_name],
                                      indices=test_indices
                                      )
        test_loader = DataLoader(test_dataset, batch_size=config.test_batch_size)
        test_loaders[dataset_name] = test_loader

    return emb_loaders, test_loaders


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


def save_audio_as_wav(root_path, waveform, audio_type,file_name, sample_rate=16000, src_voice=""):
    file_path = os.path.join(root_path, audio_type, Path(file_name).name)
    if '.flac' in file_name:
        file_path = os.path.join(root_path, audio_type, f'{Path(file_name).stem}{src_voice}.wav')


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
    # src_signal_size = signal.shape
    # print("signal.shape[1]: ",signal.shape[1])
    if signal.shape[1] > perturb_size +1:
        return signal, fs

    else:
        c_size = math.ceil(perturb_size / signal.shape[1]) +1
        # TODO: adding +1 to handle ValueError: high <= 0
        cat_size = [signal for i in range(c_size)]
        signal = torch.cat(cat_size, 1)
    # if signal.shape[1]==48000:
    #     print(f"signal len: {signal.shape} _ wav_path: {wav_path} _original_size:{src_signal_size} ")
    return signal, fs

# add 5 for margin. handle signal with size 48000 exactly
def get_signal_from_wav_random(wav_path, perturb_size=48000):
    signal, _ = get_signal_and_fs_from_wav(wav_path, perturb_size=perturb_size)
    signal_len = signal.shape[1]
    if (signal_len - (perturb_size + 1)) <= 0:
        print("temp <=0: ",wav_path )
    start_idx = np.random.randint(low=0, high=(signal_len - (perturb_size + 1)), dtype=np.int64())
    cropped_signal = signal[0][start_idx: start_idx + perturb_size]
    # print(start_idx)
    # print(f'high:_ {(signal_len - (perturb_size + 10))}_signal_len{signal_len}')

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
        cropped_signal_batch = cropped_signal_batch.to(device)
        embedding = model.encode_batch(cropped_signal_batch).detach()
        for idx in person_ids_batch.unique():
            relevant_indices = torch.nonzero(person_ids_batch == idx, as_tuple=True)
            emb = embedding[relevant_indices]
            embeddings[idx.item()] = torch.cat([embeddings[idx.item()], emb], dim=0)
    final_embeddings = [person_emb.mean(dim=0).unsqueeze(0) for person_emb in embeddings.values()]
    final_embeddings = torch.cat(final_embeddings, dim=0)
    return final_embeddings


def add_noise_awgn():
    # based on https://stackoverflow.com/questions/14058340/adding-noise-to-a-signal-in-python
    # max = 0.49
    file_path_men = "/sise/home/hanina/speaker_attack/data/libri_train-clean-100/8063/8063-274116-0029.flac"
    # max = 0.43
    file_path_men2 = "/sise/home/hanina/speaker_attack/data/libri_train-clean-100/1355/1355-39947-0018.flac"
    # max = 0.365
    file_path_w = "/sise/home/hanina/speaker_attack/data/libri_train-clean-100/8465/8465-246943-0012.flac"
    # max = 0.389
    file_path_w2 = "/sise/home/hanina/speaker_attack/data/libri_train-clean-100/32/32-21631-0008.flac"
    x_volts = get_signal_from_wav_random(file_path_men)
    # src_signal = get_signal_from_wav_random("../data/libri_train-clean-100/200/200-124140-0012.flac")

    print("x_volts = " + str(np.mean(np.abs(x_volts.unsqueeze(0)).numpy() ** 2)))
    plt.plot(x_volts)
    print("wait")
    plot_waveform(x_volts.unsqueeze(0), 16000)
    x_watts = x_volts ** 2

    # Adding noise using target SNR

    # Set a target SNR
    target_snr_db = 20

    # Calculate signal power and convert to dB
    # sig_avg_watts = torch.mean(x_watts)
    sig_avg_watts = np.mean(x_watts.numpy())
    sig_avg_db = 10 * np.log10(sig_avg_watts)

    # Calculate noise according to [2] then convert to watts
    noise_avg_db = sig_avg_db - target_snr_db

    noise_avg_watts = 10 ** (noise_avg_db / 10)
    # Generate an sample of white noise
    mean_noise = 0
    noise_volts = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(x_watts))
    plt.plot(noise_volts)
    print("noise_volts = " + str(np.mean(np.abs(torch.from_numpy(noise_volts).unsqueeze(0)).numpy() ** 2)))
    # Noise up the original signal
    y_volts = x_volts + noise_volts

    snr_calc = calculator_snr_direct(x_volts.unsqueeze(0), (y_volts).unsqueeze(0))
    print("y_volts = " + str(np.mean(np.abs(y_volts.unsqueeze(0)).numpy() ** 2)))
    plot_waveform(y_volts.unsqueeze(0), 16000)

    save_emb_as_npy("..",noise_volts,"output_files","woman_perturb")
    save_audio_as_wav("..",(torch.from_numpy(noise_volts)).unsqueeze(0).float(),"output_files","woman_perturb.wav")
    # plt.savefig(os.path.join("..","output_files","men_perturb.png"))

def create_gt_by_eps_perturbation(eps=2.):
    file_path_men = "/sise/home/hanina/speaker_attack/data/libri_train-clean-100/8063/8063-274116-0029.flac"
    # file_path_w2 = "/sise/home/hanina/speaker_attack/data/libri_train-clean-100/32/32-21631-0008.flac"
    src_signal = get_signal_from_wav_random(file_path_men)
    adv_perturb_cosim_ep100 = torch.from_numpy(load_from_npy(
        os.path.join('..', 'data', 'uap_perturbation'), 'cosim', '100ep_100spk'))

    # adv_perturb_cosim_ep100_gt_eps = adv_perturb_cosim_ep100 * eps

    adv_signal_reg = src_signal + adv_perturb_cosim_ep100
    adv_signal_eps = src_signal + adv_perturb_cosim_ep100 * eps

    # save_emb_as_npy("..", uniform_noise, "output_files", "gt_eps_man_perturb")
    save_audio_as_wav("..", adv_signal_reg, "output_files", "reg_man_perturb.wav")
    save_audio_as_wav("..", adv_signal_eps, "output_files", "eps_man_perturb.wav")

    snr_calc_reg = calculator_snr_direct(src_signal.unsqueeze(0), adv_signal_reg.unsqueeze(0))
    snr_calc_eps = calculator_snr_direct(src_signal.unsqueeze(0), adv_signal_eps.unsqueeze(0))
    print("snr_calc: ",snr_calc_reg)
    print("snr_calc: ", snr_calc_eps)
    plot_waveform(adv_signal_reg, 16000)
    plot_waveform(adv_signal_eps, 16000)
    print("finish eps perturbation")


def create_uniform_perturbation():
    file_path_men = "/sise/home/hanina/speaker_attack/data/libri_train-clean-100/8063/8063-274116-0029.flac"
    # file_path_w2 = "/sise/home/hanina/speaker_attack/data/libri_train-clean-100/32/32-21631-0008.flac"
    src_signal = get_signal_from_wav_random(file_path_men)

    uniform_noise = 0.015 * np.random.uniform(low=-1.0, high=1.0, size=(48000))
    noise = (torch.from_numpy(uniform_noise)).unsqueeze(0)
    save_emb_as_npy("..", uniform_noise, "output_files", "uniform_man_perturb")
    save_audio_as_wav("..", noise.float(), "output_files", "uniform_man_perturb.wav")
    adversarial_signal = src_signal + noise
    snr_calc = calculator_snr_direct(src_signal.unsqueeze(0), adversarial_signal.unsqueeze(0))
    print("snr_calc: ",snr_calc)
    plot_waveform(adversarial_signal, 16000)





# def apply_uap_perturbation(src_file, perturb, eps=1):
#     adv_signal = src_file + eps * perturb
#     return adv_signal


def add_noise_awgn_iterative(x_volts):
    # plot_waveform(x_volts, 16000)
    x_volts = x_volts.squeeze()
    x_watts = x_volts ** 2

    # Adding noise using target SNR

    # Set a target SNR
    target_snr_db = 20.25

    # Calculate signal power and convert to dB
    # sig_avg_watts = torch.mean(x_watts)
    sig_avg_watts = np.mean(x_watts.numpy())
    sig_avg_db = 10 * np.log10(sig_avg_watts)

    # Calculate noise according to [2] then convert to watts
    noise_avg_db = sig_avg_db - target_snr_db

    noise_avg_watts = 10 ** (noise_avg_db / 10)
    # Generate an sample of white noise
    mean_noise = 0
    noise_volts = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(x_watts))
    # Noise up the original signal
    y_volts = x_volts + noise_volts

    snr_calc = calculator_snr_direct(x_volts.unsqueeze(0), (y_volts).unsqueeze(0))
    print("snr_calc in noise iterative : ", snr_calc)
    return noise_volts

def create_mean_without_signal(is_uniform=True ):
    # plot witout signals at all
    if is_uniform:
        temp_uniform =  0.009* np.random.uniform(low=-1.0, high=1.0, size=(48000))#np.array([np.random.uniform(low=-1.0, high=1.0, size=(48000)) for x in range(10000) ])
        perturb_mean = np.mean(temp_uniform, axis=0)
        save_perturbation(torch.from_numpy(temp_uniform).unsqueeze(0), "uniform_perturb")  # snr 20.65032435243402
        print("noise_power =" + str(np.mean(np.abs(temp_uniform) ** 2))) # 3.333814736324027e-05


    else:
        temp_normal = np.random.normal(0, np.sqrt(2)/250, 48000)#np.array([np.random.normal(0, np.sqrt(2)/2, 48000) for x in range(1000)])
        perturb_mean = np.mean(temp_normal, axis=0)
        save_perturbation(torch.from_numpy(temp_normal).unsqueeze(0), "normal_perturb") #
        print("noise_power =" + str(np.mean(np.abs(temp_normal) ** 2))) #  snr 20.840624742979315
    print("wait")
    return perturb_mean



def create_mean_uniform_perturbtion(uniform_perturb=True):

    file_name = "uniform_perturb_test_mean" if uniform_perturb else "normal_perturb_test_mean"
    num_eval_spks = 1
    eps = 1
    output_adversarial = "../data/libri_adversarial/"
    eval_path = "../data/libri_train-clean-100/"
    all_spkrs = os.listdir(eval_path)

    rndr_spkers = np.random.choice(all_spkrs, num_eval_spks)

    for i, spk in enumerate(rndr_spkers):
        signals_to_eval = np.random.choice(os.listdir(os.path.join(eval_path, spk)), num_eval_spks)
        for j, signal_eval in enumerate(signals_to_eval):
            file_path = os.path.join(eval_path, spk, signal_eval)
            src_signal = get_signal_from_wav_random(file_path).unsqueeze(0)
            # if uniform_perturb:
            #     noise = np.random.uniform(low=-1.0, high=1.0, size=(48000))
            # else:
            #     noise = add_noise_awgn_iterative(src_signal)
            # noise = (torch.from_numpy(noise)).unsqueeze(0)
            # st_noise += noise
            noise = create_mean_without_signal()
            adversarial_signal = src_signal + noise
            snr_calc = calculator_snr_direct(src_signal.unsqueeze(0), adversarial_signal.unsqueeze(0))
            print("snr_calc: ", snr_calc)

    # rndr_spkers = np.random.choice(all_spkrs, num_eval_spks)
    # for spk in rndr_spkers:
    #     signals_to_eval = np.random.choice(os.listdir(os.path.join(eval_path, spk)), num_eval_spks)
    #     curr_spk_snr = []
    #     for signal_eval in signals_to_eval:
    #         file_path = os.path.join(eval_path, spk, signal_eval)
    #         src_signal = get_signal_from_wav_random(file_path).unsqueeze(0)
    #         adversarial_signal = src_signal + st_noise
    #         snr_calc = calculator_snr_direct(src_signal.unsqueeze(0), adversarial_signal.unsqueeze(0))
    #         print("snr_calc: ", snr_calc)
    #
    # save_perturbation(st_noise, file_name)
    print("finish create_mean_uniform_perturbtion ")


def create_random_perturbation(uniform_perturb=True):
    # the sum of two independent normally distributed random variables is normal,
    # Rescaling the Irwin–Hall distribution provides the exact distribution of the random variates being generated

    file_name = "uniform_perturb_test" if uniform_perturb else "normal_perturb_test"
    is_first = False if uniform_perturb else True
    num_eval_spks = 10
    eps = 1
    output_adversarial = "../data/libri_adversarial/"
    eval_path = "../data/libri_train-clean-100/"
    all_spkrs = os.listdir(eval_path)
    # exper = "../data/uap_perturbation/"
    # training_path = 'data/LIBRI/d1'

    # uap_perturb = load_from_npy(exper, "cosim", "100ep_100spk")

    snr_values_mean = []
    snr_dict = {}
    eps = 0
    st_noise = 0
    rndr_spkers = np.random.choice(all_spkrs, num_eval_spks)

    for spk in rndr_spkers:
        signals_to_eval = np.random.choice(os.listdir(os.path.join(eval_path, spk)), num_eval_spks)
        curr_spk_snr = []
        for signal_eval in signals_to_eval:
            file_path = os.path.join(eval_path, spk, signal_eval)
            src_signal = get_signal_from_wav_random(file_path).unsqueeze(0)
            if uniform_perturb:
                noise = np.random.uniform(low=-1.0, high=1.0, size=(48000))
            else:
                noise = add_noise_awgn_iterative(src_signal)
            noise = (torch.from_numpy(noise)).unsqueeze(0)
            print("noise = " + str(np.mean(np.abs(noise).numpy() ** 2)))
            st_noise += noise
            print("st_noise = " + str(np.mean(np.abs(st_noise).numpy() ** 2)))
            for i in range(1,100000):
                # if is_first:
                #     adversarial_signal = src_signal + st_noise
                #     snr_calc = calculator_snr_direct(src_signal.unsqueeze(0), adversarial_signal.unsqueeze(0))
                #     is_first = False
                #     break
                eps = 1/i
                adversarial_signal = src_signal + eps * st_noise
                snr_calc = calculator_snr_direct(src_signal.unsqueeze(0), adversarial_signal.unsqueeze(0))

                if snr_calc > 20 and snr_calc < 30:
                    print("snr_calc generate: ", snr_calc)
                    st_noise = eps * st_noise
                    print("st_noise = " + str(np.mean(np.abs(st_noise).numpy() ** 2)))
                    break

    rndr_spkers = np.random.choice(all_spkrs, num_eval_spks)
    for spk in rndr_spkers:
        signals_to_eval = np.random.choice(os.listdir(os.path.join(eval_path, spk)), num_eval_spks)
        curr_spk_snr = []
        for signal_eval in signals_to_eval:
            file_path = os.path.join(eval_path, spk, signal_eval)
            src_signal = get_signal_from_wav_random(file_path).unsqueeze(0)
            adversarial_signal = src_signal + st_noise
            snr_calc = calculator_snr_direct(src_signal.unsqueeze(0), adversarial_signal.unsqueeze(0))
            print("snr_calc: ",snr_calc)

    save_perturbation(st_noise,file_name)


def save_perturbation(waveform, name):
    save_emb_as_npy("..", waveform, "output_files", name)
    save_audio_as_wav("..", waveform.float(), "output_files", f'{name}.wav')

    plot_waveform(waveform, 16000)






def create_adversarial_files_using_perturbation(save_files = True):
    # create adversarial perturbation based on prev perturb
    # expr_50_50 = "/sise/home/hanina/speaker_attack/experiments/February/27-02-2023_174121_664051/"
    num_eval_spks = 3
    eps = 1
    output_adversarial = "../data/19_5_2023_results/libri_adversarial/"
    output_adversarial = output_adversarial + datetime.datetime.now().strftime("%H-%M_%d-%m-%Y")

    if not os.path.exists(output_adversarial):
        os.makedirs(output_adversarial)

    if not os.path.exists(os.path.join(output_adversarial , "adversarial")):
        os.makedirs(os.path.join(output_adversarial , "adversarial"))


    exper = "../data/19_5_2023_results/uap_perturbation_ecapa/"
    # ecapa, xvector, wavlm
    training_path = 'data/LIBRI/d1'
    # eval_path = "../data/libri_train-clean-100/"
    eval_path = "../data/libri-test-clean"
    uap_perturb = load_from_npy(exper,"old", "uap_ep100_spk100")
    all_spkrs = os.listdir(eval_path)
    snr_values_mean = []
    snr_dict = {}
    rndr_spkers =np.random.choice(all_spkrs,num_eval_spks)
    for spk in rndr_spkers:
        snr_dict[spk] = []
        np.random.seed(42) #
        signals_to_eval = np.random.choice(os.listdir(os.path.join(eval_path,spk)), num_eval_spks)
        curr_spk_snr = []
        for signal_eval in signals_to_eval:
            file_path = os.path.join(eval_path, spk, signal_eval)
            signal = get_signal_from_wav_random(file_path).unsqueeze(0)
            adv_signal = apply_perturbation(src_file=signal, eps=eps, perturb=uap_perturb,device=device)
            snr_calc = calculate_snr_github_direct(signal.unsqueeze(0).cpu().detach().numpy(), adv_signal.unsqueeze(0).cpu().detach().numpy())
            # (adv_cropped_signal_batch.cpu().detach().numpy(),
            # cropped_signal_batch.cpu().detach().numpy())
            # snr_calc = calculator_snr_direct(signal.unsqueeze(0), adv_signal.unsqueeze(0))

            curr_spk_snr.append(round(snr_calc, 5))
            snr_dict[spk].append(round(snr_calc, 5))
            if save_files:
                save_audio_as_wav(output_adversarial, adv_signal, "adversarial",
                                  file_path)
                save_audio_as_wav(output_adversarial, signal, "adversarial",
                                  f'src_{file_path}',src_voice='_src')

        snr_values_mean.append(np.array(curr_spk_snr).mean())
    print("wait create adversarial perturbation ")
    snr_dict = {'speakers': snr_values_mean}
    df = pd.DataFrame(snr_dict)
    df.to_csv(os.path.join(output_adversarial, "adversarial",'snr_mean.csv'))



    # src_signal = get_signal_from_wav_random("../data/libri_train-clean-100/200/200-124140-0012.flac")

if __name__ == '__main__':
    import torchaudio

    create_adversarial_files_using_perturbation()
    # create_adversarial_files_using_perturbation()
    # add_noise_awgn()
    from speechbrain.pretrained import EncoderClassifier

    # classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb",
    #                                             savedir="pretrained_models/spkrec-xvect-voxceleb")
    # signal = get_signal_from_wav_random("../data/libri_train-clean-100/200/200-124140-0012.flac")
    # embeddings = classifier.encode_batch(signal)
    # masks_path = os.path.join('.', 'data', 'uap_perturbation')  # change to perturb
    # create_gt_by_eps_perturbation()
    # create_adversarial_files_using_perturbation()

    # print("embeddings: ", embeddings.shape)
    # add_noise_awgn()
    # create_uniform_perturbation()
    # add_noise_awgn()
    # create_random_perturbation(uniform_perturb=False)

    # create_mean_uniform_perturbtion()

    # create_uniform_perturbation()
    # create_mean_uniform_perturbtion()
    # create_adversarial_files_using_perturbation()

    print("wait")
    # basic_libri_dataset = BasicLibriSpeechDataset(training_path)
    # train_dataloader = DataLoader(basic_libri_dataset, batch_size=4, shuffle=True)
    # train_features, train_labels = next(iter(train_dataloader))
    # print(f"Feature batch shape: {train_features.size()}")
    # print(f"Labels batch shape: {train_labels.size()}")
    print("finish")