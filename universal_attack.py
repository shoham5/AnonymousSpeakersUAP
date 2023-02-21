import pandas as pd
import os
import torch
import random

from configs.attacks_config import config_dict
from utils.model_utils import get_speaker_model
from utils.general import get_instance, save_config_to_file
from utils.data_utils import get_signal_and_fs_from_wav,get_signal_from_wav_random,\
    get_speaker_embedding, get_pretrained_speaker_embedding,save_audio_as_wav, create_dirs_not_exist,\
    save_emb_as_npy,load_from_npy,load_audio_from_wav,save_to_pickle
from utils.losses_utils import Loss
from visualization.plots import loss_plot
from torchaudio.datasets.librispeech import LIBRISPEECH
from utils.general import save_file_to_npy, save_signal_to_wav
from utils.data_utils import BasicLibriSpeechDataset
from torch.utils.data import DataLoader
import numpy as np

SIGNAL_SIZE = 48000
BATCH_SIZE = 4
NUMBER_OF_SPEAKERS = 1
NUMBER_OF_UTTERENCES = 16


class Base:
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = get_speaker_model(cfg)
        self.labels = None
        self.x = None
        self.accumulated_delta = None
        self.accumulated_delta_mean = None
        self.adv_x = None
        self.speakers_enroll = None
        self.enroll_list = self.cfg['speakers_id']
        self.eval_files_signal = None


class BaseAttack(Base):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        # TODO: check attack configuration. does it depend on attack type?
        self.cfg.attack_type = self.__class__.__name__  # for logging purposes when inheriting from another class
        self.datasetTrain = DataLoader(BasicLibriSpeechDataset(self.cfg['data_path'], suffix_p='/*/*'), batch_size=4,
                               shuffle=True)
        with torch.no_grad():
            enroll_list = self.cfg['speakers_id']
            for speaker in enroll_list:
                self.speakers_enroll[speaker] = get_speaker_embedding(self.model,
                                                     self.cfg['dataset_config']['root_path'],
                                                     speaker,
                                                     self.cfg['num_wavs_for_emb'],
                                                     self.cfg['fs'],
                                                     self.cfg['device'])

        self.y = torch.mean(torch.from_numpy(np.array([x.cpu().numpy() for x in list(self.speakers_enroll.values())])),
                            dim=0).to(self.cfg['device'])
        loss_funcs = []
        for loss_cfg, loss_func_params in zip(self.cfg['losses_config'], self.cfg['loss_func_params'].values()):
            loss_func = get_instance(loss_cfg['module_name'],
                                     loss_cfg['class_name'])(**loss_func_params)
            loss_funcs.append(loss_func)
        self.eval_files_names = list(self.speakers_enroll.keys())
        self.eval_files_names.sort()
        self.loss = Loss(self.model, loss_funcs, **self.cfg['loss_params'])
        self.cfg['attack_params']['loss_function'] = self.loss.loss_gradient
        self.attack = get_instance(self.cfg['attack_config']['module_name'],
                                   self.cfg['attack_config']['class_name'])(**self.cfg['attack_params'])

        save_config_to_file(self.cfg, self.cfg['current_dir'])
        create_dirs_not_exist(self.cfg['current_dir'], ["perturbation","perturbation_mean","embedding", "adversarial"])
        pd.DataFrame(columns=[f'iter {str(iteration)}' for iteration in range(self.cfg['attack_params']['max_iter'])]). \
            to_csv(os.path.join(self.cfg['current_dir'], 'loss_results.csv'))

    def generate_x(self):
        if self.x is not None:
            del self.x
            del self.labels
        train_features, labels = next(iter(self.datasetTrain))
        self.labels = labels
        self.x = train_features.to(self.cfg['device'])

    def generate(self, curr, total):
        x_adv = self.attack.generate(self.x, self.y, {'cur': curr, 'total': total})
        self.accumulated_delta += (self.x.cpu() - x_adv.cpu())
        self.accumulated_delta_mean += torch.mean(self.x.cpu() - x_adv.cpu(), dim=0)
        self.register_loss_values(curr)
        loss_plot(self.cfg['current_dir'])

    def multi_generate(self):
        # curr_batch = 1
        self.accumulated_delta = torch.zeros([BATCH_SIZE, SIGNAL_SIZE])  # inputs.shape
        self.accumulated_delta_mean = torch.zeros([NUMBER_OF_SPEAKERS, SIGNAL_SIZE])  # inputs.shape

        total = 16
        for i in range(total):
            self.generate_x()
            self.generate(curr=i, total=total)
        print("end multi_ generate")

    def register_loss_values(self, batch_id):
        batch_result = pd.Series([f'batch{batch_id}'] + self.attack.loss_values)
        batch_result.to_frame().T.to_csv(os.path.join(self.cfg['current_dir'], 'loss_results.csv'), mode='a',
                                         header=False, index=False)

    def save_files(self):
        save_audio_as_wav(self.cfg['current_dir'], self.accumulated_delta[0].unsqueeze(0), "perturbation", self.cfg['wav_path'])
        save_emb_as_npy(self.cfg['current_dir'], self.accumulated_delta[0].unsqueeze(0), "perturbation", self.cfg['wav_path'])
        save_emb_as_npy(self.cfg['current_dir'], self.accumulated_delta_mean, "perturbation_mean", self.cfg['wav_path'])
        save_audio_as_wav(self.cfg['current_dir'], self.accumulated_delta_mean, "perturbation_mean", self.cfg['wav_path'])
        save_to_pickle(self.cfg['current_dir'], self.speakers_enroll, "embedding", self.cfg['wav_path'])

    def get_current_dir(self):
        return self.cfg['current_dir']


class BaseEval(Base):
    def __init__(self, cfg, previous_dir) -> None:
        super().__init__(cfg)
        self.prev_files_loc = previous_dir
        self.similarity_values = []
        self.sims = get_instance(self.cfg['similarity_config']['module_name'],
                                 self.cfg['similarity_config']['class_name'])(**self.cfg['similarity_params'])
        self.datasetEval = DataLoader(BasicLibriSpeechDataset(self.cfg['eval_path'], suffix_p='/*/*'), batch_size=4,
                                   shuffle=True)

        with torch.no_grad():
            self.y = get_pretrained_speaker_embedding(self.prev_files_loc.split('/')[0], "embedding",
                                                      self.cfg['speaker_id'])
            self.x = get_signal_from_wav_random(os.path.join(self.cfg['wav_path'])).unsqueeze(0)

            print("next is eval self.eval file")
            self.eval = get_signal_from_wav_random(os.path.join(self.cfg['eval_wav_path'])).unsqueeze(0)  # other file from sam speaker
            self.eval2 = get_signal_from_wav_random(os.path.join(self.cfg['eval_wav_path'])).unsqueeze(
                0)  # other file from sam speaker
            self.perturbation = load_from_npy(self.prev_files_loc, "perturbation", self.cfg['wav_path'])
            self.x_adv = load_audio_from_wav(self.prev_files_loc, "adversarial", self.cfg[
                'wav_path'])  # get 32k vector instead 48k probably because sox_io compressing
            self.x_adv_npy = load_from_npy(self.prev_files_loc, "adversarial", self.cfg['wav_path'])

            self.y_mean = get_pretrained_speaker_embedding(self.prev_files_loc.split('/')[0], "embedding",
                                                      self.cfg['speaker_id'])
            self.perturbation = load_from_npy(self.prev_files_loc, "perturbation", self.cfg['wav_path'])
            self.x_adv = load_audio_from_wav(self.prev_files_loc, "adversarial", self.cfg[
                'wav_path'])  # get 32k vector instead 48k probably because sox_io compressing
            self.x_adv_npy = load_from_npy(self.prev_files_loc, "adversarial", self.cfg['wav_path'])
            self.mean_emb = None
        df_cols = ['eval_adv_single', 'eval_adv_mean', 'x_adv_emb_y']
        pd.DataFrame(columns=df_cols).to_csv(os.path.join(self.prev_files_loc, 'sims_results.csv'))

    def register_similarity_values(self, batch_id):
        batch_result = pd.Series([f'{batch_id}'] + self.similarity_values[batch_id]) # self.similarity_values)
        batch_result.to_frame().T.to_csv(os.path.join(self.prev_files_loc, 'sims_results.csv'), mode='a',
                                         header=False, index=False)

    def generate_eval_data(self):
        if self.eval2 is not None:
            del self.eval2
            # del self.labels
            del self.eval_files_signal
        eval_features, labels = next(iter(self.datasetEval))
        self.eval_files_signal = {spk: signal for spk, signal in zip(eval_features, labels)}
        self.eval2 = eval_features.to(self.cfg['device'])

    def evaluate(self):
        batch_id = 0

        eval_adv_single = torch.broadcast_to(self.eval2, [BATCH_SIZE, SIGNAL_SIZE]) + self.accumulated_delta
        eval_adv_single2 = torch.broadcast_to(self.eval, [BATCH_SIZE, SIGNAL_SIZE]) + self.accumulated_delta
        eval_adv_mean = torch.broadcast_to(self.eval, [NUMBER_OF_SPEAKERS, SIGNAL_SIZE]) + self.accumulated_delta_mean

        similarity_eval_ = self.sims(self.model.encode_batch(eval_adv_single).cpu(), self.y_mean.cpu())
        similarity_eval_2 = self.sims(self.model.encode_batch(eval_adv_single2).cpu(), self.y_mean.cpu())
        similarity_mean_ = self.sims(self.model.encode_batch(eval_adv_mean).cpu(), self.y_mean.cpu())
        similarity_clear = self.sims(self.model.encode_batch(self.eval).cpu(), self.y_mean.cpu())

        sims_results = {'eval_adv_single': similarity_eval_, 'eval_adv_mean': similarity_mean_}
        self.similarity_values.append([str(round(similarity_eval_.item(), 5))
                                          , str(round(similarity_eval_.item(), 5))])
        self.register_similarity_values(batch_id)
        print(sims_results)

        # speakers_enroll_list = self.speakers_enroll.keys()
        for speaker_id in self.speakers_enroll.keys():
            self.eval_file(speaker_id)

    def eval_file(self,eval_speaker):
        print("eval_file: ")
        spk_emb_pt = self.speakers_enroll[eval_speaker]
        eval_signal = self.eval_files_signal[eval_speaker]
        eval_adv_single = torch.broadcast_to(eval_signal, [BATCH_SIZE, SIGNAL_SIZE]) + self.accumulated_delta
        eval_adv_mean = torch.broadcast_to(eval_signal, [NUMBER_OF_SPEAKERS, SIGNAL_SIZE]) + self.accumulated_delta_mean


        print("SNR eval:",calculator_snr_direct(eval_signal, eval_adv_single[0].unsqueeze(0)))
        save_audio_as_wav(self.cfg['current_dir'], eval_adv_single[0].unsqueeze(0), "adversarial", f'{eval_speaker}.wav')

        similarity_eval_single = self.sims(self.model.encode_batch(eval_adv_single).cpu(), spk_emb_pt.cpu())
        similarity_mean_ = self.sims(self.model.encode_batch(eval_adv_mean).cpu(), spk_emb_pt.cpu())

        sims_results = {'eval_adv_single': similarity_eval_single, 'eval_adv_mean': similarity_mean_,
                        'eval_adv_input_mean': similarity_mean_input}

        self.similarity_values.append([str(round(similarity_eval_.item(), 5))
                                          , str(round(similarity_eval_.item(), 5))])
        print(sims_results)
        sims_results = {'eval_emb_y': similarity_eval_emb, 'x_segment_emb_y': similarity_x_segment_emb,
                        'x_adv_emb_y': similarity_x_adv_emb}
        print(sims_results)
        print("%%%%%%")


class SingleAttackSingleFile:
    pass


class SingleEvalSingleFile:
    pass


class SingleWAVAttackEval:
    def __init__(self, cfg, previous_dir) -> None:
        super().__init__()
        self.cfg = cfg
        self.cfg.attack_type = self.__class__.__name__  # for logging purposes when inheriting from another class
        self.model = get_speaker_model(cfg)
        self.prev_files_loc = previous_dir
        self.similarity_values = []
        self.sims = get_instance(self.cfg['similarity_config_one']['module_name'],
                                 self.cfg['similarity_config_one']['class_name'])(**self.cfg['similarity_params'])

        with torch.no_grad():
            self.x = get_signal_from_wav_random(os.path.join(self.cfg['wav_path']))#.unsqueeze(0)
            print("next is eval self.eval file")
            self.eval = get_signal_from_wav_random(os.path.join(self.cfg['eval_wav_path']))#.unsqueeze(0) # other file from sam speaker
            self.y = get_pretrained_speaker_embedding(self.prev_files_loc.split('/')[0], "embedding", self.cfg['speaker_id'])
            self.perturbation = load_from_npy(self.prev_files_loc, "perturbation", self.cfg['wav_path'])
            self.x_adv = load_audio_from_wav(self.prev_files_loc, "adversarial", self.cfg['wav_path']) # get 32k vector instead 48k probably because sox_io compressing
            self.x_adv_npy = load_from_npy(self.prev_files_loc, "adversarial", self.cfg['wav_path'])

        df_cols = ['eval_emb_y', 'x_segment_emb_y', 'x_adv_emb_y']
        pd.DataFrame(columns=df_cols).to_csv(os.path.join(self.prev_files_loc, 'sims_results.csv'))


class SingleWAVAttack:
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.cfg.attack_type = self.__class__.__name__  # for logging purposes when inheriting from another class
        self.model = get_speaker_model(cfg)

        signal, _ = get_signal_and_fs_from_wav(os.path.join(self.cfg['wav_path']))
        signal_len = signal.shape[1]
        start_idx = random.randint(0, signal_len - (self.cfg['fs'] * 3 + 1))
        print(start_idx)
        cropped_signal = signal[0][start_idx: start_idx + (self.cfg['fs'] * 3)]
        self.x = cropped_signal.to(self.cfg['device'])
        self.perturbation = torch.zeros([1, (self.cfg['fs'] * 3)])
        self.x_adv = None
        with torch.no_grad():
            # self.y = self.model.encode_batch(cropped_signal)
            # self.speaker_emb
            self.y = get_speaker_embedding(self.model,
                                                     self.cfg['dataset_config']['root_path'],
                                                     self.cfg['speaker_id'],
                                                     self.cfg['num_wavs_for_emb'],
                                                     self.cfg['fs'],
                                                     self.cfg['device']).unsqueeze(0)

        # Use different distance metrics
        loss_funcs = []
        for loss_cfg, loss_func_params in zip(self.cfg['losses_config'], self.cfg['loss_func_params'].values()):
            loss_func = get_instance(loss_cfg['module_name'],
                                     loss_cfg['class_name'])(**loss_func_params)
            loss_funcs.append(loss_func)

        self.loss = Loss(self.model, loss_funcs, **self.cfg['loss_params'])
        self.cfg['attack_params']['loss_function'] = self.loss.loss_gradient
        self.attack = get_instance(self.cfg['attack_config']['module_name'],
                                   self.cfg['attack_config']['class_name'])(**self.cfg['attack_params'])

        save_config_to_file(self.cfg, self.cfg['current_dir'])

        pd.DataFrame(columns=[f'iter {str(iteration)}' for iteration in range(self.cfg['attack_params']['max_iter'])]).\
            to_csv(os.path.join(self.cfg['current_dir'], 'loss_results.csv'))

        create_dirs_not_exist(self.cfg['current_dir'], ["perturbation", "adversarial", "embedding"])
        print(self.cfg['current_dir'])

    def generate(self):
        x_adv = self.attack.generate(self.x, self.y, {'cur': 1, 'total': 1})
        self.perturbation += x_adv.cpu() - self.x.cpu()
        self.x_adv = x_adv.cpu().unsqueeze(0)
        self.register_loss_values(batch_id=0)
        loss_plot(self.cfg['current_dir'])

    def register_loss_values(self, batch_id):
        batch_result = pd.Series([f'batch{batch_id}'] + self.attack.loss_values)
        batch_result.to_frame().T.to_csv(os.path.join(self.cfg['current_dir'], 'loss_results.csv'), mode='a', header=False, index=False)



def check_single_wav():
    config_type = 'SingleWAV'
    cfg = config_dict[config_type]()
    attack = SingleWAVAttack(cfg)
    attack.generate()
    attack.save_files()

    previous_files_place = attack.get_current_dir()

    config_type_eval = 'SingleWavEval'
    cfg_eval = config_dict[config_type_eval]()
    single_wav_eval = SingleWAVAttackEval(cfg_eval, previous_files_place)
    single_wav_eval.evaluate()


def main():
    # config_type = 'SingleWAV'
    # cfg = config_dict[config_type]()
    # attack = SingleWAVAttack(cfg)
    # attack.generate()

    # check_different_segment()
    check_single_wav()


if __name__ == '__main__':
    main()
    print("end")
