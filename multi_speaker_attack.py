import pandas as pd
import os
import torch
import random
import numpy as np
from utils.general import calculator_snr_direct

from configs.attacks_config import config_dict
from utils.model_utils import get_speaker_model
from utils.general import get_instance, save_config_to_file
from utils.data_utils import get_signal_and_fs_from_wav,get_signal_from_wav_random, get_speaker_embedding
from utils.losses_utils import Loss
from visualization.plots import loss_plot
from torchaudio.datasets.librispeech import LIBRISPEECH
from utils.general import save_file_to_npy, save_signal_to_wav
from utils.data_utils import get_signal_and_fs_from_wav,get_signal_from_wav_random,\
    get_speaker_embedding, get_pretrained_speaker_embedding,save_audio_as_wav, create_dirs_not_exist,\
    save_emb_as_npy,load_from_npy,load_audio_from_wav
from utils.data_utils import BasicLibriSpeechDataset
from torch.utils.data import DataLoader

# TODO Change constant var to generic,using input size. input size define only in generate_x()- A PROBLEM.
NUMBER_OF_SPEAKERS = 3
BATCH_SIZE = 16 #4, 16, 32
SIGNAL_SIZE = 48000
ZERO_INIT_TENSOR = 1


class SingleSpeakerEval:
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

        # Use different distance metrics
        sims_funcs = []
        for loss_cfg, loss_func_params in zip(self.cfg['similarity_config'], self.cfg['similarity_func_params'].values()):
            sims_func = get_instance(loss_cfg['module_name'],
                                     loss_cfg['class_name'])(**loss_func_params)
            sims_funcs.append(sims_func)

        self.sims2 = sims_funcs[0]  # TODO change configuration file to handle only one similarity
        # self.sims = torch.nn.CosineSimilarity(dim=1) # TODO: change hard-coded function
        # self.loss = Loss(self.model, loss_funcs, **self.cfg['loss_params'])
        # self.cfg['attack_params']['loss_function'] = self.loss.loss_gradient
        # self.attack = get_instance(self.cfg['attack_config']['module_name'],
        #                            self.cfg['attack_config']['class_name'])(**self.cfg['attack_params'])

        # save_config_to_file(self.cfg, self.cfg['current_dir'])
        df_cols = ['eval_emb_y', 'x_segment_emb_y', 'x_adv_emb_y']
        pd.DataFrame(columns=df_cols).to_csv(os.path.join(self.prev_files_loc, 'sims_results.csv'))

    def evaluate(self):

        # self.sims = torch.nn.CosineSimilarity(dim=2)
        eval_adv = self.eval + self.perturbation
        x_other_segment = self.x + self.perturbation

        similarity_eval_emb = self.sims(self.model.encode_batch(eval_adv).cpu(), torch.from_numpy(self.y)) # other file
        similarity_x_segment_emb = self.sims(self.model.encode_batch(x_other_segment).cpu(), torch.from_numpy(self.y)) # same file with other segment
        similarity_x_adv_emb = self.sims(self.model.encode_batch(torch.from_numpy(self.x_adv_npy)).cpu(), torch.from_numpy(self.y)) # original adversarial file

        # similarity_eval_emb2 = self.sims2(self.model.encode_batch(eval_adv).cpu(), torch.from_numpy(self.y))  # other file
        # similarity_x_emb2 = self.sims2(self.model.encode_batch(x_other_segment).cpu(),
        #                              torch.from_numpy(self.y))  # same file with other segment
        # similarity_x_adv_emb2 = self.sims2(self.model.encode_batch(torch.from_numpy(self.x_adv_npy)).cpu(),
        #                                  torch.from_numpy(self.y))  # original adversarial file

        sims_results = {'eval_emb_y':similarity_eval_emb,'x_segment_emb_y':similarity_x_segment_emb,'x_adv_emb_y':similarity_x_adv_emb}
        print(sims_results)
        batch_id = 0
        self.similarity_values.append([str(round(similarity_eval_emb.item(), 5))
                                      ,str(round(similarity_x_segment_emb.item(), 5))
                                      ,str(round(similarity_x_adv_emb.item(),5))])

        self.register_similarity_values(batch_id)
        # pd.DataFrame(columns=[f'iter {str(iteration)}' for iteration in range(1)]).to_csv(os.path.join(self.cfg['current_dir'], 'sims.csv'))

    def register_similarity_values(self, batch_id):
        batch_result = pd.Series([f'{batch_id}'] + self.similarity_values[batch_id]) # self.similarity_values)
        batch_result.to_frame().T.to_csv(os.path.join(self.prev_files_loc, 'sims_results.csv'), mode='a',
                                         header=False, index=False)


def get_indices_from_labels(labels):
    spk_id_labels = []
    array_ind = torch.arange(0, BATCH_SIZE)

    curr_indices = []
    curr_indices_bool = []
    for i, label in enumerate(labels):
        if str(label.item()) not in spk_id_labels:
            spk_id_labels.append(str(label.item()))
            curr_indices.append(i)
            curr_indices_bool.append(True) # i
        else:
            curr_indices_bool.append(False)
    curr = torch.tensor(curr_indices)
    mask = torch.all(array_ind - curr.unsqueeze(0).T, dim=0)
    return mask, spk_id_labels, curr_indices

    # self.indices = [str(label.item()) for i,label in enumerate(labels) if label not in curr_index]


class MultiSpeakersAttack:
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.cfg.attack_type = self.__class__.__name__  # for logging purposes when inheriting from another class
        self.model = get_speaker_model(cfg)
        self.datasetI = DataLoader(BasicLibriSpeechDataset(self.cfg['data_path'], suffix_p='/*/*'), batch_size=BATCH_SIZE, shuffle=True)
        # TODO: create class eval which derived from base eval
        # self.sims = get_instance(self.cfg['similarity_config_one']['module_name'],
        #                          self.cfg['similarity_config_one']['class_name'])(**self.cfg['similarity_params'])
        # basic_libri_dataset = BasicLibriSpeechDataset(self.cfg['data_path'])
        # train_dataloader = DataLoader(basic_libri_dataset, batch_size=4, shuffle=False)
        # train_features = next(iter(self.datasetI))
        self.labels = None
        self.x = None
        self.accumulated_delta = None
        self.accumulated_delta_mean = None
        self.accumulated_delta_mean_input_delta = None
        self.adv_x = None
        self.eval = get_signal_from_wav_random(os.path.join(self.cfg['wav_path'])).unsqueeze(0)
        self.eval2 = get_signal_from_wav_random(os.path.join(self.cfg['wav_path2'])).unsqueeze(0)
        self.eval3 = get_signal_from_wav_random(os.path.join(self.cfg['wav_path3'])).unsqueeze(0)
        self.sims2 = get_instance(self.cfg['similarity_config_one']['module_name'],
                                  self.cfg['similarity_config_one']['class_name'])(**self.cfg['similarity_params'])
        # self.sims2 = torch.nn.CosineSimilarity(dim=2)
        self.bool_indices = None
        self.enroll_order_list = self.cfg['speakers_id']
        self.enroll_order_list.sort()

        self.speakers_enroll = {}
        with torch.no_grad():
            # enroll_list = self.cfg['speakers_id']
            #self.y = self.model.encode_batch(train_features)
            for speaker in self.enroll_order_list:
                self.speakers_enroll[speaker] = get_speaker_embedding(self.model,
                                                     self.cfg['dataset_config']['root_path'],
                                                     speaker,
                                                     self.cfg['num_wavs_for_emb'],
                                                     self.cfg['fs'],
                                                     self.cfg['device'])

        # Use different distance metrics
        self.y = torch.mean(torch.from_numpy(np.array([x.cpu().numpy() for x in list(self.speakers_enroll.values())])).to(self.cfg['device']))
        # self.y = torch.mean(torch.from_numpy(np.array([x.cpu().numpy() for x in list(self.speakers_enroll.values())])),
        #                     dim=0).to(self.cfg['device'])
        self.accumulated_delta_dict = {spk: torch.zeros([ZERO_INIT_TENSOR, SIGNAL_SIZE]) for spk in self.enroll_order_list}

        loss_funcs = []
        for loss_cfg, loss_func_params in zip(self.cfg['losses_config'], self.cfg['loss_func_params'].values()):
            loss_func = get_instance(loss_cfg['module_name'],
                                     loss_cfg['class_name'])(**loss_func_params)
            loss_funcs.append(loss_func)
        self.eval_files_names = list(self.speakers_enroll.keys())
        self.eval_files_names.sort()
        self.eval_files_src = [self.eval, self.eval2, self.eval3]
        self.eval_files_signal = {spk: signal for spk, signal in zip(self.eval_files_names, self.eval_files_src)}

        self.loss = Loss(self.model, loss_funcs, **self.cfg['loss_params'])
        self.cfg['attack_params']['loss_function'] = self.loss.loss_gradient
        self.attack = get_instance(self.cfg['attack_config']['module_name'],
                                   self.cfg['attack_config']['class_name'])(**self.cfg['attack_params'])

        save_config_to_file(self.cfg, self.cfg['current_dir'])
        create_dirs_not_exist(self.cfg['current_dir'], ["perturbation", "adversarial"])
        pd.DataFrame(columns=[f'iter {str(iteration)}' for iteration in range(self.cfg['attack_params']['max_iter'])]). \
            to_csv(os.path.join(self.cfg['current_dir'], 'loss_results.csv'))

    # TODO: Perhaps use that function in other place ?
    def generate_x(self):

        if self.x is not None:
            del self.x
            del self.labels
            del self.bool_indices
        train_features, labels = next(iter(self.datasetI))
        self.bool_indices, self.spk_id_in_batch, self.indices_as_int = get_indices_from_labels(labels)
        curr_accumulated_delta = torch.stack([self.accumulated_delta_dict[speaker] for speaker in self.spk_id_in_batch]).squeeze()
        mask = self.bool_indices
        self.bool_indices = mask.logical_not()
        self.pre_x = torch.zeros(curr_accumulated_delta.shape[0] + mask.count_nonzero(), SIGNAL_SIZE)
        self.pre_x[mask.logical_not(), :] = curr_accumulated_delta
        self.x = (self.pre_x + train_features).to(self.cfg['device'])
        self.labels = torch.stack([self.speakers_enroll[str(label.item())] for label in labels]).to(self.cfg['device'])
        # self.x = train_features.to(self.cfg['device']) # working
        # self.x = torch.stack[for i,t_emb in enumerate(train_features) if i in self.indices else   ]
        # train_features[self.indices] + self.accumulated_delta
    def generate(self, curr, total):
        x_adv = self.attack.generate(self.x, self.labels, {'cur': curr, 'total': total})
        x_delta_all = (self.x.cpu() - x_adv.cpu())
        self.update_delta(x_delta_all)


        # self.accumulated_delta += (self.x.cpu() - x_adv.cpu())
        self.accumulated_delta_mean += torch.mean(self.x.cpu() - x_adv.cpu(), dim=0)
        self.accumulated_delta_mean_input_delta += torch.mean(self.x.cpu(), dim=0) - torch.mean(x_adv.cpu(), dim=0)
        self.register_loss_values(curr)
        loss_plot(self.cfg['current_dir'])

    def update_delta(self, all_delta):
        for ind, speaker in zip(self.indices_as_int, self.spk_id_in_batch):
            self.accumulated_delta_dict[speaker] = all_delta[ind].unsqueeze(0)




    def multi_generate(self):
        # curr_batch = 1
        # self.accumulated_delta = torch.zeros([BATCH_SIZE, SIGNAL_SIZE]) # inputs.shape
        # self.accumulated_delta = torch.zeros([NUMBER_OF_SPEAKERS, SIGNAL_SIZE])  # inputs.shape

        self.accumulated_delta_mean = torch.zeros([ZERO_INIT_TENSOR, SIGNAL_SIZE])  # inputs.shape
        self.accumulated_delta_mean_input_delta = torch.zeros([ZERO_INIT_TENSOR, SIGNAL_SIZE])  # inputs.shape

        total = 64
        for i in range(total):
            self.generate_x()
            self.generate(curr=i, total=total)
        print("end multi_ generate")

    def register_loss_values(self, batch_id):
        batch_result = pd.Series([f'batch{batch_id}'] + self.attack.loss_values)
        batch_result.to_frame().T.to_csv(os.path.join(self.cfg['current_dir'], 'loss_results.csv'), mode='a',
                                         header=False, index=False)

    def evaluate(self):
        # eval_adv_single = torch.broadcast_to(self.eval, [BATCH_SIZE, SIGNAL_SIZE]) + self.accumulated_delta
        eval_adv_mean = torch.broadcast_to(self.eval, [ZERO_INIT_TENSOR, SIGNAL_SIZE]) + self.accumulated_delta_mean
        eval_adv_input_mean = self.eval + self.accumulated_delta_mean_input_delta

        # similarity_eval_single = self.sims2(self.model.encode_batch(eval_adv_single).cpu(), self.y.cpu())
        similarity_mean_ = self.sims2(self.model.encode_batch(eval_adv_mean).cpu(), self.y.cpu())
        similarity_mean_input = self.sims2(self.model.encode_batch(eval_adv_input_mean).cpu(), self.y.cpu())

        sims_results = { 'eval_adv_mean': similarity_mean_,
                        'eval_adv_input_mean': similarity_mean_input}
        print(sims_results)
        print("%%%%%%")

        print("*****")
        for speaker_id in self.speakers_enroll.keys():
            self.eval_file(speaker_id)



        # _ = self.attack.generate(self.x, self.y, {'cur': 1, 'total': 1})
        # self.register_loss_values(0)
        # loss_plot(self.cfg['current_dir'])

    def eval_file(self,eval_speaker):
        print("eval_file: ",eval_speaker)
        eval_emb = self.speakers_enroll[eval_speaker]
        eval_signal = self.eval_files_signal[eval_speaker]
        eval_adv_single = torch.broadcast_to(eval_signal, [ZERO_INIT_TENSOR, SIGNAL_SIZE]) + self.accumulated_delta_dict[eval_speaker]
        eval_adv_mean = torch.broadcast_to(eval_signal, [ZERO_INIT_TENSOR, SIGNAL_SIZE]) + self.accumulated_delta_mean
        eval_adv_input_mean = eval_signal + self.accumulated_delta_mean_input_delta

        print("SNR eval single:",calculator_snr_direct(eval_signal, eval_adv_single))
        print("SNR eval mean:", calculator_snr_direct(eval_signal, eval_adv_mean))

        save_audio_as_wav(self.cfg['current_dir'], eval_adv_single[0].unsqueeze(0), "adversarial", f'{eval_speaker}.wav')
        save_audio_as_wav(self.cfg['current_dir'], eval_adv_mean, "adversarial",f'{eval_speaker}_mean.wav')
        similarity_eval_single = self.sims2(self.model.encode_batch(eval_adv_single).cpu(), eval_emb.cpu())
        similarity_mean_ = self.sims2(self.model.encode_batch(eval_adv_mean).cpu(), eval_emb.cpu())
        similarity_mean_input = self.sims2(self.model.encode_batch(eval_adv_input_mean).cpu(), eval_emb.cpu())
        similarity_eval_clear = self.sims2(self.model.encode_batch(eval_signal).cpu(), eval_emb.cpu())
        sims_results = {'eval_adv_single': similarity_eval_single, 'eval_adv_mean': similarity_mean_,
                        'eval_adv_input_mean': similarity_mean_input,'clear':similarity_eval_clear}
        print(sims_results)
        print("%%%%%%")


def check_multi_speakers():
    config_type = 'MultiSpeakers'
    cfg = config_dict[config_type]()
    attack = MultiSpeakersAttack(cfg)
    # attack.generate()
    attack.multi_generate()
    attack.evaluate()


def main():
    # config_type = 'SingleWAV'
    # cfg = config_dict[config_type]()
    # attack = SingleWAVAttack(cfg)
    # attack.generate()
    #######################################################
    # A = torch.arange(0, 10)
    # B = torch.tensor([0, 2, 4, 7])
    #
    # mask = torch.all(A - B.unsqueeze(0).T, dim=0)
    #
    # C = torch.rand(4, 5)
    # D = torch.zeros(C.shape[0] + mask.count_nonzero(), 5)
    #
    # D[mask.logical_not(), :] = C
    # print(C)
    # print(D)
    # E = torch.zeros(10, 5)
    #
    # E[B] = C
    # print(E)
    ##########################################
    check_multi_speakers()
    print("finish main ")

if __name__ == '__main__':
    main()
