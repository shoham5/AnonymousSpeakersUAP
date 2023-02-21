import pandas as pd
import os
import torch
import random

from configs.attacks_config import config_dict
from utils.model_utils import get_speaker_model
from utils.general import get_instance, save_config_to_file
from utils.data_utils import get_signal_and_fs_from_wav,get_signal_from_wav_random,\
    get_speaker_embedding, get_pretrained_speaker_embedding,save_audio_as_wav, create_dirs_not_exist,\
    save_emb_as_npy,load_from_npy,load_audio_from_wav
from utils.losses_utils import Loss
from visualization.plots import loss_plot
from torchaudio.datasets.librispeech import LIBRISPEECH
from utils.general import save_file_to_npy, save_signal_to_wav


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

    def evaluate(self):

        # self.sims = torch.nn.CosineSimilarity(dim=2)
        eval_adv = self.eval + self.perturbation
        x_other_segment = self.x + self.perturbation

        similarity_eval_emb = self.sims(self.model.encode_batch(eval_adv).cpu(), torch.from_numpy(self.y)) # other file
        similarity_x_segment_emb = self.sims(self.model.encode_batch(x_other_segment).cpu(), torch.from_numpy(self.y)) # same file with other segment
        similarity_x_adv_emb = self.sims(self.model.encode_batch(torch.from_numpy(self.x_adv_npy)).cpu(), torch.from_numpy(self.y)) # original adversarial file

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

    def save_files(self):
        save_audio_as_wav(self.cfg['current_dir'], self.perturbation, "perturbation", self.cfg['wav_path'])
        save_audio_as_wav(self.cfg['current_dir'], self.x_adv, "adversarial", self.cfg['wav_path'])
        # save_emb_as_npy(self.cfg['current_dir'].split('/')[0], self.y.cpu(), "embedding", self.cfg['wav_path'])
        save_emb_as_npy(self.cfg['current_dir'], self.perturbation, "perturbation", self.cfg['wav_path'])
        save_emb_as_npy(self.cfg['current_dir'], self.x_adv, "adversarial", self.cfg['wav_path'])

    def get_current_dir(self):
        return self.cfg['current_dir']

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
