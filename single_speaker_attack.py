import pandas as pd
import os
import torch
import random

from configs.attacks_config import config_dict
from utils.model_utils import get_speaker_model
from utils.general import get_instance, save_config_to_file
from utils.data_utils import get_signal_and_fs_from_wav  # , get_speaker_embedding
from utils.losses_utils import Loss
from visualization.plots import loss_plot


class SingleWAVAttack:
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.cfg.attack_type = self.__class__.__name__  # for logging purposes when inheriting from another class
        self.model = get_speaker_model(cfg)

        signal, _ = get_signal_and_fs_from_wav(os.path.join(self.cfg['wav_path']))
        signal_len = signal.shape[1]
        start_idx = random.randint(0, signal_len - (self.cfg['fs'] * 3 + 1))
        cropped_signal = signal[0][start_idx: (start_idx + 1) + (self.cfg['fs'] * 3)]
        self.x = cropped_signal.to(self.cfg['device'])

        with torch.no_grad():
            self.y = self.model.encode_batch(cropped_signal)
        # self.speaker_emb = get_speaker_embedding(self.model,
        #                                          self.cfg['dataset_config']['root_path'],
        #                                          self.cfg['speaker_id'],
        #                                          self.cfg['num_wavs_for_emb'],
        #                                          self.cfg['fs'],
        #                                          self.cfg['device'])

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

    def generate(self):
        _ = self.attack.generate(self.x, self.y, {'cur': 1, 'total': 1})
        self.register_loss_values(0)
        loss_plot(self.cfg['current_dir'])

    def register_loss_values(self, batch_id):
        batch_result = pd.Series([f'batch{batch_id}'] + self.attack.loss_values)
        batch_result.to_frame().T.to_csv(os.path.join(self.cfg['current_dir'], 'loss_results.csv'), mode='a', header=False, index=False)


def main():
    config_type = 'SingleWAV'
    cfg = config_dict[config_type]()
    attack = SingleWAVAttack(cfg)
    attack.generate()


if __name__ == '__main__':
    main()
