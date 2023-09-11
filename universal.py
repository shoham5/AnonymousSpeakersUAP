from builtins import enumerate
import torch
import gc
import os
import pandas as pd
from tqdm import tqdm
from configs.attacks_config import config_dict
from utils.model_utils import get_speaker_model, get_multiple_speaker_models
from utils.general import get_instance, save_config_to_file, get_pert, PESQ, calculate_snr_github_direct_pkg
from utils.data_utils import get_embeddings, get_loaders
from utils.data_utils import save_audio_as_wav, create_dirs_not_exist, save_emb_as_npy,save_to_pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class UniversalAttack:

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.cfg.attack_type = self.__class__.__name__  # for logging purposes when inheriting from another class
        self.norm = 2
        self.eps_projection = 1
        self.alpha_snr = 0
        print(f"eps_projection: {self.eps_projection}")
        print(f"self.alpha_snr: {self.alpha_snr}")
        self.model = get_speaker_model(cfg)
        self.multi_models = get_multiple_speaker_models(cfg)
        self.is_first = True
        self.train_loader, self.val_loader, _ = get_loaders(self.cfg['loader_params'],
                                                            self.cfg['dataset_config'],
                                                            splits_to_load=['train', 'validation'])

        self.speaker_embeddings = get_embeddings(self.model,
                                                 self.train_loader,
                                                 self.cfg['dataset_config']['speaker_labels_mapper'].keys(),
                                                 self.cfg['device'])

        self.loss_fns = []
        for i, (loss_cfg, loss_func_params) in enumerate(zip(self.cfg['losses_config'], self.cfg['loss_func_params'].values())):
            loss_func = get_instance(loss_cfg['module_name'],
                                     loss_cfg['class_name'])(**loss_func_params)
            self.loss_fns.append((loss_func, self.cfg['loss_params']['weights'][i]))

        save_config_to_file(self.cfg, self.cfg['current_dir'])

        self.loss_train_values = []
        self.loss_eval_values = []
        self.similarity_train_values = []
        self.similarity_eval_values = []
        self.snr_train_values = []
        self.snr_eval_values = []
        self.pesq_train_values = []
        self.pesq_eval_values = []
        self.counter = 0
        self.prev_files_loc = self.cfg['current_dir']
        df_cols = ['first', 'sec', 'third']
        pd.DataFrame(columns=df_cols).to_csv(os.path.join(self.prev_files_loc, 'pesq_results.csv'))
        pd.DataFrame(columns=df_cols).to_csv(os.path.join(self.prev_files_loc, 'sims_results.csv'))
        pd.DataFrame(columns=df_cols).to_csv(os.path.join(self.prev_files_loc, 'snr_results.csv'))
        pd.DataFrame(columns=[f'iter {str(iteration)}' for iteration in range(self.cfg['attack_params']['max_iter'])]). \
            to_csv(os.path.join(self.cfg['current_dir'], 'loss_results.csv'))
        create_dirs_not_exist(self.cfg['current_dir'], ["perturbation", "embedding", "adversarial"])

    def generate(self):
        adv_pert = get_pert(self.cfg['init_pert_type'], size=self.cfg['fs'] * self.cfg['num_of_seconds'])
        loss_config = "snr"  # "snr" or "cosim"
        optimizer = torch.optim.Adam([adv_pert], lr=self.cfg.start_learning_rate, amsgrad=True)
        scheduler = self.cfg.scheduler_factory(optimizer)
        print("eps_projection: ",self.eps_projection)
        print(f"\nconfig generate: # speakers: {self.cfg['dataset_config']['number_of_speakers']}_uap_init_{self.cfg['init_pert_type']}_{loss_config}_ep100_spk100")
        for epoch in range(self.cfg.epochs):
            running_loss = 0.0
            running_snr = 0.0
            progress_bar = tqdm(enumerate(self.train_loader), desc=f'Epoch {epoch}', total=len(self.train_loader), ncols=150)
            prog_bar_desc = 'Batch Loss: {:.6}'  # , SNR: {:.6}'
            is_snr = False
            if loss_config == "snr":
                is_snr = True

            for i_batch, (cropped_signal_batch, person_ids_batch) in progress_bar:
                if cropped_signal_batch.shape[0] != 64: continue
                print("adv_pert: ", adv_pert)
                loss, running_loss, adv_cropped_signal_batch = self.forward_step(adv_pert, cropped_signal_batch,
                                                                                 person_ids_batch, running_loss,
                                                                                 snr=is_snr)

                self.similarity_train_values.append(str(round(loss.item(), 5)))

                pesq_loss = PESQ(cropped_signal_batch, adv_cropped_signal_batch)
                snr_loss_sec = calculate_snr_github_direct_pkg(adv_cropped_signal_batch.cpu().detach(),
                                                           cropped_signal_batch).item()

                self.snr_train_values.append(str(round(snr_loss_sec, 5)))
                self.pesq_train_values.append(str(round(pesq_loss, 5)))
                self.loss_train_values.append(str(round(loss.item(), 5)))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                adv_pert.data = self.projection(adv_pert)
                adv_pert.data.clamp(-1, 1)
                progress_bar.set_postfix_str(prog_bar_desc.format(running_loss / (i_batch + 1)))

            val_loss = self.evaluate(adv_pert)
            print("evaluate_running_loss: ",val_loss)

            scheduler.step(val_loss)

        self.save_files(adv_pert)

    def apply_perturbation(self, input_batch, adv_pert, clip=True):
        input_batch = input_batch.to(self.cfg['device'])
        adv_pert = adv_pert.to(self.cfg['device'])
        adv = input_batch + adv_pert
        if clip:
            adv.data.clamp_(-1, 1)
        del input_batch
        del adv_pert
        torch.cuda.empty_cache()
        return adv

    def forward_step(self, adv_pert, cropped_signal_batch, person_ids_batch, running_loss, snr):

        cropped_signal_batch = cropped_signal_batch.to(self.cfg['device'])
        person_ids_batch = person_ids_batch.to(self.cfg['device'])
        adv_pert = adv_pert.to(self.cfg['device'])
        adv_batch = self.apply_perturbation(cropped_signal_batch, adv_pert)
        adv_embs = self.model.encode_batch(adv_batch)
        loss = self.loss_fn(adv_embs, person_ids_batch)

        if snr:
            snr_loss_sec = calculate_snr_github_direct_pkg(adv_batch,
                                                       cropped_signal_batch)
            snr_loss = ((125 - snr_loss_sec) / 125) * self.alpha_snr

            loss += snr_loss

        running_loss += loss.item()
        del cropped_signal_batch
        del person_ids_batch
        del adv_pert
        torch.cuda.empty_cache()
        return loss, running_loss, adv_batch

    def loss_fn(self, adv_embs, speaker_ids):
        gt_embeddings = torch.index_select(self.speaker_embeddings, index=speaker_ids, dim=0)  # .squeeze(-2)
        loss = torch.zeros(1, device=self.cfg['device'])
        for loss_fn, loss_weight in self.loss_fns:
            loss += loss_weight * loss_fn(adv_embs, gt_embeddings).squeeze().mean()
        return loss

    def projection(self, values):
        tol = 10e-8
        values_tmp = values.reshape(values.shape[0], -1)
        if self.norm == 2:
            values_tmp = (values_tmp *
                          torch.min(
                              torch.tensor([1.0], dtype=torch.float32),
                              self.eps_projection / (torch.norm(values_tmp, p=2, dim=1) + tol),
                          ).unsqueeze_(-1)
                          )
        elif self.norm == 1:
            values_tmp = (values_tmp *
                          torch.min(
                              torch.tensor([1.0], dtype=torch.float32),#.to(self.device),
                              self.eps_projection / (torch.norm(values_tmp, p=1, dim=1) + tol),
                          ).unsqueeze_(-1)
                          )
        elif self.norm == 'inf':
            values_tmp = values_tmp.sign() * torch.min(values_tmp.abs(),
                                             torch.broadcast_to(torch.Tensor([self.eps_projection]), values_tmp.shape))
        values = values_tmp.reshape(values.shape)

        del values_tmp
        torch.cuda.empty_cache()
        return values

    @torch.no_grad()
    def evaluate(self, adv_pert):
        self.counter += 1
        running_loss = 0.0
        running_snr = 0.0
        running_sims = 0.0
        progress_bar = tqdm(enumerate(self.val_loader), desc=f'Eval', total=len(self.val_loader), ncols=150)
        prog_bar_desc = 'Batch Loss: {:.6}, SNR: {:.6}'
        for i_batch, (cropped_signal_batch, person_ids_batch) in progress_bar:
            if cropped_signal_batch.shape[0] != 64: continue
            loss, running_loss, adv_cropped_signal_batch = self.forward_step(adv_pert, cropped_signal_batch,
                                                                                  person_ids_batch, running_loss,
                                                                             snr=True)

            adv_snr = calculate_snr_github_direct_pkg(adv_cropped_signal_batch.cpu().detach(),
                                                           cropped_signal_batch).item()

            self.loss_eval_values.append(round(loss.item(), 5))
            self.snr_eval_values.append(round(adv_snr, 5))

            running_snr += adv_snr
            running_sims += round(loss.item(), 5)
            progress_bar.set_postfix_str(prog_bar_desc.format(running_loss / (i_batch + 1),
                                                              running_snr / (i_batch + 1)))

        return running_loss / len(self.val_loader)

    def save_files(self, adv_pert):
        self.register_similarity_values(self.counter)
        self.register_snr_values(self.counter)
        self.register_loss_values(self.counter)
        self.register_pesq_values(self.counter)

        save_audio_as_wav(self.cfg['current_dir'], adv_pert.detach(), "perturbation", 'uap_ep100_spk100.wav')
        save_emb_as_npy(self.cfg['current_dir'], adv_pert.detach().numpy(), "perturbation", 'uap_ep100_spk100')
        save_to_pickle(self.cfg['current_dir'], self.speaker_embeddings, "embedding", 'speaker_embeddings')

    def register_pesq_values(self, batch_id):
        batch_result = pd.Series([f'batch_train_{batch_id}'] + self.pesq_train_values)
        batch_result.to_frame().T.to_csv(os.path.join(self.cfg['current_dir'], 'pesq_results.csv'), mode='a',
                                         header=False, index=False)
        batch_result = pd.Series([f'batch_eval_{batch_id}'] + self.pesq_eval_values)
        batch_result.to_frame().T.to_csv(os.path.join(self.cfg['current_dir'], 'pesq_results.csv'), mode='a',
                                         header=False, index=False)

    def register_loss_values(self, batch_id):
        batch_result = pd.Series([f'batch_train_{batch_id}'] + self.loss_train_values)
        batch_result.to_frame().T.to_csv(os.path.join(self.cfg['current_dir'], 'loss_results.csv'), mode='a',
                                         header=False, index=False)
        batch_result = pd.Series([f'batch_eval_{batch_id}'] + self.loss_eval_values)
        batch_result.to_frame().T.to_csv(os.path.join(self.cfg['current_dir'], 'loss_results.csv'), mode='a',
                                         header=False, index=False)

    def register_similarity_values(self, batch_id):
        batch_result = pd.Series([f'eval_{batch_id}'] + self.similarity_eval_values) # self.similarity_values)
        batch_result.to_frame().T.to_csv(os.path.join(self.prev_files_loc, 'sims_results.csv'), mode='a',
                                         header=False, index=False)


    def register_snr_values(self, batch_id):
        batch_result = pd.Series([f'train_{batch_id}'] + self.snr_train_values)  # self.similarity_values)
        batch_result.to_frame().T.to_csv(os.path.join(self.prev_files_loc, 'snr_results.csv'), mode='a',
                                         header=False, index=False)
        batch_result = pd.Series([f'eval_{batch_id}'] + self.snr_eval_values) # self.similarity_values)
        batch_result.to_frame().T.to_csv(os.path.join(self.prev_files_loc, 'snr_results.csv'), mode='a',
                                         header=False, index=False)

def main():

    torch.cuda.empty_cache()
    config_type = 'Universal'
    cfg = config_dict[config_type]()
    attack = UniversalAttack(cfg)
    attack.generate()
    print("finish main: ")
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == '__main__':
    main()
