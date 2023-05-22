from transformers import AdamW
import torch

import gc
import os
import math
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import sys
import numpy as np
from configs.attacks_config import config_dict
from utils.model_utils import get_speaker_model, get_speaker_model_by_name
from utils.general import get_instance, save_config_to_file, calculator_snr_direct, get_pert, PESQ, calculate_l2, calculate_snr_github_direct
from utils.data_utils import get_embeddings, get_loaders
from utils.losses_utils import WSNRLoss,SNRLoss,custom_clip
from utils.similarities_utils import CosineSimilaritySims
from utils.data_utils import save_audio_as_wav, create_dirs_not_exist, save_emb_as_npy,\
    load_from_npy, load_audio_from_wav, save_to_pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[1]
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))
#
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))



EPS = 1. #0.9 # 0.007,0.05, 0.01, 0.1, 0.5,
EPS_PROJECTION = 0.3#0.7 # 0.1, 0.3, 0.5, 0.75, 1
COUNTER = 0
class UniversalAttack:

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.cfg.attack_type = self.__class__.__name__  # for logging purposes when inheriting from another class
        self.norm = 2
        self.eps_projection = 1 # 100  # 10 ,50 , 100 , 1

        self.model = get_speaker_model(cfg)
        # self.model_by_name = get_speaker_model_by_name(self.cfg.model_name,cfg, device)
        self.is_first = True
        self.train_loader, self.val_loader, _ = get_loaders(self.cfg['loader_params'],
                                                            self.cfg['dataset_config'],
                                                            splits_to_load=['train', 'validation'])

        self.speaker_embeddings = get_embeddings(self.model,
                                                 self.train_loader,
                                                 self.cfg['dataset_config']['speaker_labels_mapper'].keys(),
                                                 self.cfg['device'])

        # Use different distance metrics
        self.loss_fns = []
        for i, (loss_cfg, loss_func_params) in enumerate(zip(self.cfg['losses_config'], self.cfg['loss_func_params'].values())):
            loss_func = get_instance(loss_cfg['module_name'],
                                     loss_cfg['class_name'])(**loss_func_params)
            self.loss_fns.append((loss_func, self.cfg['loss_params']['weights'][i]))

        save_config_to_file(self.cfg, self.cfg['current_dir'])

        self.snr_loss = WSNRLoss()
        self.loss_train_values = []
        self.loss_eval_values = []
        self.similarity_eval_values = []
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
        temp_adv_pert = 0

        # rndr_eps = np.random.uniform(0.1,self.eps_projection)
        # # change_eps_projection = math.floor(self.cfg.epochs / 4)
        # print("rndr_eps: ", rndr_eps)
        loss_config = "cosim" #"pesq_snr" # snr   cosim
        optimizer = AdamW([adv_pert], lr=self.cfg.start_learning_rate)
        # optimizer = torch.optim.Adam([adv_pert], lr=self.cfg.start_learning_rate, amsgrad=True)
        scheduler = self.cfg.scheduler_factory(optimizer)
        print("eps_projection: ",self.eps_projection)
        print(f"\nconfig generate: uap_init_{self.cfg['init_pert_type']}_{loss_config}_ep100_spk100")
        for epoch in range(self.cfg.epochs):
            running_loss = 0.0
            running_snr = 0.0
            progress_bar = tqdm(enumerate(self.train_loader), desc=f'Epoch {epoch}', total=len(self.train_loader), ncols=150)
            prog_bar_desc = 'Batch Loss: {:.6}, SNR: {:.6}'
            # if epoch % change_eps_projection ==0 and epoch != 0:
            #     self.eps_projection *= 0.36 # 0.7
            #     print("\n####### change eps to #######: ", self.eps_projection )

            for i_batch, (cropped_signal_batch, person_ids_batch) in progress_bar:
                loss, running_loss, adv_cropped_signal_batch = self.forward_step(adv_pert, cropped_signal_batch, person_ids_batch, running_loss)

                pesq_loss = PESQ(cropped_signal_batch, adv_cropped_signal_batch)
                print("\npesq mean: ", pesq_loss)
                snr_loss = calculate_snr_github_direct(adv_cropped_signal_batch.cpu().detach().numpy(),
                                                           cropped_signal_batch.cpu().detach().numpy())

                # snr_loss = calculator_snr_direct(cropped_signal_batch, adv_cropped_signal_batch,
                #                                  device=self.cfg['device'])

                print("snr : ", snr_loss)
                print("cosim loss: ", loss.item())

                # snr_git_loss = calculate_snr_github_direct(adv_cropped_signal_batch.cpu().detach().numpy(),
                #                                            cropped_signal_batch.cpu().detach().numpy())
                # print("snr_git : ", snr_git_loss)
                ###################################################################################
                temp_loss = ((100 - snr_loss) / 100)  # + (factor_snr/60)
                print("snr loss : ", temp_loss)
                pesq_temp_loss = (4.5 - pesq_loss) / 4.5
                print("pesq loss : ", pesq_temp_loss)

                # if self.is_first: # i_batch is 0:
                #     snr_loss = 46.5
                #     temp_loss = ((60 - snr_loss) / 60)
                #     self.is_first = False

                if loss_config == "snr":
                    loss.data += temp_loss
                elif loss_config == "pesq_snr":
                    loss.data += (0.5 * pesq_temp_loss)  # using pesq + snr
                    loss.data += (0.5 * temp_loss)  # using pesq + snr

                    # loss.data += temp_loss


                self.loss_train_values.append(str(round(loss.item(), 5)))
                self.snr_train_values.append(str(round(snr_loss, 5)))
                self.pesq_train_values.append(str(round(pesq_loss, 5)))
                #


                print("loss after: ", loss.item())
                running_snr += snr_loss

                # print("log loss: " ,-1 * math.log(loss.data, 2))
                # loss.data = torch.tensor([-1]).to(device) * math.log(loss.data, 2)
                # # print("log loss update : ", loss.data)
########################################################################################
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # projection - attack wise - self.projection()
                adv_pert.data = self.projection(adv_pert)
                # clip - stay in audio domain  - adv_pert.clamp(-1, 1)
                adv_pert.data.clamp(-1, 1)
                # adv_pert.clamp(-2 ** 15, 2 ** 15 - 1)

                #adv_snr = calculator_snr_direct(src_array=cropped_signal_batch, adv_array=self.apply_perturbation(cropped_signal_batch, adv_pert, clip=False).to('cpu'))
                # adv_snr = snr_loss
                    # calculator_snr_direct(src_array=cropped_signal_batch, adv_array=adv_cropped_signal_batch, device=self.cfg['device'])
                # print("loss after adv snr: ", loss.item())



                progress_bar.set_postfix_str(prog_bar_desc.format(running_loss / (i_batch + 1),
                                                                  running_snr / (i_batch + 1)))

            # if epoch != 0 :
            #     adv_pert.data = 0.6 * adv_pert.data + 0.4 * temp_adv_pert
            #     # adv_pert.data = self.projection(adv_pert)
            # temp_adv_pert = adv_pert.data
            # if epoch % 7 ==0 and i_batch % 25 ==0:


            val_loss = self.evaluate(adv_pert)
            print("evaluate_loss: ",val_loss)

            scheduler.step(val_loss)

        self.save_files(adv_pert)

    def apply_perturbation(self, input, adv_pert, clip=True):
        input = input.to(self.cfg['device'])
        adv_pert = adv_pert.to(self.cfg['device'])

        # adv = input + self.projection(adv_pert)
                              #with EPS= 0.1  #  Batch Loss: 0.494944, SNR: 15.0485, SIMS: 0.376576]
        # adv = custom_clip(input, input + adv_pert, 0.01) # self.cfg['device']
        adv = input + adv_pert #* EPS  # EPS=0.1   Batch Loss: 0.649802, SNR: 20.1466, SIMS: 0.446894] multiply by epsilon work
        # adv = input + adv_pert * EPS  #EPS=0.5  Batch Loss: 0.346293, SNR: 4.0024, SIMS: 0.330975]
        # adv = input + adv_pert   #  EPS=1       Batch Loss: 0.278237, SNR: -0.228931, SIMS: 0.297413]
        if clip:
            adv.data.clamp_(-1, 1)  # clip with eps 0.1 get sims less then 0.5
        del input
        del adv_pert
        torch.cuda.empty_cache()
        return adv

    # def momentom_norm(self,x_adv,targets, momentum):
    #     decay = 0.1
    #     tol = 10e-8
    #     targeted = True
    #     adv_x_shape = x_adv.shape
    #
    #     grad, loss_value = self.loss_fn(x_adv, targets)
    #     grad = grad * (1 - 2 * int(targeted))
    #     if torch.any(grad.isnan()):
    #         grad[grad.isnan()] = 0.0
    #
    #     ind = tuple(range(1, len(adv_x.shape)))
    #     grad = grad / (torch.sum(grad.abs(), dim=ind, keepdims=True) + tol)  # type: ignore
    #     grad = decay * momentum + grad
    #     # Accumulate the gradient for the next iter
    #     momentum += grad
    #
    #     ind = tuple(range(1, len(adv_x_shape)))
    #     grad = grad / (torch.sqrt(torch.sum(grad * grad, dim=ind, keepdim=True)) + tol)
    #     return grad

    def forward_step(self, adv_pert, cropped_signal_batch, person_ids_batch, running_loss):
        cropped_signal_batch = cropped_signal_batch.to(self.cfg['device'])
        person_ids_batch = person_ids_batch.to(self.cfg['device'])
        adv_pert = adv_pert.to(self.cfg['device'])

        adv_batch = self.apply_perturbation(cropped_signal_batch, adv_pert)
        adv_embs = self.model.encode_batch(adv_batch)
        loss = self.loss_fn(adv_embs, person_ids_batch)
        running_loss += loss.item()
        del cropped_signal_batch
        del person_ids_batch
        del adv_pert
        torch.cuda.empty_cache()
        return loss, running_loss, adv_batch

    def forward_step_eval(self, adv_pert, cropped_signal_batch, person_ids_batch, running_loss):
        cropped_signal_batch = cropped_signal_batch.to(self.cfg['device'])
        person_ids_batch = person_ids_batch.to(self.cfg['device'])
        adv_pert = adv_pert.to(self.cfg['device'])

        adv_batch = self.apply_perturbation(cropped_signal_batch, adv_pert)
        adv_embs = self.model.encode_batch(adv_batch)
        loss = self.loss_fn(adv_embs, person_ids_batch)
        running_loss += loss.item()
        # return loss, running_loss
        return loss, running_loss, adv_batch

    def loss_fn(self, adv_embs, speaker_ids):
        gt_embeddings = torch.index_select(self.speaker_embeddings, index=speaker_ids, dim=0)  # .squeeze(-2)
        loss = torch.zeros(1, device=self.cfg['device'])
        for loss_fn, loss_weight in self.loss_fns:
            loss += loss_weight * loss_fn(adv_embs, gt_embeddings).squeeze().mean()
        return loss

    #
    # def projection(self,values):
    #     tol = 10e-8
    #     # values = values.to(self.cfg['device'])
    #     values_tmp = values.reshape(values.shape[0], -1)#.to(self.cfg['device'])
    #     values_tmp = (values_tmp *
    #                   torch.min(
    #                       torch.tensor([1.0], dtype=torch.float32),#.to(self.cfg['device']),
    #                       self.eps_projection / (torch.norm(values_tmp, p=2, dim=1) + tol),#.to(self.cfg['device']),
    #                   ).unsqueeze_(-1)
    #                   )
    #     del values
    #     torch.cuda.empty_cache()
    #     return values_tmp

    def projection(self,values):
        tol = 10e-8
        values_tmp = values.reshape(values.shape[0], -1)
        if self.norm == 2:
            values_tmp = (values_tmp *
                          torch.min(
                              torch.tensor([1.0], dtype=torch.float32),#.to(self.device),
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
        dim = 1 if self.cfg['model_name'] == 'wavlm' else 2
        sims = CosineSimilaritySims(dim=dim)
        curr_speakers=self.speaker_embeddings.clone().cpu()
        running_loss = 0.0
        running_snr = 0.0
        running_sims = 0.0
        similarity_values = []
        similarity_snr = []
        progress_bar = tqdm(enumerate(self.val_loader), desc=f'Eval', total=len(self.val_loader), ncols=150)
        prog_bar_desc = 'Batch Loss: {:.6}, SNR: {:.6}, SIMS: {:.6}'
        for i_batch, (cropped_signal_batch, person_ids_batch) in progress_bar:
            loss, running_loss, adv_cropped_signal_batch = self.forward_step_eval(adv_pert, cropped_signal_batch, person_ids_batch, running_loss)
            adv_snr = calculate_snr_github_direct(adv_cropped_signal_batch.cpu().detach().numpy(),
                                                  cropped_signal_batch.cpu().detach().numpy())
                                                   # self.apply_perturbation(cropped_signal_batch, adv_pert, clip=False).cpu().detach().numpy())# cropped_signal_batch.cpu().detach().numpy())
            # adv_snr = calculator_snr_direct(src_array=cropped_signal_batch,adv_array=self.apply_perturbation(cropped_signal_batch, adv_pert, clip=False))
            # adv_snr = calculator_snr_direct(src_array=cropped_signal_batch, adv_array=adv_cropped_signal_batch)

            temp_emb = self.model.encode_batch(cropped_signal_batch + adv_pert).cpu() # TODO: can be change using adv_cropped_signal_batch?
            temp_labels = torch.index_select(curr_speakers, index=person_ids_batch, dim=0).cpu()
            adv_sims = sims(temp_emb, temp_labels).mean()
            print("\nadv_snr: ",adv_snr)
            pesq_loss = PESQ(cropped_signal_batch, adv_cropped_signal_batch)
            print("pesq: ", pesq_loss)
            # print("adv_snr TYPE:" , type(adv_snr))
            self.similarity_eval_values.append(str(round(adv_sims.item(), 5)))
            self.loss_eval_values.append(str(round(loss.item(), 5)))
            self.snr_eval_values.append(str(round(adv_snr, 5)))
            self.pesq_eval_values.append(str(round(pesq_loss, 5)))
            print("sims: ", adv_sims)
            running_snr += adv_snr
            running_sims += adv_sims
            progress_bar.set_postfix_str(prog_bar_desc.format(running_loss / (i_batch + 1),
                                                              running_snr / (i_batch + 1),
                                                              running_sims / (i_batch + 1)))

        # self.similarity_values.append(similarity_values)
        return running_loss / len(self.val_loader)

    def save_eval(self):
        pass

    def save_files(self,adv_pert):
        self.register_similarity_values(self.counter)
        self.register_snr_values(self.counter)
        self.register_loss_values(self.counter)
        self.register_pesq_values(self.counter)

        save_audio_as_wav(self.cfg['current_dir'], adv_pert.detach(), "perturbation", 'uap_ep100_spk100.wav')
        save_emb_as_npy(self.cfg['current_dir'], adv_pert.detach().numpy(), "perturbation", 'uap_ep100_spk100')
        save_to_pickle(self.cfg['current_dir'], self.speaker_embeddings, "embedding", 'speaker_embeddings')

    def get_current_dir(self):
        return self.cfg['current_dir']

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
    curr_eps = EPS
    config_type = 'Universal'
    cfg = config_dict[config_type]()
    attack = UniversalAttack(cfg)
    attack.generate()
    print("finish main: ")
    torch.cuda.empty_cache()
    gc.collect()
    print("using: ",curr_eps )


if __name__ == '__main__':
    main()
