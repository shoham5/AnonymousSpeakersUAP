import sys
import os

import gc
import warnings
import utils
import torch
from collections import defaultdict
from configs.attacks_config import config_dict
from utils.model_utils import get_speaker_model
from utils.data_utils import get_embeddings, get_loaders
from utils.data_utils import load_from_npy
from utils.general import get_instance, save_config_to_file,calculator_snr_per_signal, calculator_snr_direct, get_pert, PESQ, calculate_l2
from utils.data_utils import save_audio_as_wav, create_dirs_not_exist, save_emb_as_npy, get_test_loaders,  get_person_embedding
from utils.losses_utils import WSNRLoss,SNRLoss,custom_clip
from utils.model_utils import load_embedder
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from utils.general import save_config_to_file
from sklearn.preprocessing import label_binarize

import matplotlib
from pathlib import Path
import pickle
from configs.eval_config import UniversalAttackEval
import seaborn as sns
import pandas as pd
matplotlib.use('Agg')

global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



config_dict_eval = {'UniversalTest':UniversalAttackEval}

class Evaluator:
    def __init__(self, config, best_perturb, prev_directory) -> None:
        super().__init__()
        # self.cfg = config
        self.config = config # dataset, model
        self.best_patch = best_perturb

        # self.prev_files_loc = prev_directory
        self.embedders = load_embedder(self.config.test_embedder_names, self.config.test_embedder_classes, device=device)  # $$$
        emb_loaders, self.test_loaders = utils.data_utils.get_test_loaders(self.config,
                                                                           self.config.test_celeb_lab.keys())  # $$$

        self.target_embedding_w_mask, self.target_embedding_wo_mask = {}, {}
        for dataset_name, loader in emb_loaders.items():
            # self.target_embedding_w_mask[dataset_name] = get_person_embedding(self.config, loader,
            #                                                                                self.config.test_celeb_lab_mapper[dataset_name],
            #                                                                                self.embedders,
            #                                                                                device, include_others=False)
            # get_person_embedding(self.config, loader,
            #                                                                   self.config.test_celeb_lab_mapper[
            #                                                                       dataset_name],
            #                                                                   self.location_extractor,
            #                                                                   self.fxz_projector, self.embedders,
            #                                                                   device, include_others=True)
            self.target_embedding_wo_mask[dataset_name] = get_person_embedding(self.config, loader,
                                                                               self.config.test_celeb_lab_mapper[dataset_name],
                                                                               self.embedders,
                                                                               device, include_others=False)


        # self.random_mask_t = utils.load_mask(self.config, self.config.random_mask_path, device)  # $$$ add perturbation according to previus known
        # self.perturbations = load_perturbations_from_files()
        self.sims2 = get_instance(self.config.similarity_config_one['module_name'], self.config.similarity_config_one['class_name'])(**self.config.similarity_params)
        self.mask_names = ['Clean','AWGN','UNIFORM','SIM100','SNR100','PESQ100','EPS']
            #,'SI100','SN100','PESQ100','com']
                           # 'AWGN-M','AWGN-W','UNI-M','UNI-W',
                           #'COS10' ,'COS50',
                           # 'COS100' #,
                           #'SNR10','SNR50','SNR100',
                           #'PESQ10',#'PESQ-50','PESQ-100'
                           # ]

        Path(self.config.current_dir).mkdir(parents=True, exist_ok=True)
        # save_config_to_file(self.config, self.config.current_dir)
        utils.general.save_class_to_file(self.config, self.config.current_dir)
        # Test options
        # self.masks_path = os.path.join('.', 'data', 'uap_perturbation_ecapa')  # change to perturb
        self.xvector_path = os.path.join('.', 'data', 'uap_perturbation','xvector')  # change to perturb
        self.ecapa_path = os.path.join('.', 'data', 'uap_perturbation', 'ecapa')  # change to perturb
        # load_from_npy

        self.random_perturb_man_awgn = torch.from_numpy(load_from_npy(self.ecapa_path, 'random', 'awgn')).unsqueeze(0).to(device)
        self.random_perturb_man_uniform = torch.from_numpy(
            load_from_npy(self.ecapa_path, 'random', 'uniform')).unsqueeze(0).to(device)

        self.adv_perturb_cosim_ep100 = torch.from_numpy(load_from_npy(self.ecapa_path, 'cosim', '100ep_100spk')).to(
            device)
        self.adv_perturb_snr_cosim_ep100 = torch.from_numpy(load_from_npy(self.ecapa_path, 'snr', '100ep_100spk')).to(
            device)
        self.adv_perturb_pesq_snr_cosim_ep100 = torch.from_numpy(load_from_npy(self.ecapa_path, 'pesq_snr', '100ep_100spk')).to(device)

        # self.random_perturb_woman_awgn = torch.from_numpy(load_from_npy(self.masks_path, 'random', 'awgn_woman')).unsqueeze(0).to(device)
        # self.random_perturb_woman_uniform = torch.from_numpy(load_from_npy(self.masks_path, 'random', 'uniform_woman')).unsqueeze(0).to(device)

        # self.adv_perturb_cosim_ep10 = torch.from_numpy(load_from_npy(self.masks_path, 'cosim', '10ep_100spk')).to(device)
        # self.adv_perturb_cosim_ep50 = torch.from_numpy(load_from_npy(self.masks_path, 'cosim', '50ep_100spk')).to(device)


        # self.adv_perturb_snr_cosim_ep10 = torch.from_numpy(load_from_npy(self.masks_path, 'snr', '10ep_100spk')).to(device)
        # self.adv_perturb_snr_cosim_ep50 = torch.from_numpy(load_from_npy(self.masks_path, 'snr', '50ep_100spk')).to(device)


        # self.adv_perturb_pesq_snr_cosim_ep10 = torch.from_numpy(load_from_npy(self.masks_path, 'pesq_snr', '10ep_100spk')).to(device)
        # self.adv_perturb_pesq_snr_cosim_ep50 = torch.from_numpy(load_from_npy(self.masks_path, 'pesq_snr', '50ep_100spk')).to(device)

        self.adv_perturb_cosim_ep100_xvector = torch.from_numpy(load_from_npy(self.xvector_path, 'cosim', '100ep_100spk')).to(
            device)
        self.adv_perturb_snr_cosim_ep100_xector = torch.from_numpy(load_from_npy(self.xvector_path, 'snr', '100ep_100spk')).to(
            device)
        self.adv_perturb_pesq_snr_cosim_ep100_xvector = torch.from_numpy(load_from_npy(self.xvector_path, 'pesq_snr', '100ep_100spk')).to(device)

        #
        # self._update_current_dir()
        # save_config_to_file(self.cfg, self.cfg['current_dir'])
        #
        #
        #
        # self.cfg.attack_type = self.__class__.__name__  # for logging purposes when inheriting from another class
        # self.model = get_speaker_model(config)
        # self.is_first = True
        #
        # self.train_loader, self.val_loader, _ = get_loaders(self.cfg['loader_params'],
        #                                                     self.cfg['dataset_config'],
        #                                                     splits_to_load=['train', 'validation'])
        #
        # self.speaker_embeddings = get_embeddings(self.model,
        #                                          self.train_loader,
        #                                          self.cfg['dataset_config']['speaker_labels_mapper'].keys(),
        #                                          self.cfg['device'])
        #
        # self.loss_fns = []
        # for i, (loss_cfg, loss_func_params) in enumerate(
        #         zip(self.cfg['losses_config'], self.cfg['loss_func_params'].values())):
        #     loss_func = get_instance(loss_cfg['module_name'],
        #                              loss_cfg['class_name'])(**loss_func_params)
        #     self.loss_fns.append((loss_func, self.cfg['loss_params']['weights'][i]))
        #
        #
        #
        # self.snr_loss = WSNRLoss()
        # self.loss_test_values = []
        # self.similarity_test_values = []
        # self.snr_test_values = []
        # self.pesq_test_values = []
        #
        # self.counter = 0
        #
        # df_cols = ['first', 'sec', 'third']
        # pd.DataFrame(columns=df_cols).to_csv(os.path.join(self.cfg['attack_params'], 'sims_results.csv'))
        # pd.DataFrame(columns=df_cols).to_csv(os.path.join(self.cfg['attack_params'], 'snr_results.csv'))
        # pd.DataFrame(columns=df_cols).to_csv(os.path.join(self.cfg['attack_params'], 'pesq_results.csv'))
        # pd.DataFrame(columns=[f'iter {str(iteration)}' for iteration in range(self.cfg['attack_params']['max_iter'])]). \
        #     to_csv(os.path.join(self.cfg['current_dir'], 'loss_results.csv'))
        # create_dirs_not_exist(self.cfg['current_dir'], ["perturbation", "embedding", "adversarial"])



    def test(self):
        self.calc_overall_similarity()
        for dataset_name in self.test_loaders.keys():
            # similarities_target_with_mask_by_person = self.get_final_similarity_from_disk('with_mask', dataset_name=dataset_name, by_person=True)
            similarities_target_without_mask_by_person = self.get_final_similarity_from_disk('without_mask', dataset_name=dataset_name, by_person=True)
            # self.calc_similarity_statistics(similarities_target_with_mask_by_person, target_type='with', dataset_name=dataset_name, by_person=True)
            self.calc_similarity_statistics(similarities_target_without_mask_by_person, target_type='without', dataset_name=dataset_name, by_person=True)
            # self.plot_sim_box(similarities_target_with_mask_by_person, target_type='with', dataset_name=dataset_name, by_person=True)
            self.plot_sim_box(similarities_target_without_mask_by_person, target_type='without', dataset_name=dataset_name, by_person=True)

    @torch.no_grad()
    def calc_overall_similarity(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)

            adv_patch = self.best_patch.to(device)
            for dataset_name, loader in self.test_loaders.items():
                df_with_mask = pd.DataFrame(columns=['y_true', 'y_pred'])
                df_without_mask = pd.DataFrame(columns=['y_true', 'y_pred'])
                df_snr = pd.DataFrame(columns=self.mask_names[1:])

                for img_batch, cls_id in tqdm(loader):
                    img_batch = img_batch.to(device)
                    cls_id = cls_id.to(device).type(torch.int32)

                    # Apply different types of masks
                    img_batch_applied = self.apply_all_masks(img_batch, adv_patch,device)
                    snr_batch_calculate = self.get_all_snrs(img_batch, img_batch_applied)

                    # Get embedding
                    all_embeddings = self.get_all_embeddings(img_batch, img_batch_applied)

                    # self.calc_all_similarity(all_embeddings, cls_id, 'with_mask', dataset_name)
                    self.calc_all_similarity(all_embeddings, cls_id, 'without_mask', dataset_name) # TODO: change to ___

                    # df_with_mask = df_with_mask.append(self.calc_preds(cls_id, all_embeddings, target_type='with_mask', dataset_name=dataset_name))
                    # df_without_mask = df_without_mask.append(self.calc_preds(cls_id, all_embeddings, target_type='without_mask', dataset_name=dataset_name))
                    # df_snr = pd.concat([df_snr, self.save_all_snrs(all_snr=snr_batch_calculate, target_type='without_mask',
                    #                                              dataset_name=dataset_name)])

                    df_snr = pd.concat([df_snr, self.save_all_snrs(snr_batch_calculate)])

                    df_without_mask = pd.concat([df_without_mask,
                        self.calc_preds(cls_id, all_embeddings, target_type='without_mask', dataset_name=dataset_name)])

                Path(os.path.join(self.config.current_dir, 'saved_preds', dataset_name)).mkdir(parents=True, exist_ok=True)
                # df_with_mask.to_csv(os.path.join(self.config.current_dir, 'saved_preds', dataset_name, 'preds_with_mask.csv'), index=False)
                df_without_mask.to_csv(os.path.join(self.config.current_dir, 'saved_preds', dataset_name, 'preds_without_mask.csv'), index=False)
                Path(os.path.join(self.config.current_dir, 'final_results', 'stats', 'snr', dataset_name,
                                  'without')).mkdir(parents=True, exist_ok=True)
                df_snr.to_csv(
                    os.path.join(self.config.current_dir, 'final_results', 'stats','snr', dataset_name,'without', 'mean_snrs_per_batch.csv'),
                    index=False)
                del df_snr
                del df_without_mask
                torch.cuda.empty_cache()


    def plot_sim_box(self, similarities, target_type, dataset_name, by_person=False):
        Path(os.path.join(self.config.current_dir, 'final_results', 'sim-boxes', dataset_name, target_type)).mkdir(parents=True, exist_ok=True)
        for emb_name in self.config.test_embedder_names:
            sim_df = pd.DataFrame()
            for i in range(len(similarities[emb_name])):
                sim_df[self.mask_names[i]] = similarities[emb_name][i]
            sorted_index = sim_df.mean().sort_values(ascending=False).index
            sim_df_sorted = sim_df[sorted_index]
            sns.boxplot(data=sim_df_sorted, order=self.mask_names).set_title('Similarities for Different Perturbations')
            plt.xlabel('Perturb Type')
            plt.ylabel('Similarity [0,1]')
            avg_type = 'person' if by_person else 'image'
            plt.savefig(os.path.join(self.config.current_dir, 'final_results', 'sim-boxes', dataset_name, target_type, avg_type + '_' + emb_name + '.png'))
            plt.close()

    def write_similarities_to_disk(self, sims, cls_ids, sim_type, emb_name, dataset_name):
        Path(os.path.join(self.config.current_dir, 'saved_similarities', dataset_name, emb_name)).mkdir(parents=True, exist_ok=True)
        for i, lab in self.config.test_celeb_lab_mapper[dataset_name].items():
            Path(os.path.join(self.config.current_dir, 'saved_similarities', dataset_name, emb_name, lab)).mkdir(parents=True, exist_ok=True)
            for similarity, mask_name in zip(sims, self.mask_names):
                # if len(similarity.shape) == 1:
                #     np.round_(torch.broadcast_to(torch.from_numpy(similarity), cls_ids.shape)[
                #                   cls_ids.cpu().numpy() == i].tolist(), decimals=5)
                sim = similarity[cls_ids.cpu().numpy() == i].tolist()
                sim = {img_name: s for img_name, s in enumerate(sim)}
                with open(os.path.join(self.config.current_dir, 'saved_similarities', dataset_name, emb_name, lab, sim_type + '_' + mask_name + '.pickle'), 'ab') as f:
                    pickle.dump(sim, f)
        for similarity, mask_name in zip(sims, self.mask_names):
            sim = {img_name: s for img_name, s in enumerate(similarity.tolist())}
            with open(os.path.join(self.config.current_dir, 'saved_similarities', dataset_name, emb_name, sim_type + '_' + mask_name + '.pickle'), 'ab') as f:
                pickle.dump(sim, f)

    def apply_all_masks(self, img_batch, adv_patch, device):
        # applied_adv = utils.data_utils.apply_perturbation(img_batch, adv_patch, device)
        applied_random_perturb_awgn = utils.data_utils.apply_perturbation( img_batch,
                                                                                 self.random_perturb_man_awgn,device)
        applied_random_perturb_uniform = utils.data_utils.apply_perturbation(img_batch,
                                                                             self.random_perturb_man_uniform, device)

        applied_adv_perturb_cosim_ep100 = utils.data_utils.apply_perturbation(img_batch,
                                                                                  self.adv_perturb_cosim_ep100,
                                                                                  device)
        applied_adv_perturb_snr_cosim_ep100 = utils.data_utils.apply_perturbation(img_batch,
                                                                                      self.adv_perturb_snr_cosim_ep100,
                                                                                      device)
        applied_adv_perturb_pesq_snr_cosim_ep100 = utils.data_utils.apply_perturbation(img_batch,
                                                                                      self.adv_perturb_pesq_snr_cosim_ep100,
                                                                                      device)

        applied_adv_perturb_cosim_ep100_eps_com = utils.data_utils.apply_perturbation(img_batch,
                                                                                      self.adv_perturb_cosim_ep100,
                                                                                      device,
                                                                                      eps=(3 / 2))

        applied_adv_perturb_cosim_ep100_xvec = utils.data_utils.apply_perturbation(img_batch,
                                                                              self.adv_perturb_cosim_ep100_xvector,
                                                                              device)
        applied_adv_perturb_snr_cosim_ep100_xvec = utils.data_utils.apply_perturbation(img_batch,
                                                                                  self.adv_perturb_snr_cosim_ep100_xector,
                                                                                  device)
        applied_adv_perturb_pesq_snr_cosim_ep100_xvec = utils.data_utils.apply_perturbation(img_batch,
                                                                                      self.adv_perturb_pesq_snr_cosim_ep100_xvector,
                                                                                      device)



        applied_adv_perturb_cosim_ep100_eps_xvec = utils.data_utils.apply_perturbation(img_batch,
                                                                                      self.adv_perturb_snr_cosim_ep100_xector,
                                                                                      device,
                                                                                      eps=(3 / 2))



        # applied_random_perturb_woman_awgn = utils.data_utils.apply_perturbation(img_batch,
        #                                                                          self.random_perturb_woman_awgn, device)

        # applied_random_perturb_woman_uniform = utils.data_utils.apply_perturbation(img_batch,
        #                                                                          self.random_perturb_woman_uniform, device)
        # applied_adv_perturb_cosim_ep10 = utils.data_utils.apply_perturbation(img_batch,
        #                                                                          self.adv_perturb_cosim_ep10, device)
        # applied_adv_perturb_cosim_ep50 = utils.data_utils.apply_perturbation(img_batch,
        #                                                                          self.adv_perturb_cosim_ep50, device)

        # applied_adv_perturb_snr_cosim_ep10 = utils.data_utils.apply_perturbation(img_batch,
        #                                                                          self.adv_perturb_snr_cosim_ep10, device)
        # applied_adv_perturb_snr_cosim_ep50 = utils.data_utils.apply_perturbation(img_batch,
        #                                                                          self.adv_perturb_snr_cosim_ep50,
        #                                                                          device)





        return applied_random_perturb_awgn, applied_random_perturb_uniform, \
               applied_adv_perturb_cosim_ep100_xvec, applied_adv_perturb_snr_cosim_ep100_xvec, applied_adv_perturb_pesq_snr_cosim_ep100_xvec, \
               applied_adv_perturb_cosim_ep100_eps_xvec

               #applied_adv_perturb_cosim_ep100_xvec,applied_adv_perturb_snr_cosim_ep100_xvec,applied_adv_perturb_pesq_snr_cosim_ep100_xvec,\
               #applied_adv_perturb_cosim_ep100_eps_com

               #applied_adv_perturb_cosim_ep100, applied_adv_perturb_snr_cosim_ep100, applied_adv_perturb_pesq_snr_cosim_ep100, \
    # applied_adv_perturb_cosim_ep10, applied_adv_perturb_cosim_ep50, applied_adv_perturb_cosim_ep100, applied_adv_perturb_snr_cosim_ep10 ,\
               # applied_adv_perturb_snr_cosim_ep50, applied_adv_perturb_snr_cosim_ep100, applied_adv_perturb_pesq_snr_cosim_ep10

    # applied_random_perturb_man_awgn, applied_random_perturb_woman_awgn, applied_random_perturb_man_uniform, applied_random_perturb_woman_uniform,\

    def save_all_snrs(self, all_snr):
        # Path(os.path.join(self.config.current_dir, 'final_results', 'stats', dataset_name, target_type)).mkdir(
        #     parents=True, exist_ok=True)
        df = pd.DataFrame({key: pd.Series(val) for key, val in all_snr.items()})
        # df = df.reset_index(drop=True)
        # print("df: ", df)

        return df



    def get_all_snrs(self, img_batch, img_batch_applied_masks):
        batch_snrs = {}
        adv_perturbs = self.mask_names[1:]
        for perturb_name, img_batch_applied_mask in zip(adv_perturbs, img_batch_applied_masks):
            # batch_snrs[perturb_name].append(calculator_snr_per_signal(img_batch, img_batch_applied_mask))
            if perturb_name not in batch_snrs.keys():
                batch_snrs[perturb_name] = []
            batch_snrs[perturb_name].append(round(calculator_snr_direct(img_batch, img_batch_applied_mask),5))

        return batch_snrs


    def get_all_embeddings(self, img_batch, img_batch_applied_masks):
        batch_embs = {}
        for emb_name, emb_model in self.embedders.items():
            batch_embs[emb_name] = [emb_model.encode_batch(img_batch.to(device)).squeeze().cpu().numpy()]
            for img_batch_applied_mask in img_batch_applied_masks:
                batch_embs[emb_name].append(emb_model.encode_batch(img_batch_applied_mask.to(device)).squeeze().cpu().numpy())
        return batch_embs

    def calc_all_similarity(self, all_embeddings, cls_id, target_type, dataset_name):
        for emb_name in self.config.test_embedder_names:
            target = self.target_embedding_w_mask[dataset_name][emb_name] if target_type == 'with_mask' else self.target_embedding_wo_mask[dataset_name][emb_name]
            target_embedding = torch.index_select(target, index=cls_id, dim=0).squeeze().cpu().numpy()
            simcosine = torch.nn.CosineSimilarity(dim=1)
            sims = []
            sims_torch = []
            for emb in all_embeddings[emb_name]:
                # sims_torch.append(simcosine(torch.from_numpy(emb), torch.from_numpy(target_embedding)))
                # self.sims2(torch.from_numpy(emb).unsqueeze(0), torch.from_numpy(target_embedding).unsqueeze(0))
                # if len(emb.shape) == 1:
                #     emb = np.expand_dims(emb, axis=0).copy()
                #     print("in emb.shape")
                sims.append((np.diag(cosine_similarity(emb, target_embedding)) +1) / 2)
            self.write_similarities_to_disk(sims, cls_id, sim_type=target_type, emb_name=emb_name, dataset_name=dataset_name)

    def get_final_similarity_from_disk(self, sim_type, dataset_name, by_person=False):
        sims = {}
        for emb_name in self.config.test_embedder_names:
            sims[emb_name] = []
            for i, mask_name in enumerate(self.mask_names):
                if not by_person:
                    with open(os.path.join(self.config.current_dir, 'saved_similarities', dataset_name, emb_name, sim_type + '_' + mask_name + '.pickle'), 'rb') as f:
                        sims[emb_name].append([])
                        while True:
                            try:
                                data = pickle.load(f).values()
                                sims[emb_name][i].extend(list(data))
                            except EOFError:
                                break
                else:
                    sims[emb_name].append([])
                    for lab in self.config.test_celeb_lab[dataset_name]:
                        with open(os.path.join(self.config.current_dir, 'saved_similarities', dataset_name, emb_name, lab, sim_type + '_' + mask_name + '.pickle'), 'rb') as f:
                            person_sims = []
                            while True:
                                try:
                                    data = pickle.load(f).values()
                                    person_sims.extend(list(data))
                                except EOFError:
                                    break
                            person_avg_sim = sum(person_sims) / len(person_sims)
                            sims[emb_name][i].append(person_avg_sim)
        return sims


    def calc_preds(self, cls_id, all_embeddings, target_type, dataset_name):
        df = pd.DataFrame(columns=['emb_name', 'mask_name', 'y_true', 'y_pred'])
        # df2 = pd.DataFrame(columns=['emb_name', 'mask_name', 'y_true', 'y_pred'])
        class_labels = list(range(0, len(self.config.test_celeb_lab_mapper[dataset_name])))
        y_true = label_binarize(cls_id.cpu().numpy(), classes=class_labels)
        y_true = [lab.tolist() for lab in y_true]
        for emb_name in self.config.test_embedder_names:
            target_embedding = self.target_embedding_w_mask[dataset_name][emb_name] \
                if target_type == 'with_mask' else self.target_embedding_wo_mask[dataset_name][emb_name]
            target_embedding = target_embedding.cpu().numpy()
            for i, mask_name in enumerate(self.mask_names):
                emb = all_embeddings[emb_name][i]
                cos_sim = cosine_similarity(emb, target_embedding)
                y_pred = [lab.tolist() for lab in cos_sim]
                new_rows = pd.DataFrame({
                    'emb_name': [emb_name] * len(y_true),
                    'mask_name': [mask_name] * len(y_true),
                    'y_true': y_true,
                    'y_pred': y_pred
                })
                # df2 = df.append(new_rows, ignore_index=True)
                df = pd.concat([df,new_rows])
        return df

    def calc_similarity_statistics(self, sim_dict, target_type, dataset_name, by_person=False):
        df_mean = pd.DataFrame(columns=['emb_name'] + self.mask_names)
        df_std = pd.DataFrame(columns=['emb_name'] + self.mask_names)
        for emb_name, sim_values in sim_dict.items():
            sim_values = np.array([np.array(lst) for lst in sim_values])
            sim_mean = np.round(sim_values.mean(axis=1), decimals=3)
            sim_std = np.round(sim_values.std(axis=1), decimals=3)

            df_mean = pd.concat([df_mean,pd.Series([emb_name] + sim_mean.tolist(), index=df_mean.columns).to_frame().T],axis=0,ignore_index=True)
            df_std = pd.concat([df_std, pd.Series([emb_name] + sim_std.tolist(), index=df_std.columns).to_frame().T], axis=0,
                ignore_index=True)

            # df_mean = df_mean.append(pd.Series([emb_name] + sim_mean.tolist(), index=df_mean.columns), ignore_index=True)
            # df_std = df_std.append(pd.Series([emb_name] + sim_std.tolist(), index=df_std.columns), ignore_index=True)

        avg_type = 'person' if by_person else 'image'
        Path(os.path.join(self.config.current_dir, 'final_results', 'stats', 'similarity', dataset_name, target_type)).mkdir(parents=True, exist_ok=True)
        df_mean.to_csv(os.path.join(self.config.current_dir, 'final_results', 'stats', 'similarity', dataset_name, target_type, 'mean_df' + '_' + avg_type + '.csv'), index=False)
        df_std.to_csv(os.path.join(self.config.current_dir, 'final_results', 'stats', 'similarity', dataset_name, target_type, 'std_df' + '_' + avg_type + '.csv'), index=False)


def main():
    config_type = 'UniversalTest'
    # pre_train_path = ""
    pre_train_exper = "/sise/home/hanina/speaker_attack/experiments/February/27-02-2023_161447_660590/"
    cfg = config_dict_eval[config_type]()
    uap_perturbation = torch.from_numpy(load_from_npy(pre_train_exper, "perturbation", 'uap_ep50_spk100'))
    print('Starting test...', flush=True)
    evaluator = Evaluator(config=cfg, best_perturb=uap_perturbation, prev_directory=pre_train_exper)
    evaluator.test()

    print('Finished test...', flush=True)
    torch.cuda.empty_cache()
    gc.collect()
    print('Finished cleaning...', flush=True)

if __name__ == '__main__':
    main()
