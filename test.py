import os
import gc
import warnings
import utils
import torch
from utils.data_utils import get_embeddings
from utils.data_utils import load_from_npy
from utils.general import get_instance, calculate_snr_github_direct_pkg
from utils.data_utils import get_person_embedding
from utils.model_utils import load_embedder
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import label_binarize
import math
import matplotlib
from pathlib import Path
import pickle
from configs.eval_config import UniversalAttackEval
import seaborn as sns
import pandas as pd
from models.generator import Generator1D,load_checkpoint
matplotlib.use('Agg')

global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config_dict_eval = {'UniversalTest':UniversalAttackEval}

def load_gan_model(pt_file = "./data/uap_perturbation/JUL_SPK_NUMS_ALL_50spk_VOX_CLOSE_OPEN/Gan/best_epoch_9.pth"):
    model = Generator1D()
    # load GAN model from checkpoint
    if pt_file != 'none':
        print("load model from:", pt_file)
        if os.path.splitext(pt_file)[1] == '.pkl':
            checkpoint_load = torch.load(pt_file)
            model.load_raw_state_dict(checkpoint_load)
        else:
            load_checkpoint(model, pt_file)
    model = model.cuda()
    return model

class Evaluator:
    def __init__(self, config, is_gan) -> None:
        super().__init__()
        self.config = config
        self.embedders = load_embedder(self.config.test_embedder_names, self.config.test_embedder_classes, device=device)  # $$$
        emb_loaders, self.test_loaders = utils.data_utils.get_test_loaders(self.config,
                                                                           self.config.test_celeb_lab.keys())
        self.target_embedding_w_mask, self.target_embedding_wo_mask = {}, {}
        for dataset_name, loader in emb_loaders.items():

            self.target_embedding_wo_mask[dataset_name] = get_person_embedding(self.config, loader,
                                                                               self.config.test_celeb_lab_mapper[dataset_name],
                                                                               self.embedders,
                                                                               device, include_others=False)

        self.gan_model = None
        if is_gan:
            self.gan_model = load_gan_model()

        print("emb_loaders.items(): ", emb_loaders.items())
        self.sims2 = get_instance(self.config.similarity_config_one['module_name'], self.config.similarity_config_one['class_name'])(**self.config.similarity_params)
        self.mask_names = ['Clean', 'AWGN', 'GAN', 'WavLM', 'ECAPA','HuBERT']  # all 50 speakers on libri and vox experiment 'GAN',
        Path(self.config.current_dir).mkdir(parents=True, exist_ok=True)
        utils.general.save_class_to_file(self.config, self.config.current_dir)
        base_path = os.path.join('.', 'data', "uap_perturbation", "JUL_SPK_NUMS_ALL_50spk_VOX_CLOSE_OPEN")
        print("base_uap_path: ",base_path)
        self.wavlm_path = os.path.join(base_path, 'wavlm')
        self.ecapa_path = os.path.join(base_path, 'ecapa')
        self.hubert_path = os.path.join(base_path, 'hubert')

        self.random_perturb_man_awgn = torch.from_numpy(load_from_npy(base_path, 'random', 'awgn')).unsqueeze(0).float().to(device)

        self.adv_perturb_cosim_ep100_wavlm = torch.from_numpy(
            load_from_npy(self.wavlm_path, 'cosim', '100ep_100spk')).to(
            device)
        self.adv_perturb_cosim_ep100_ecapa = torch.from_numpy(load_from_npy(self.ecapa_path, 'cosim', '100ep_100spk')).to(
            device)
        self.adv_perturb_cosim_ep100_hubert = torch.from_numpy(load_from_npy(self.hubert_path, 'cosim', '100ep_100spk')).to(
            device)

    def test(self):
        self.calc_overall_similarity()
        for dataset_name in self.test_loaders.keys():
            similarities_target_without_mask_by_person = self.get_final_similarity_from_disk('without_mask', dataset_name=dataset_name, by_person=True)
            self.calc_similarity_statistics(similarities_target_without_mask_by_person, target_type='without', dataset_name=dataset_name, by_person=True)
            self.plot_sim_box(similarities_target_without_mask_by_person, target_type='without', dataset_name=dataset_name, by_person=True)

    @torch.no_grad()
    def calc_overall_similarity(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)

            for dataset_name, loader in self.test_loaders.items():
                df_without_mask = pd.DataFrame(columns=['y_true', 'y_pred'])
                df_snr = pd.DataFrame(columns=self.mask_names[1:])

                for img_batch, cls_id in tqdm(loader):
                    if len(img_batch.shape) == 1:
                        continue
                    elif img_batch.shape[0] != self.config.test_batch_size:
                        continue
                    img_batch = img_batch.to(device)
                    cls_id = cls_id.to(device).type(torch.int32)

                    # Apply different types of masks
                    img_batch_applied = self.apply_all_masks(img_batch, device)
                    snr_batch_calculate = self.get_all_snrs(img_batch, img_batch_applied)

                    # Get embedding
                    all_embeddings = self.get_all_embeddings(img_batch, img_batch_applied)
                    self.calc_all_similarity(all_embeddings, cls_id, 'without_mask', dataset_name)
                    df_snr = pd.concat([df_snr, self.save_all_snrs(snr_batch_calculate)])
                    df_without_mask = pd.concat([df_without_mask,
                        self.calc_preds(cls_id, all_embeddings, target_type='without_mask', dataset_name=dataset_name)])

                Path(os.path.join(self.config.current_dir, 'saved_preds', dataset_name)).mkdir(parents=True, exist_ok=True)
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

    def apply_all_masks(self, img_batch, device):
        applied_random_perturb_awgn = utils.data_utils.apply_perturbation( img_batch,
                                                                                 self.random_perturb_man_awgn,device)

        applied_uap_gan = self.create_uap_gan(img_batch, device)
        applied_adv_perturb_cosim_ep100_wavlm = utils.data_utils.apply_perturbation(img_batch,
                                                                              self.adv_perturb_cosim_ep100_wavlm,
                                                                              device)

        applied_adv_perturb_cosim_ep100_ecapa = utils.data_utils.apply_perturbation(img_batch,
                                                                                  self.adv_perturb_cosim_ep100_ecapa,
                                                                                  device)

        applied_adv_perturb_cosim_ep100_hubert = utils.data_utils.apply_perturbation(img_batch,
                                                                              self.adv_perturb_cosim_ep100_hubert,
                                                                              device)

        return applied_random_perturb_awgn, applied_uap_gan,   \
               applied_adv_perturb_cosim_ep100_wavlm, applied_adv_perturb_cosim_ep100_ecapa, applied_adv_perturb_cosim_ep100_hubert

    def save_all_snrs(self, all_snr):

        df = pd.DataFrame({key: pd.Series(val) for key, val in all_snr.items()})

        return df



    def get_all_snrs(self, img_batch, img_batch_applied_masks):
        batch_snrs = {}
        adv_perturbs = self.mask_names[1:]
        for perturb_name, img_batch_applied_mask in zip(adv_perturbs, img_batch_applied_masks):
            if perturb_name not in batch_snrs.keys():
                batch_snrs[perturb_name] = []
            batch_snrs[perturb_name].append(torch.round(calculate_snr_github_direct_pkg(img_batch_applied_mask.cpu().detach(),
                                                  img_batch.cpu().detach()),decimals= 5).item())

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
                if len(emb.shape) == 1:
                    emb = np.expand_dims(emb, axis=0).copy()
                    if len(target_embedding.shape) == 1:
                        target_embedding = np.expand_dims(target_embedding, axis=0).copy()
                    print(f"in if emb.shape : {emb.shape} target_embedding: {target_embedding.shape}" )
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
                        print("lab: ", lab)
                        with open(os.path.join(self.config.current_dir, 'saved_similarities', dataset_name, emb_name, lab, sim_type + '_' + mask_name + '.pickle'), 'rb') as f:
                            person_sims = []
                            while True:
                                try:
                                    data = pickle.load(f).values()
                                    person_sims.extend(list(data))
                                except EOFError:
                                    break
                            if len(person_sims) == 0:
                                person_avg_sim = 0
                                print("len(person_sims) + lab : ",lab)
                            else:
                                person_avg_sim = sum(person_sims) / len(person_sims)
                            sims[emb_name][i].append(person_avg_sim)
        return sims


    def calc_preds(self, cls_id, all_embeddings, target_type, dataset_name):
        df = pd.DataFrame(columns=['emb_name', 'mask_name', 'y_true', 'y_pred'])
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

        avg_type = 'person' if by_person else 'image'
        Path(os.path.join(self.config.current_dir, 'final_results', 'stats', 'similarity', dataset_name, target_type)).mkdir(parents=True, exist_ok=True)
        df_mean.to_csv(os.path.join(self.config.current_dir, 'final_results', 'stats', 'similarity', dataset_name, target_type, 'mean_df' + '_' + avg_type + '.csv'), index=False)
        df_std.to_csv(os.path.join(self.config.current_dir, 'final_results', 'stats', 'similarity', dataset_name, target_type, 'std_df' + '_' + avg_type + '.csv'), index=False)

    def create_uap_gan(self, batch_i, noise_scale=1,perturb_size=48000):
        noise_dim = self.gan_model.noise_dim
        batch_size = self.config.test_batch_size
        noise = torch.randn(size=(batch_size, noise_dim))
        pout = self.gan_model.forward(noise.float().cuda()).squeeze().detach().cpu().numpy()
        noise_all = np.concatenate([pout] * int(math.ceil(len(batch_i) / float(len(pout)))))[:len(batch_i)]
        noise_all_extand = np.concatenate([pout for i in range(math.ceil(perturb_size / pout.shape[1]))], 1)
        fake_data = torch.add(torch.from_numpy(noise_all_extand).to(device), batch_i).clip(-1, 1)
        fake_data_norm = fake_data.cpu() / np.abs(fake_data.cpu()).max()
        return fake_data_norm


def main():
    config_type = 'UniversalTest'
    uap_gan = True
    cfg = config_dict_eval[config_type]()
    print('Starting test...', flush=True)
    evaluator = Evaluator(config=cfg, is_gan=uap_gan)
    evaluator.test()
    print('Finished test...', flush=True)
    torch.cuda.empty_cache()
    gc.collect()
    print('Finished cleaning...', flush=True)

if __name__ == '__main__':
    main()
