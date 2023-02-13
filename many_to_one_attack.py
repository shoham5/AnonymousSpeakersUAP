import torchaudio
import torch
import random
from datetime import datetime
from tqdm import tqdm
from statistics import mean
from sklearn.metrics.pairwise import cosine_similarity
import scipy.io.wavfile as wav
from utils.losses_utils import AS2TL2Loss,AS2TL2LossSNR
from pathlib import Path
from utils.data_utils import  get_uap_perturb_npy
from configs.attacks_config import config_dict
# from utils.model_utils import get_lensless_model
from utils.general import load_npy_image_to_tensor, get_instance
# from losses import Loss
import matplotlib.pyplot as plt
from collections import defaultdict
from utils.data_utils import prepare_wav_from_json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from utils.losses_utils import custom_clip
import pandas as pd
from utils.general import calculator_snr_direct, calculator_snr_sec
from constants import S,A,B,SNR_smooth
from models.speechbrain_ecpa import SRModel
from utils.data_utils import read_json,read_npy,read_pickle,write_npy
from utils.data_utils import get_anc_wav_from_json
import sys
import os

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))


# class SimpleAttack:
#     def __init__(self, cfg) -> None:
#         super().__init__()
#         self.cfg = cfg
#
#         self.model = get_lensless_model(cfg)
#         self.loss_func = get_instance(self.cfg['losses_config']['module_name'],
#                                       self.cfg['losses_config']['class_name'])(**self.cfg['loss_params'])
#         self.convert_fn = get_instance(self.cfg['model_config']['model_path'].replace('/', '.') + '.utils', 'convert_to_save_format')
#         self.loss = Loss(self.model, self.loss_func, self.convert_fn)
#         self.cfg['attack_params']['loss_function'] = self.loss.loss_gradient
#         self.attack = get_instance(self.cfg['attack_config']['module_name'],
#                                    self.cfg['attack_config']['class_name'])(**self.cfg['attack_params'])
#
#         self.attack_image_diff = load_npy_image_to_tensor(cfg['attack_img_diff_path'], cfg['device'])
#
#         target_image_orig = load_npy_image_to_tensor(cfg['target_img_orig_path'], cfg['device'])
#         with torch.no_grad():
#             self.pred_target = self.model(target_image_orig).detach()
#
#
#     def generate(self):
#         self.attack.generate(self.attack_image_diff, self.pred_target)


PATH_LIST_DATA ='data/LIBRI/data_lists/d3'

class AttackSRModelCustomLossWithSNR:

    def __init__(self, enroll_utters, batch_size=32,
                 aud_length=3, pertb_length=1, fs=16000, eps=0.0001,
                 attack_name='PGD', loss_name='CLossSNR',
                 const_speaker=False):  # 1#3
        self.classifier_class = SRModel.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
        self.model = SRModel.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb").mods.embedding_model

        self.batch_size = batch_size
        self.aud_length = aud_length
        self.pertb_length = pertb_length
        self.fs = fs
        # fs a.k.a sample-rate - 16kHz
        self.max_audio_len = int(fs * aud_length)

        self.delta = {"PGD": torch.tensor(np.zeros((self.pertb_length * fs * 3), dtype=np.float32), requires_grad=True),
                      "FGSM": torch.tensor(np.random.rand((self.pertb_length * fs)), requires_grad=True),
                      "RPGD": (torch.randn(3 * 16000) / 100000).requires_grad_()}[attack_name]
        self.loss_name = attack_name

        self.loss = {'COSE': torch.nn.CosineEmbeddingLoss(),
                     'CRE': torch.nn.CrossEntropyLoss(),
                     'CLOSS': AS2TL2Loss(enroll_utters=enroll_utters),
                     'CLossSNR': AS2TL2LossSNR(enroll_utters=enroll_utters, const_speaker=const_speaker),
                     }[loss_name]
        self.loss_name = loss_name

        params_to_train = [self.delta]
        params_to_train += list(self.classifier_class.mods.embedding_model.parameters())
        self.apply_sign_optimizer = torch.optim.SGD(params_to_train, lr=0.01)
        self.eps = eps
        self.A = []
        self.B = []
        self.loss_pgd = []
        self.loss_all = []
        self.snr_loss = []

    def forward(self, audio_batch, batch_size):
        #################################################
        ### Configuring the UAP with the legit audios ###
        #################################################

        # we tiling the UAP three times
        # DON'T mess with the UAP length and the tiling, because it's overfitting to our usecase - 3 second UAP, tiled
        # 3 times to 9 seconds audio files

        # print(self.aud_length/ self.pertb_length )
        apply_delta = torch.stack([self.delta.clone() for _ in range(int(self.aud_length / self.pertb_length))])  # 3
        apply_delta = torch.reshape(apply_delta, [-1])

        # copying the UAP batch size times, so we can patch it for each audio file in out training
        apply_delta = torch.broadcast_to(apply_delta, [batch_size, apply_delta.shape[0]])
        new_input = audio_batch + self.eps * apply_delta

        # pass_in = torch.clamp(new_input, -2 ** 15, 2 ** 15 - 1) # del regular clip, adding new clip shoham
        pass_in = custom_clip(audio_batch, new_input, self.eps)
        # pass_in = new_input

        embeddings_batch = self.classifier_class.encode_batch(pass_in)
        return embeddings_batch

    def forward_clean(self, audio_batch, batch_size):
        audio_batch.requires_grad_(True)
        embeddings_batch = self.classifier_class.encode_batch(audio_batch)
        return embeddings_batch

    def get_numpy(self, embeddings_batch):
        return embeddings_batch.detach().numpy()[0]


    def calc__SNR(self, audios_path, uap_path):
        audios = []
        for audio_path in audios_path:
            audio, fs = torchaudio.load(audio_path)  # wavfile.read(os.path.join(audios_path, audio_path))
            audios.append(np.mean(np.abs(audio.detach().numpy())))

        audios_mean_db = np.mean(audios)

        uap, _ = torchaudio.load(uap_path)
        uap_mean_dB = np.mean(np.abs(uap.detach().numpy()))

        SNR = 10 * np.log10(audios_mean_db / uap_mean_dB)
        print(f'{SNR:.2f}')
        return SNR

    def plot_loss_by_parts_A_B(self):


        df_loss = pd.DataFrame(self.A, columns=['sims'])
        print("df loss A: ", df_loss)
        df_loss2 = pd.DataFrame(self.B, columns=['sims'])

        # print(df_loss)
        # print(df_loss2)
        df_loss = df_loss.reset_index()
        df_loss2 = df_loss2.reset_index()

        sns.lineplot(data=df_loss, x="index", y="sims")
        sns.lineplot(data=df_loss2, x="index", y="sims")

        plt.title("Sims vs PGD Steps")
        plt.ylim(-1, 2)
        plt.legend('AB', ncol=2, loc='upper left');
        print("df loss 1-B : ", df_loss2)
        self.A = []
        self.B = []
        del df_loss
        del df_loss2
        plt.show()

        df_loss3 = pd.DataFrame(self.snr_loss, columns=['norm snr'])
        df_loss3 = df_loss3.reset_index()
        sns.lineplot(data=df_loss3, x="index", y="norm snr")
        plt.title("norm snr vs steps")
        plt.ylim(0, 50)
        plt.legend('S', ncol=2, loc='upper left');
        self.snr_loss = []
        del df_loss3
        plt.show()

        df_loss = pd.DataFrame(self.loss_pgd, columns=['loss'])
        df_loss = df_loss.reset_index()
        sns.lineplot(data=df_loss, x="index", y="loss")
        plt.title("Loss pgd vs PGD Steps")
        del df_loss
        plt.show()


    # eps=0.3 epx = 0.0001
    def projected_gradient_descent(self, x, y, signal, num_steps=40, eps=0.3):
        # self.delta.requires_grad = True
        """Performs the projected gradient descent attack on a batch of images."""
        # print("x:" ,x)
        x_adv = x.clone().detach().requires_grad_(True).to(x.device)
        print("x_adv in pgd AttackSRModelCustomLossWithSNR : ", x_adv)
        print("self.delta: ", self.delta)
        for i in tqdm(range(num_steps), desc="Attack step:"):
            # print(f"Step number: {i} out of {num_steps}")
            print("i in pgd : ", i)
            _x_adv = x_adv.clone().detach().requires_grad_(True)
            print("_x_adv: ", _x_adv)

            prediction = self.classifier_class.encode_batch(_x_adv)  # self.forward_clean(_x_adv,1)
            # print("prediction:" ,prediction.shape) $$$
            # print("y:" ,y) $$$
            loss, A_loss, B_loss = self.loss(prediction, y)  # torch.squeeze(y,0)) #CLOSS
            print("A loss:  ", A_loss)
            print("B loss:  ", B_loss)
            print("loss:  ", loss)
            print("_x_adv: ", _x_adv)

            print("num_steps:  ", num_steps)

            self.A.append(A_loss.detach().clone().numpy())

            # print("type Aloss: ", type(A_loss))
            self.B.append(B_loss.detach().clone().numpy())
            # print("B loss 1-loss: ", B_loss)
            # print("loss in pgd 1-loss: ", loss)
            new_input = signal + self.delta.clone() * self.eps  # + SNR_smooth * self.eps
            input_clip = torch.clamp(new_input, -2 ** 15, 2 ** 15 - 1)
            snr_direct = calculator_snr_direct(signal, input_clip)
            # loss = self.loss(torch.squeeze(prediction,0), y,Variable(torch.Tensor([1]))) #COS
            print("snr_direct: ", snr_direct)
            print("loss: ", loss)
            print("input_clip: ", input_clip)
            # snr_square = np.square(snr_direct / 150)
            # print("snr_square: ", snr_square)

            snr_square = snr_direct / 150
            # snr_square = (50 -snr_direct)/100
            print("snr_norm: ", snr_square)
            snr_part = S * snr_square
            loss = loss + snr_part  # (snr_direct / 100 )
            print("loss IN PGD: ", loss)
            # loss = 1 - loss

            self.loss_pgd.append(loss.detach().clone().numpy())
            self.snr_loss.append(snr_direct)

            loss_score = loss
            loss.backward()
            print("loss IN PGD after backward: ", loss)
            print("\nself.delta after backward: ", self.delta)
            self.apply_sign_optimizer.step()
            print("self.delta after opt: ", self.delta)

            with torch.no_grad():
                # Force the gradient step to be a fixed size in a certain norm
                # if step_norm == 'inf':
                gradients = _x_adv.grad.sign()  # * self.eps
                self.delta = self.delta + gradients

                # Untargeted: Gradient ascent on the loss of the correct label w.r.t.
                # the model parameters
                x_adv += self.delta

            # Project back into l_norm ball and correct range
            # if eps_norm == 'inf':
            # Workaround as PyTorch doesn't have elementwise clip

            # x_adv = torch.max(torch.min(x_adv, x + eps), x - eps) # check the diff between
            # x_adv = custom_clip(x,x_adv,self.eps) custom clip, shoham
            x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
            print("self.delta after torch.no_grad() : ", self.delta)

            x_adv = x_adv.clamp(-2 ** 15, 2 ** 15 - 1)  # del regular clip, adding new clip shoham
            print("x_adv after torch.no_grad() : ", x_adv)
            # x_adv = custom_clip(x,x_adv,eps)
            # x_adv = x_adv.clamp(-1,1)
        pertubed_embeddings_batch = self.classifier_class.encode_batch(x_adv)
        print("custom ################################################8")
        self.plot_loss_by_parts_A_B()
        print(" plot")
        return pertubed_embeddings_batch, loss_score  # x_adv


def main():
    # config_type = 'Base'
    # cfg = config_dict[config_type]()
    # attack = SimpleAttack(cfg)
    # attack.generate()

    # read enroll files
    path_obj = Path(PATH_LIST_DATA)
    filename = path_obj.stem
    OUTPUT_PATH = Path(f'{path_obj.parent.parent}/output/{filename}')
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    # enroll_files_dict = read_json('enroll_files_dict.json') # which files chosen

    train_data = read_json(f'{PATH_LIST_DATA}/train_files.json')
    enroll_emb_dict = read_pickle(f'{PATH_LIST_DATA}/enroll_emb_dict')


    train_data_keys = list(set(train_data.keys()))  # should be separate?
    train_data_keys.sort()

    # Configure perturbation epsilon
    # eps=0.0001
    num_epochs = 12  # 3#12#7# 3# 18#3# 3# 12
    num_steps = 3  # 12# 30#12 #5 # 30

    eps = 0.002 / num_steps

    enroll_dict_sec = defaultdict()
    enroll_files_train = defaultdict()

    enroll_dict_list = list(enroll_emb_dict)  # define labels using enroll emb dict
    enroll_dict_list.sort()


    eval_file_data = defaultdict()
    avg_sim_train_per_speaker = []
    avg_snr_per_speaker = []
    avg_snr_per_speaker_custom = []
    avg_sim_eval_per_speaker_with_clip = []
    # avg_sim_eval_per_speaker = []



    print("train_data_keys: ", train_data_keys)
    print("enroll_dict_list: ", enroll_dict_list)
    for speaker_id in tqdm(train_data, desc="Speaker:"):

        trainind_model_custom_loss = AttackSRModelCustomLossWithSNR(enroll_emb_dict, eps=eps, const_speaker=True)
        loss_lst = []
        delta_lst = []
        loss_lst_custom_loss = []
        delta_lst_custom_loss = []
        train_sim_lst_custom = []
        negative_id_speakers = train_data_keys.copy()
        negative_id_speakers.remove(speaker_id)
        print("\nnegative_id_speakers: ", negative_id_speakers)
        print("\nspeaker id: ", speaker_id)

        for epoch in tqdm(range(0, num_epochs), desc="Utterance #:"):

            signal, fs, anc_path = get_anc_wav_from_json(train_data[speaker_id])
            spk_id_labels = enroll_dict_list.index(speaker_id)
            if speaker_id not in enroll_emb_dict.keys():
                print("speaker_id not in enroll")  # enroll_dict_sec[speaker_id] = embd_np
            signal_len = signal.shape[1]
            start_idx = random.randint(0, signal_len - (trainind_model_custom_loss.fs * 3 + 1))
            cropped_signal = signal[0][start_idx: start_idx + (trainind_model_custom_loss.fs * 3)]
            if speaker_id not in enroll_files_train.keys():
                enroll_files_train[speaker_id] = [(anc_path, start_idx)]
            else:
                enroll_files_train[speaker_id].append((anc_path, start_idx))
            embd2_custom_loss = trainind_model_custom_loss.forward_clean(cropped_signal, 1)
            embd2_np_custom_loss = embd2_custom_loss.detach().numpy()[0]
            embd3_custom_loss, loss_score_custom_loss = trainind_model_custom_loss.projected_gradient_descent(
                cropped_signal, torch.tensor([spk_id_labels]), cropped_signal, num_steps=num_steps)
            # embd3_custom_loss, loss_score_custom_loss = trainind_model_custom_loss.cw_gradient(cropped_signal, torch.tensor([spk_id_labels]),cropped_signal, num_steps=num_steps)
            print("\n$$$loss_score_custom_loss in train: ", loss_score_custom_loss)
            embd3_np_custom_loss = embd3_custom_loss.detach().numpy()[0]

            print("**** custom loss ****")
            print(f'Anchor: {speaker_id}')
            print(f"cosine_similarity anchor: {(cosine_similarity(embd2_np_custom_loss, embd2_np_custom_loss)[0][0] + 1) / 2}")
            print(f"cosine_similarity perturbed anchor: {(cosine_similarity(embd3_np_custom_loss, embd2_np_custom_loss)[0][0] + 1) / 2}")

            loss_lst_custom_loss.append(loss_score_custom_loss)
            train_sim_lst_custom.append(
                (cosine_similarity(embd3_np_custom_loss, embd2_np_custom_loss)[0][0] + 1) / 2)  # perturb sim
            # loss_lst_multi.append(loss_score_multi)
            print("\n\ndeltas after utters: ", trainind_model_custom_loss.delta.clone())

        delta_lst_custom_loss.append(trainind_model_custom_loss.delta.clone())
        print("\n\ndeltas after ALL utters: ", trainind_model_custom_loss.delta.clone())

        # save mean sim
        avg_sim_train_per_speaker.append([speaker_id, mean(train_sim_lst_custom)])
        print("speaker id, mean : ", [speaker_id, mean(train_sim_lst_custom)])

        ##############################33
        # SAVE UAP - for last record for each speaker
        print("***CUSTOM MODEL***")
        old_delta_custom = trainind_model_custom_loss.delta.clone() * trainind_model_custom_loss.eps
        old_delta_custom = torch.clamp(old_delta_custom, -2 ** 15, 2 ** 15 - 1)
        date_str_custom = datetime.now().strftime("%d%m%Y_%H:%Mm")  # adding model name
        BASE_FILE_NAME_custom = f"{OUTPUT_PATH}/{speaker_id}_id_{date_str_custom}"

        UAP_FILE_NAME_costum = f'{BASE_FILE_NAME_custom}_UAP_.wav'
        wav.write(f'./{UAP_FILE_NAME_costum}', 16000, old_delta_custom.detach().numpy())
        write_npy(f'./{OUTPUT_PATH}/{speaker_id}_np',old_delta_custom.detach().numpy())

        PERTURBED_FILE_NAME_custom = f'{BASE_FILE_NAME_custom}_PERTUBED.wav'
        pass_in_custom = cropped_signal + trainind_model_custom_loss.delta.clone() * trainind_model_custom_loss.eps
        pass_in_custom = torch.clamp(pass_in_custom, -2 ** 15, 2 ** 15 - 1)
        wav.write(f'./{PERTURBED_FILE_NAME_custom}', 16000, pass_in_custom.detach().numpy())

        CLEAN_FILE_NAME_custom = f'{BASE_FILE_NAME_custom}_CLEAN.wav'
        wav.write(f'./{CLEAN_FILE_NAME_custom}', 16000, cropped_signal.detach().numpy())

        # calc SNR score
        print(f"SPEAKER: {speaker_id}")
        print("BASE_FILE_NAME: ", BASE_FILE_NAME_custom)
        print("PERTURBED_FILE_NAME: ", PERTURBED_FILE_NAME_custom)
        print("UAP_FILE_NAME: ", UAP_FILE_NAME_costum)
        calculator_snr_sec(PERTURBED_FILE_NAME_custom, UAP_FILE_NAME_costum)

        # del df_loss
        print("***CUSTOM MODEL***")
        # visualize loss
        # df_loss = pd.DataFrame([1 - x.detach().numpy() for x in loss_lst_custom_loss], columns =['loss'])
        df_loss = pd.DataFrame([x.detach().numpy() for x in loss_lst_custom_loss], columns=['loss'])
        df_loss = df_loss.reset_index()
        plt.title(f"{speaker_id}_Custom model loss vs Utterences index ")
        plt.ylim(-1, 1)
        sns.lineplot(data=df_loss, x="index", y="loss")
        plt.show()









# ##################3

#     perturbs = get_uap_perturb_npy(OUTPUT_PATH)
def evaluate_uap_per_speaker(num_eval = 5,num_steps=3, eps = 0.002):
    path_obj = Path(PATH_LIST_DATA)
    filename = path_obj.stem
    OUTPUT_PATH = Path(f'{path_obj.parent.parent}/output/{filename}')
    test_data = read_json(f'{PATH_LIST_DATA}/test_files.json')
    enroll_emb_dict = read_pickle(f'{PATH_LIST_DATA}/enroll_emb_dict')

    avg_snr_per_speaker_custom = []
    avg_sim_eval_per_speaker_with_clip = []

    num_epochs = 12  # 3#12#7# 3# 18#3# 3# 12
    num_steps = 3  # 12# 30#12 #5 # 30
    num_eval = num_eval
    num_steps = num_steps  # 12# 30#12 #5 # 30
    eps = eps / num_steps

    for speaker_id in tqdm(test_data.keys(), desc="Speaker:"):
        # Evaluate UAP custom
        evaluate_model_custom_loss = AttackSRModelCustomLossWithSNR(enroll_emb_dict, eps=eps, const_speaker=True)
        uap_delta = read_npy(f'{OUTPUT_PATH}/{speaker_id}_np')
        test_speaker_paths = test_data[speaker_id]
        print("train id evaluate custom: ", speaker_id)
        id_lst_custom = []
        id_lst_custom_snr = []
        id_lst_custom_perturb_clip_avg = []

        for i in range(num_eval):
            cropped_signal, test_path, start_idx = prepare_wav_from_json(test_speaker_paths)
            embd = evaluate_model_custom_loss.forward_clean(cropped_signal, 1)
            embd_np = embd.detach().numpy()[0]
            new_input_clip = cropped_signal + uap_delta #  * evaluate_model_custom_loss.eps
            file_clip = custom_clip(cropped_signal, new_input_clip, 0.3)
            file_clip2 = torch.clamp(new_input_clip, -2 ** 15, 2 ** 15 - 1)
            embd_clip = evaluate_model_custom_loss.classifier_class.encode_batch(file_clip)
            embd3_clip_np = embd_clip.detach().numpy()[0]
            clean_sim = (cosine_similarity(embd_np, embd_np)[0][0] + 1) / 2
            perturbed_sim_custom_clip = (cosine_similarity(embd3_clip_np, embd_np)[0][0] + 1) / 2
            print("#### perturbed_sim_custom_clip ####  ", perturbed_sim_custom_clip)
            snr_direct = calculator_snr_direct(cropped_signal, new_input_clip)
            id_lst_custom = id_lst_custom + [[speaker_id, test_path.split("/")[-1], clean_sim, "clean"]]
            id_lst_custom = id_lst_custom + [
                [speaker_id, test_path.split("/")[-1], perturbed_sim_custom_clip, "perturbed"]]
            # id_lst_custom_perturb_avg.append(perturbed_sim)
            id_lst_custom_snr.append(snr_direct)
            id_lst_custom_perturb_clip_avg.append(perturbed_sim_custom_clip)
            # print("id_lst: ", id_lst)

        df = pd.DataFrame(id_lst_custom, columns=['speaker_id', 'tested_uttr', 'sim_score', 'sim_type'])
        df.to_csv(
            f"{OUTPUT_PATH}/uap_res_{evaluate_model_custom_loss.eps}_epsilon_{num_epochs}_{num_steps}_costum_{speaker_id}.csv")
        print(
            f"{OUTPUT_PATH}/uap_res_{evaluate_model_custom_loss.eps}_epsilon_custom_{num_epochs}_{num_steps}_{speaker_id}.csv")
        # print(f"mean speaker {speaker_id}: {mean(id_lst_custom_perturb_avg)}")
        avg_sim_eval_per_speaker_with_clip.append([speaker_id, mean(id_lst_custom_perturb_clip_avg)])
        avg_snr_per_speaker_custom.append([speaker_id, mean(id_lst_custom_snr)])  #
        sns.boxplot(x="sim_type", y="sim_score", data=df)
        plt.title(f"{speaker_id}_similarity_eval ")
        plt.show()

#     attack = AttackSRModelCustomLossWithSNR()


if __name__ == '__main__':
    main()
    # evaluate_uap_per_speaker()