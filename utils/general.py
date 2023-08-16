import random
import numpy as np
import torch
import importlib
import os
import json
import pathlib
import inspect
import torchaudio
from pypesq import pesq
# import cv2
# from sklearn import metrics
import matplotlib as plt
import seaborn as sns
import scipy.io.wavfile as wav
from pathlib import Path
from torchmetrics import SignalNoiseRatio
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_seeds(seed=0):
    # Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
    # cudnn seed 0 settings are slower and more reproducible, else faster and less reproducible
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    # np.random.seed(0) # change by shoham
    torch.manual_seed(seed)
    cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 0 else (True, False)


def preplot(image):
    image = image * 255
    image = np.transpose(image, (1, 2, 0))
    out_image = np.flipud(np.clip(image, 0, 255))
    return out_image[60:, 62:-60, :]


def process_imgs(imgs, x1, x2, y1, y2):
    imgs = torch.flip(imgs, [2, 3])
    imgs = imgs[:, [2, 1, 0]]
    return crop_images(imgs, x1, x2, y1, y2)


def crop_images(images, x1, x2, y1, y2):
    return images[:, :, x1:x2, y1:y2]


def intersect_dicts(da, db, exclude=()):
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
    return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}


def load_npy_image_to_tensor(image_path, device):
    image_np = np.load(image_path).transpose((2, 0, 1))
    image_t = torch.from_numpy(image_np).unsqueeze(0)
    image_t = image_t.to(device)
    return image_t


def get_instance(module_name, instance_name):
    module = importlib.import_module(module_name)
    obj = getattr(module, instance_name)
    return obj


def save_file_to_npy(current_folder, file_name, data):
    signal_path = os.path.join(current_folder, file_name)
    np.save(f'{signal_path}.npy',data)


def save_signal_to_wav(current_folder, file_name, np_signal):
    signal_path = os.path.join(current_folder, file_name)
    wav.write(f'./{signal_path}.wav', 16000, np_signal)


def save_config_to_file(config, current_folder):
    with open(os.path.join(current_folder, 'config.json'), 'w') as config_file:
        d = dict(vars(config))
        d_new = check_dict(d)
        d_new.pop('scheduler_factory')
        json.dump(d_new, config_file)


def save_class_to_file(config, current_folder):
    with open(os.path.join(current_folder, 'config.json'), 'w') as config_file:
        d = dict(vars(config))
        # d.pop('scheduler_factory')
        json.dump(d, config_file)

def check_dict(d):
    d_new = {}
    for key, value in d.items():
        if isinstance(value, dict):
            value = check_dict(value)
        if isinstance(value, list):
            for i, val in enumerate(value):
                if isinstance(val, dict):
                    value[i] = check_dict(val)
        elif isinstance(value, (pathlib.PosixPath, torch.device)):
            value = str(value)
        elif isinstance(value, torch.Tensor):
            continue
        elif inspect.ismethod(value):
            value = value.__qualname__
        d_new[key] = value
    return d_new


def imgs_resize(imgs, resize_scale_height = 256,resize_scale_width=256,keep_graph=True):
    outputs = torch.FloatTensor(imgs.size()[0], imgs.size()[1], resize_scale_height, resize_scale_width)
    imgs = imgs.permute(0, 2, 3, 1)
    for i in range(imgs.size()[0]):
        if keep_graph:
            img = cv2.resize(src=imgs[i].numpy(), dsize=[resize_scale_height, resize_scale_width],interpolation=cv2.INTER_CUBIC)
        else:
            img = cv2.resize(src=imgs[i].detach().cpu().numpy(), dsize=[resize_scale_width,resize_scale_height],
                             interpolation=cv2.INTER_CUBIC)
        outputs[i] = torch.FloatTensor(img.transpose(2, 0, 1).astype(np.float32))

    return outputs


def transform_for_save(image):
    if not isinstance(image, (np.ndarray, np.generic) ):
        image =image.cpu().data.numpy()
    im_transpose = image.transpose(1, 2, 0)
    clip = np.clip(im_transpose,0,1)
    return clip


def auroc_aupr_scores(gts, preds, average_types):
    auc_dict = {}
    for average_type in average_types:
        auc = metrics.roc_auc_score(gts, preds, average=average_type)
        auc_dict[average_type] = auc

    return auc_dict

import pandas as pd
def plot_loss_by_parts_A_B(self):
    # visualize loss


    df_loss = pd.DataFrame(self.A, columns=['sims'])
    print("df loss A: ", df_loss)
    df_loss2 = pd.DataFrame(self.B, columns=['sims'])

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


def calculator_snr_sec(perturbed_p, delta_uap_p):
    adv, fs = torchaudio.load(perturbed_p)#wavfile.read(os.path.join(audios_path, audio_path))
    # print("adv: ",adv)
    delta_uap, fs = torchaudio.load(delta_uap_p)#wavfile.read(os.path.join(audios_path, audio_path))
    # print("delta_uap: ",delta_uap)
    src = adv-delta_uap
    power_sig = torch.sum(src * src).cpu().detach().numpy()
    delta = src - adv
    # print("delta: ",delta)
    # print( f"delta equal: {torch.eq(delta, delta_uap)}")
    power_noise = torch.sum(delta * delta).cpu().detach().numpy()
    snr = 10 * np.log10(power_sig / power_noise)
    print("SNR: ",snr)
    return snr


def calculator_snr_(src_p, perturbed_p):
    adv, fs = torchaudio.load(perturbed_p)#wavfile.read(os.path.join(audios_path, audio_path))
    # print("adv: ",adv)
    src, fs = torchaudio.load(src_p)#wavfile.read(os.path.join(audios_path, audio_path))
    # print("delta_uap: ",delta_uap)
    # src = adv-delta_uap
    power_sig = torch.sum(src * src).cpu().detach().numpy()
    delta = src - adv
    # print("delta: ",delta)
    # print( f"delta equal: {torch.eq(delta, delta_uap)}")
    power_noise = torch.sum(delta * delta).cpu().detach().numpy()
    snr = 10 * np.log10(power_sig / power_noise)
    print("SNR: ",snr)
    return snr


# def save_class_to_file(config, current_folder):
#     with open(os.path.join(current_folder, 'config.json'), 'w') as config_file:
#         d = dict(vars(config))
#         d.pop('scheduler_factory')
#         json.dump(d, config_file)

def PESQ(src_array, adv_array, sr=16000, device='cpu',eps=0):
    src_array = (src_array + eps).to(device)
    adv_array = adv_array.detach().to(device)
    pesq_calc = 0.0
    vector_numbers = 3# src_array.shape[0]
    for i in range(vector_numbers):
        pesq_curr = pesq(src_array[i].detach(), adv_array[i].detach(), sr)
        pesq_calc += pesq_curr
        # print("SNR curr direct: ",snr_curr)
    return pesq_calc / vector_numbers


def calculate_l2(src_array, adv_array,device='cpu'):
    src_array = src_array.to(device)
    adv_array = adv_array.detach().to(device)

    l2_calc = 0.0
    vector_numbers = src_array.shape[0]
    for i in range(vector_numbers):
        l2_curr = torch.dist(src_array[i], adv_array[i], 2)
        l2_calc += l2_curr
    return l2_calc / vector_numbers

def calculator_snr_per_signal(src_array ,adv_array, device='cpu',eps=0):
    # adv, fs = torchaudio.load(perturbed_p)#wavfile.read(os.path.join(audios_path, audio_path))
    # print("src_array: ",src_array)
    # delta_uap, fs = torchaudio.load(delta_uap_p)#wavfile.read(os.path.join(audios_path, audio_path))
    # print("adv_array: ",adv_array)
    # src = adv_array-src_array
    src_array = (src_array + eps).to(device)
    adv_array = adv_array.to(device)
    snr = []
    vector_numbers = src_array.shape[0]
    for i in range(vector_numbers):
        power_sig = torch.sum(src_array[i] * src_array[i]).cpu().detach().numpy()
        delta = src_array[i] - adv_array[i] # checking
        # delta = adv_array - src_array  # checking
        power_noise = torch.sum(delta * delta).cpu().detach().numpy()
        snr_curr = 10 * np.log10(power_sig / power_noise)
        snr.append(snr_curr)
        # print("SNR curr direct: ",snr_curr)
    return snr

def calculator_snr_direct(src_array ,adv_array, device='cpu',eps=0):
    # adv, fs = torchaudio.load(perturbed_p)#wavfile.read(os.path.join(audios_path, audio_path))
    # print("src_array: ",src_array)
    # delta_uap, fs = torchaudio.load(delta_uap_p)#wavfile.read(os.path.join(audios_path, audio_path))
    # print("adv_array: ",adv_array)
    # src = adv_array-src_array
    src_array = (src_array + eps).to(device)
    adv_array = adv_array.to(device)
    snr = 0.0
    vector_numbers = src_array.shape[0]
    for i in range(vector_numbers):
        power_sig = torch.sum(src_array[i] * src_array[i]).cpu().detach().numpy()
        delta = src_array[i] - adv_array[i] # checking
        # delta = adv_array - src_array  # checking
        power_noise = torch.sum(delta * delta).cpu().detach().numpy()
        snr_curr = 10 * np.log10(power_sig / power_noise)
        snr += snr_curr
        # print("SNR curr direct: ",snr_curr)
    return snr/vector_numbers


def calculate_snr_github_direct(pred: np.array, label: np.array): # pred = adversarial, label = source data
    # assert pred.shape == label.shape, "the shape of pred and label must be the same"
    pred, label = (pred + 1) / 2, (label + 1) / 2
    if len(pred.shape) > 1:
        sigma_s_square = np.mean(label * label, axis=1)
        sigma_e_square = np.mean((pred - label) * (pred - label), axis=1)

        # sigma_s_square = np.mean(label.numpy() * label.numpy(), axis=1)
        # np.mean((pred.cpu().detach().numpy() - label.cpu().detach().numpy()) * (
        #             pred.cpu().detach().numpy() - label.cpu().detach().numpy()), axis=1)

        # snr = 10 * np.log10((sigma_s_square / max(sigma_e_square, 1e-9))) # signal / noise
        snr = 10 * np.log10((sigma_s_square / np.maximum(sigma_e_square, 1e-9)))
        snr = snr.mean()
    else:
        sigma_s_square = np.mean(label * label)
        # print('sigma_s_square:', sigma_s_square)
        sigma_e_square = np.mean((pred - label) * (pred - label))
        # print('sigma_e_square:', sigma_e_square)
        # print(sigma_s_square/max(sigma_e_square, 1e-9))
        snr = 10 * np.log10((sigma_s_square / np.maximum(sigma_e_square, 1e-9)))
    return snr


def calculate_snr_github_direct_pkg(pred, label): # pred = adversarial, label = source data
    # assert pred.shape == label.shape, "the shape of pred and label must be the same"
    snr_func =  SignalNoiseRatio().to(device=device)
    # print("wait snr ")
    snr = snr_func(pred, label)
    # pred, label = (pred + 1) / 2, (label + 1) / 2
    # if len(pred.shape) > 1:
    #     sigma_s_square = np.mean(label * label, axis=1)
    #     sigma_e_square = np.mean((pred - label) * (pred - label), axis=1)
    #
    #     # sigma_s_square = np.mean(label.numpy() * label.numpy(), axis=1)
    #     # np.mean((pred.cpu().detach().numpy() - label.cpu().detach().numpy()) * (
    #     #             pred.cpu().detach().numpy() - label.cpu().detach().numpy()), axis=1)
    #
    #     # snr = 10 * np.log10((sigma_s_square / max(sigma_e_square, 1e-9))) # signal / noise
    #     snr = 10 * np.log10((sigma_s_square / np.maximum(sigma_e_square, 1e-9)))
    #     snr = snr.mean()
    # else:
    #     sigma_s_square = np.mean(label * label)
    #     # print('sigma_s_square:', sigma_s_square)
    #     sigma_e_square = np.mean((pred - label) * (pred - label))
    #     # print('sigma_e_square:', sigma_e_square)
    #     # print(sigma_s_square/max(sigma_e_square, 1e-9))
    #     snr = 10 * np.log10((sigma_s_square / np.maximum(sigma_e_square, 1e-9)))
    return snr



def load_from_npy(root_path, audio_type, file_name):
    file_path = os.path.join(root_path, audio_type, Path(file_name).stem)
    return np.load(f'{file_path}.npy')


def get_pert(pert_type, size):
    if pert_type == 'zeros':
        adv_pert = torch.zeros((1, size))
    elif pert_type == 'ones':
        adv_pert = torch.ones((1, size))
    elif pert_type == 'random':
        adv_pert = torch.FloatTensor(1, size).uniform_(-0.0001, 0.0001)
    elif pert_type == 'prev':
        adv_pert = torch.from_numpy(load_from_npy( os.path.join('.', 'data', 'uap_perturbation', 'ecapa'), 'snr', '100ep_100spk'))
    adv_pert.requires_grad_(True)
    return adv_pert
