import random
import numpy as np
import torch
import importlib
import os
import json
import pathlib
import inspect
import torchaudio
# import cv2
# from sklearn import metrics
import matplotlib as plt
import seaborn as sns

def init_seeds(seed=0):
    # Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
    # cudnn seed 0 settings are slower and more reproducible, else faster and less reproducible
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    np.random.seed(seed)
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


def save_class_to_file(config, current_folder):
    with open(os.path.join(current_folder, 'config.json'), 'w') as config_file:
        d = dict(vars(config))
        d_new = check_dict(d)
        json.dump(d_new, config_file)


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

def calculator_snr_direct(src_array ,adv_array):
    # adv, fs = torchaudio.load(perturbed_p)#wavfile.read(os.path.join(audios_path, audio_path))
    # print("src_array: ",src_array)
    # delta_uap, fs = torchaudio.load(delta_uap_p)#wavfile.read(os.path.join(audios_path, audio_path))
    # print("adv_array: ",adv_array)
    # src = adv_array-src_array
    power_sig = torch.sum(src_array * src_array).cpu().detach().numpy()
    delta = src_array - adv_array # checking
    # delta = adv_array - src_array  # checking
    power_noise = torch.sum(delta * delta).cpu().detach().numpy()
    snr = 10 * np.log10(power_sig / power_noise)
    print("SNR direct: ",snr)
    return snr
