from speechbrain.pretrained import EncoderClassifier
from models.wavlm_base import Titanet
from models.WavLM.WavLM import WavLM, WavLMConfig
from models.hubert import Hubert
from pathlib import Path
import os
import sys
import torch


FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))


def load_embedder(embedder_names, embedder_config, device):
    embedders = {}
    for embedder_name in embedder_names:
        curr_model = get_speaker_model_by_name(embedder_name,embedder_config[embedder_name], device)
        embedders[embedder_name] = curr_model
    return embedders


def get_speaker_model_by_name(embedder_name, cfg, device):
    if embedder_name == 'spkrec-ecapa-voxceleb': # 'SpeechBrain' :
        model = EncoderClassifier.from_hparams(cfg['files_path'])
        model.device = device
        model.eval()
        model.to(device)
    elif embedder_name == 'spkrec-xvect-voxceleb':
        model = EncoderClassifier.from_hparams(cfg['files_path'])
        model.device = device
        model.eval()
        model.to(device)
        # model.eval()
        # model.to(device)
    # elif embedder_name == 'wavlm':
    #     model = WavLm(cfg['device'])
        # wavlm = WavLm()
        # model = wavlm.get_model()
        # model.device = device
    elif embedder_name == 'wavlm':
        checkpoint = torch.load(cfg['files_path'])
        cfg_model = WavLMConfig(checkpoint['cfg'])
        model = WavLM(cfg_model)
        model.load_state_dict(checkpoint['model'])
        # model = model.double()
        model.eval()
        model.to(device)
        # model.to(cfg['device'])

    elif embedder_name == 'hubert':
        model = Hubert()
        model.eval()
        model.to(device)

    else:
        raise Exception('Model type {} not found'.format(embedder_name))

    # model.eval()
    # model.to(device)
    return model


# def _set_model(self, model_name):
#     with open(ROOT / 'configs/model_config.yaml', 'r') as stream:
#         curr_model_class = yaml.safe_load(stream)[model_name]
#         self.test_embedder_classes.append(curr_model_class)


def get_speaker_model(cfg):

    if cfg['model_name'] == 'spkrec-ecapa-voxceleb':
    #if cfg['model_source'] == 'SpeechBrain':
        model = EncoderClassifier.from_hparams(cfg['model_config']['files_path'])
        model.device = cfg['device']
        model.eval()
        model.to(cfg['device'])
        # model.eval()

    elif cfg['model_name'] == 'spkrec-xvect-voxceleb':
        model = EncoderClassifier.from_hparams(cfg['model_config']['files_path'])
        model.device = cfg['device']
        model.eval()
        model.to(cfg['device'])
        # model.eval()

    # elif cfg['model_name'] == 'wavlm':
    #     model = WavLm(cfg['device'])
    #     model = wavlm.get_model()
    #     model.device = cfg['device']
    elif cfg['model_name'] == 'wavlm':
        checkpoint = torch.load(cfg['model_config']['files_path'])
        cfg_model = WavLMConfig(checkpoint['cfg'])
        model = WavLM(cfg_model)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        model.to(cfg['device'])

    elif cfg['model_name'] == 'titanet':
        model = Titanet(cfg['device'])
        # model = titanet.get_model()

        # print("wait")
        # checkpoint = torch.load(cfg['model_config']['files_path'])
        # cfg_model = WavLMConfig(checkpoint['cfg'])
        # model = WavLM(cfg_model)
        # model.load_state_dict(checkpoint['model'])
    elif cfg['model_name'] == 'hubert':
        model = Hubert()
        model.eval()
        model.to(cfg['device'])
    else:
        raise Exception('Model type {} not found'.format(cfg['model_type']))


    print("model name in get speaker model is : ", cfg['model_name'])
    return model
