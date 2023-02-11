import os
import torch
import warnings
import torch.nn as nn
# import torchvision

# from models.LIT.model import Rec_Transformer
from utils.general import intersect_dicts


def get_lensless_model(cfg):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        print('Loading {} lensless model...'.format(cfg['model_name']))
        load_weights = True
        if cfg['model_name'] == 'LIT':
            model = Rec_Transformer(input_size=cfg['model_config']['input_size'], rec_size=cfg['model_config']['output_size'])
        elif 'ADMM' in cfg['model_name'] or cfg['model_name'] == 'UNet':
            import sys
            sys.path.append(os.path.join(cfg['root_path'], 'models', 'ADMM'))
            model = torch.load(cfg['root_path'] / cfg['model_config']['weights_path'], map_location=cfg['device'])
            model.cuda_device = cfg['device']
            if hasattr(model, 'admm_model'):
                model.admm_model.cuda_device = cfg['device']
            sys.path.remove(os.path.join(cfg['root_path'], 'models', 'ADMM'))
            load_weights = False
        else:
            raise RuntimeError(f"Model name {cfg['model_name']} not found")

        if load_weights:
            csd = torch.load(cfg['root_path'] / cfg['model_config']['weights_path'], map_location=cfg['device'])
            csd = intersect_dicts(csd, model.state_dict(), exclude=[])  # intersect
            model.load_state_dict(csd)  # load
            print(f'Transferred {len(csd)}/{len(model.state_dict())} items')  # report
            del csd

        # model.eval()
        model.apply(freeze_bn_stats)
        model.to(cfg['device'])
        return model


def get_classification_model(estimator_config):
    if estimator_config['model_arch'] == 'resnet18':
        model = torchvision.models.resnet18(pretrained=True)
        model.fc = nn.Sequential(
            nn.Linear(in_features=model.fc.in_features, out_features=estimator_config['num_of_classes'])
        )
        print("Load Resnet18 pretrained model")
    elif estimator_config['model_arch'] == 'resnet34':
        model = torchvision.models.resnet34(pretrained=True)
        model.fc = nn.Sequential(
            nn.Linear(in_features=model.fc.in_features, out_features=estimator_config['num_of_classes'])
        )
        print("Load Resnet34 pretrained model")
    elif estimator_config['model_arch'] == 'vgg16':
        model = torchvision.models.vgg16(pretrained=True)
        num_features = model.classifier[6].in_features
        features = list(model.classifier.children())[:-1]
        features.extend([torch.nn.Linear(num_features, estimator_config['num_of_classes'])])
        model.classifier = torch.nn.Sequential(*features)
        print("Load vgg16 pretrained model")
    elif estimator_config['model_arch'] == "DenseNet":
        model = torchvision.models.densenet121(pretrained=True)
        num_features = model.classifier.in_features
        model.classifier = torch.nn.Linear(num_features, estimator_config['num_of_classes'])
        print("Load DenseNet pretrained model")

    if estimator_config['mode'] == 'eval':
        print('Weights loaded from {}'.format(estimator_config['weights_path']))
        sd = torch.load(estimator_config['weights_path'], map_location=estimator_config['device'])['model_state_dict']
        model.load_state_dict(sd)
        model.eval()
        model.apply(freeze_bn)

    model.to(estimator_config['device'])
    return model


def freeze_bn_stats(m):
    if isinstance(m, nn.BatchNorm2d):
        m.track_running_stats = False


def freeze_bn(m):
    if isinstance(m, nn.BatchNorm2d):
        m.eval()


def unfreeze_bn(m):
    if isinstance(m, nn.BatchNorm2d):
        m.train()
