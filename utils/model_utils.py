from speechbrain.pretrained import EncoderClassifier


def get_speaker_model(cfg):
    if cfg['model_source'] == 'SpeechBrain':
        model = EncoderClassifier.from_hparams(cfg['model_config']['files_path'])
        model.device = cfg['device']
    else:
        raise Exception('Model type {} not found'.format(cfg['model_type']))

    model.eval()
    model.to(cfg['device'])
    return model
