from speechbrain.pretrained import EncoderClassifier

import torch
import numpy as np
from torch.autograd import Variable


class SRModel(EncoderClassifier):
  def get_batch_features(self, wavs, wav_lens=None, normalize=False):
      # Manage single waveforms in input
    if len(wavs.shape) == 1:
        wavs = wavs.unsqueeze(0)

    # Assign full length if wav_lens is not assigned
    if wav_lens is None:
        wav_lens = torch.ones(wavs.shape[0], device=self.device)

    # Storing waveform in the specified device
    wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
    wavs = wavs.float()

    # Computing features and embeddings
    feats = self.mods.compute_features(wavs)
    feats = self.mods.mean_var_norm(feats, wav_lens)
    return feats


# regular model with cosineSimilarity loss
class AttackSRModel:

    def __init__(self, batch_size=32, aud_length=3, pertb_length=1,
                 fs=16000, eps=0.0001, attack_name='PGD', loss_name='COSE'):  # 1#3
        self.classifier_class = SRModel.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
        self.model = SRModel.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb").mods.embedding_model

        self.batch_size = batch_size
        self.aud_length = aud_length
        self.pertb_length = pertb_length
        self.fs = fs
        self.max_audio_len = int(fs * aud_length)

        self.delta = {"PGD": torch.tensor(np.zeros((self.pertb_length * fs * 3), dtype=np.float32), requires_grad=True),
                      "FGSM": torch.tensor(np.random.rand((self.pertb_length * fs)), requires_grad=True)}[attack_name]
        self.loss_name = attack_name

        self.loss = {'COSE': torch.nn.CosineEmbeddingLoss(), 'CRE': torch.nn.CrossEntropyLoss()}[loss_name]
        self.loss_name = loss_name

        params_to_train = [self.delta]
        params_to_train += list(self.classifier_class.mods.embedding_model.parameters())
        self.apply_sign_optimizer = torch.optim.SGD(params_to_train, lr=0.001)
        self.eps = eps
        self.A = []
        self.B = []
        self.loss_pgd = []

    def forward(self, audio_batch, batch_size):
        #################################################
        ### Configuring the UAP with the legit audios ###
        #################################################

        # we tiling the UAP three times

        apply_delta = torch.stack([self.delta.clone() for _ in range(int(self.aud_length / self.pertb_length))])  # 3
        apply_delta = torch.reshape(apply_delta, [-1])

        # copying the UAP batch size times, so we can patch it for each audio file in out training
        apply_delta = torch.broadcast_to(apply_delta, [batch_size, apply_delta.shape[0]])
        new_input = audio_batch + self.eps * apply_delta

        pass_in = torch.clamp(new_input, -2 ** 15, 2 ** 15 - 1)
        # pass_in = new_input

        embeddings_batch = self.classifier_class.encode_batch(pass_in)
        return embeddings_batch

    def forward_clean(self, audio_batch, batch_size):
        embeddings_batch = self.classifier_class.encode_batch(audio_batch)
        return embeddings_batch

    def get_numpy(self, embeddings_batch):
        return embeddings_batch.detach().numpy()[0]

    def test_fgsm(self, audio_batch, target_embeddings_batch, batch_size, eps=1e-8):
        self.delta.requires_grad = True

        embeddings_batch = self.forward(audio_batch, batch_size)

        cost = self.loss(torch.squeeze(embeddings_batch, 0),
                         torch.squeeze(target_embeddings_batch, 0),
                         Variable(torch.Tensor([1])))

        self.model.zero_grad()

        cost.backward(retain_graph=True)
        signed_delta = self.delta.grad.data.sign()
        apply_delta = torch.stack([signed_delta.clone() for _ in range(int(self.aud_length / self.pertb_length))])  # 3
        apply_delta = torch.reshape(apply_delta, [-1])

        # copying the UAP batch size times, so we can patch it for each audio file in out training
        apply_delta = torch.broadcast_to(apply_delta, [batch_size, apply_delta.shape[0]])
        new_input = audio_batch + apply_delta  # * self.eps

        # new_input = torch.max(torch.min(new_input, audio_batch + eps), audio_batch - eps)
        pass_in = torch.clamp(new_input, -2 ** 15, 2 ** 15 - 1)

        # pass_in = new_input
        pertubed_embeddings_batch = self.classifier_class.encode_batch(pass_in)
        return pertubed_embeddings_batch
