from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
from datasets import load_dataset
import torch
import operator, functools

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class WavLm:

    def __init__(self, device):

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base-sv')
        self.model = WavLMForXVector.from_pretrained('microsoft/wavlm-base-sv')# .to(device)
        self.model.output_hidden_states = True
        self.model.eval()
        # self.model.train()
        self.sampling_rate = 16_000
        self.do_normalize = False
        self.padding = True
        self.padding_value = 0.0
        self.return_attention_mask = True
        self.return_tensors = "pt"
        self.device = device
        self.model = self.model.to(self.device)
        # for param in self.model.base_model.parameters():
        #     param.requires_grad = False



    def get_model(self):
        return self.model

    def encode_batch(self, wavs, wav_lens=None, normalize=False):
        """Encodes the input audio into a single vector embedding.

        The waveforms should already be in the model's desired format.
        You can call:
        ``normalized = <this>.normalizer(signal, sample_rate)``
        to get a correctly converted signal in most cases.

        Arguments
        ---------
        wavs : torch.tensor
            Batch of waveforms [batch, time, channels] or [batch, time]
            depending on the model. Make sure the sample rate is fs=16000 Hz.
        wav_lens : torch.tensor
            Lengths of the waveforms relative to the longest one in the
            batch, tensor of shape [batch]. The longest one should have
            relative length 1.0 and others len(waveform) / max_length.
            Used for ignoring padding.
        normalize : bool
            If True, it normalizes the embeddings with the statistics
            contained in mean_var_norm_emb.

        Returns
        -------
        torch.tensor
            The encoded batch
        """
        # Manage single waveforms in input
        if len(wavs.shape) == 1:
            wavs = wavs.unsqueeze(0)

        # Assign full length if wav_lens is not assigned
        # if wav_lens is None:
        #     wav_lens = torch.ones(wavs.shape[0], device=self.device)
        #
        # # Storing waveform in the specified device
        # wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        # wavs = wavs.float()
        # wavs = wavs.to(self.device)
        # print("type")
        # wavs_np = list(wavs.cpu().numpy()) #.to(self.device)
        wavs_np = list(wavs.cpu().detach().numpy())
        del wavs

        # signals_list = list(map(operator.itemgetter('array'), dataset[:2]["audio"]))

        inputs = self.feature_extractor(wavs_np, sampling_rate=16_000, do_normalize=False, padding=True,
                                   padding_value=0.0, return_attention_mask=True,
                                   return_tensors="pt").to(self.device)
        # ,**args_dict # dataset[:2]["audio"]["array"], return_tensors="pt")

        # inputs_squeeze = {'input_values': inputs_pre['input_values'].squeeze(),
        #           'attention_mask': inputs_pre['attention_mask'].squeeze()}

        # inputs = {'input_values': inputs_pre['input_values'].squeeze().to(self.device),
        #           'attention_mask': inputs_pre['attention_mask'].squeeze().to(self.device)}

        # inputs = inputs_pre.data['input_values'].squeeze(),inputs_pre.data['attention_mask'].squeeze()

        embeddings = self.model(**inputs).embeddings
        del inputs
        del wavs_np
        torch.cuda.empty_cache()
        return embeddings# .to(self.device)

# def transformer_to_gpu():
# tokens_tensor = Tokenizer_output['input_ids'].to('cuda:0')
# token_type_ids = Tokenizer_output['token_type_ids'].to('cuda:0')
# attention_mask = Tokenizer_output['attention_mask'].to('cuda:0')
#
# output = {'input_ids' : tokens_tensor,
#           'token_type_ids' : token_type_ids,
#           'attention_mask' : attention_mask}
#
# return output
