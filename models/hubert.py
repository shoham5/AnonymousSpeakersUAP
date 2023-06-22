import torch.nn as nn
from transformers import HubertForSequenceClassification#, Wav2Vec2FeatureExtractor
from transformers.feature_extraction_utils import BatchFeature


class Hubert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = HubertForSequenceClassification.from_pretrained("superb/hubert-base-superb-sid")
        # self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/hubert-base-superb-sid")

    def encode_batch(self, inputs):
        # inputs = self.feature_extractor(inputs, sampling_rate=16000, padding=True, return_tensors="pt")
        inputs = BatchFeature({"input_values": inputs}) # this is the only thing that happends within wav2vec feature extractor in our scenario
        full_output = self.model(**inputs)
        emb = full_output.hidden_states[-1].mean(dim=1)
        return emb

