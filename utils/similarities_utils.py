import torch


class CosineSimilaritySims:
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.func = torch.nn.CosineSimilarity(**kwargs)

    def __call__(self, x, y):
        return (self.func(x, y) + 1) / 2
