import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import Module
from torch import Tensor
from utils.general import calculator_snr_direct
import numpy as np
from constants import A,B,S,SNR_smooth

import torch


class MSE:
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.func = torch.nn.MSELoss(**kwargs)

    def __call__(self, x, y):
        return self.func(x, y)


class L1:
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.func = torch.nn.L1Loss(**kwargs)

    def __call__(self, x, y):
        return self.func(x, y)


class LPIPS:
    def __init__(self, **kwargs) -> None:
        self.func = lpips.LPIPS(net=kwargs['net'])
        self.func.to(kwargs['device'])

    def __call__(self, x, y):
        return self.func(x, y)


class CosineSimilarity:
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.func = torch.nn.CosineSimilarity(**kwargs)

    def __call__(self, x, y):
        return (self.func(x.squeeze(), y.squeeze()) + 1) / 2


class WSNRLoss:
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.func = SNRLoss(**kwargs)

    def __call__(self, x, y):
        return self.func(x, y)


class SNRLoss(Module):

    __constants__ = ['dim', 'eps']
    dim: int
    eps: float
    #1e-10
    def __init__(self,eps: float =1e-10 ) -> None:
        super(SNRLoss, self).__init__()
        self.eps = eps

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        return calculator_snr_direct(x1, x2)



class Loss:
    def __init__(self, model, loss_fns, weights=None, **kwargs) -> None:
        super().__init__()
        self.model = model
        self.loss_fns = loss_fns
        if weights is not None:
            self.loss_weights = weights
        else:
            self.loss_weights = [1] * len(loss_fns)

    def loss_gradient(self, x, y):
        x_grad = x.clone().detach().requires_grad_(True)
        y_c = y.clone().detach()
        pred = self.model.encode_batch(x_grad)  # TODO: remove hard-coded encode_batch

        loss = torch.zeros(1, device=x_grad.device)
        for loss_fn, loss_weight in zip(self.loss_fns, self.loss_weights):
            loss += loss_weight * loss_fn(pred, y_c).squeeze().mean()

        self.model.zero_grad()
        loss.backward()
        grads = x_grad.grad
        return grads, loss.item()


# ATTACK WITH SNR IN LOSS, CUSTOM CLIP,
class AS2TL2LossSNR(nn.Module):

    def __init__(self, enroll_utters, threshold=0, labels=None, const_speaker=False):
        '''
        Implementation of the l2s custom loss function in

        Accepts an input of size (N, M, D)

            where N is the number of speakers in the batch,
            M is
            and D is the dimensionality of the embedding vector (e.g. d-vector)

        Args:
        self.enroll (ndarray)
        '''
        super(AS2TL2LossSNR, self).__init__()
        self.enroll = enroll_utters
        self.threshold = threshold
        # self.embed_loss = self.embed_loss_l2
        self.order_labels = list(enroll_utters.keys())
        self.order_labels.sort()
        self.enroll_matrix = torch.from_numpy(np.array([enroll_utters[i] for i in self.order_labels]))
        self.size_enroll = len(self.enroll_matrix)
        self.labels = labels
        print("Threshold: ", threshold)
        self.const_speaker = const_speaker
        if const_speaker:
            print("in const speaker")
            self.def_speaker = False

    def calc_cosine_sim(self, dvecs):
        '''
        Make the cosine similarity matrix with dims (N,M,N)
        '''
        # if not dvecs.shape[0] == 1: dim = 1
        sims = nn.CosineSimilarity(dim=1)
        cos_sim_matrix = []
        for dvec in enumerate(dvecs):
            # Calculate the cosine similarity matrix
            cs_row = sims(dvec, self.enroll_matrix)
            cos_sim_matrix.append(cs_row)
        return torch.stack(cos_sim_matrix)

    def embed_loss_softmax(self, dvecs, cos_sim_matrix):
        '''
        Calculates the loss on each embedding $L(e_{ji})$ by taking softmax
        '''
        N, M, _ = dvecs.shape
        L = []
        for j in range(N):
            L_row = []
            for i in range(M):
                L_row.append(-F.log_softmax(cos_sim_matrix[j, i], 0)[j])
            L_row = torch.stack(L_row)
            L.append(L_row)
        return torch.stack(L)

    def embed_loss_l2(self, dvecs, cos_sim_matrix, gt_labels):
        '''
         Calculates the loss on each embedding $L(e_{ji})$ by l2 loss
         l2(x,t) = max{threshold,S(x)_s} -max i in G, i!=s [S(x)]i
         dvecs(tensor): current vector batch
         cos_sim_matirx(tensor): matrix similarty, each [i,j]cell include similarty between, i vector in dvec and j vector in src
         gt_labels(tensor): labels of current batch
        '''

        # TODO CHANGE LIST OF SPEAKERS TO DICT WITH KEY:SPK_ID,  VAL: NUMBER
        N, M, _ = dvecs.shape

        L = []
        for i in range(N):

            src_ind = gt_labels.tolist()[i]
            print("src_ind: ", src_ind)
            S_src, S_i, max_ind = -1, -1, -1
            L_row = []
            for j in range(self.size_enroll):
                print("j: ", j)
                if j == src_ind:
                    # print("cos sim matrix" , cos_sim_matrix[i,j])
                    print("cos sim matrix src_ind: ", cos_sim_matrix[i * j + j])
                    print("in j==src_ind")
                    print(f"i: {i} , j: {j} ")
                    S_src = cos_sim_matrix[i * self.size_enroll + j]

                elif self.const_speaker and self.def_speaker:
                    print("in self const: ")
                    S_i = cos_sim_matrix[i * self.size_enroll + self.max_index]
                    S_src = cos_sim_matrix[i * self.size_enroll + self.src_ind]

                    break

                elif cos_sim_matrix[i * self.size_enroll + j] > S_i:
                    S_i = cos_sim_matrix[i * self.size_enroll + j]
                    print("in regular cos_sim matrix: ")
                    max_ind = j
                    max_row_ind = i  # handle batch, should update in self.cons_speaker too
                print("cos sim matrix max_ind: ", S_i)

            print("cos sim matrix src_ind return : ", cos_sim_matrix[i * self.size_enroll + src_ind])
            print("cos sim matrix src_ind return S_src : ", S_src)
            print("cos sim matrix retrun S_i: ", S_i)
            print("cos sim matrix retrun B *S_i: ", B * S_i)
            print("loss_val: ", A * max(self.threshold, S_src) - B * S_i)
            if self.const_speaker and not self.def_speaker:
                print("only  once")
                self.def_speaker = True
                self.max_index = max_ind
                self.src_ind = src_ind
                self.max_row_ind = max_row_ind
                self.S_i = S_i

            L.append((A * max(torch.tensor(self.threshold), S_src)) - (B * S_i))  # without square
        return torch.stack(L), A * max(torch.tensor(self.threshold), S_src), B * (1 - S_i)



    def S_x(self, dvecs, cos_sim_matrix, x, s):
        return s



    def forward(self, dvecs, labels):
        '''
        Calculates the AS2T l2 loss for an input of dimensions (num_speakers, num_utts_per_speaker, dvec_feats)
        '''

        print(f"\nlabels ind_ {labels}labels: {self.order_labels[labels]}")
        print("order_labels: ", self.order_labels)
        sims = nn.CosineSimilarity(dim=1)
        cos_sim_matrix = sims(torch.squeeze(dvecs, 0), torch.squeeze(self.enroll_matrix, 1))

        cos_sim_matrix = (cos_sim_matrix + 1) / 2
        print("\ncos_sim_matrix_after: ", cos_sim_matrix)


        L, A_loss, B_loss = self.embed_loss_l2(dvecs, cos_sim_matrix, labels)



        return L.sum(), A_loss, B_loss






class AS2TL2Loss(nn.Module):

    def __init__(self, enroll_utters, threshold=0, labels=None):
        '''
        Implementation of the l2s custom loss function in

        Accepts an input of size (N, M, D)

            where N is the number of speakers in the batch,
            M is
            and D is the dimensionality of the embedding vector (e.g. d-vector)

        Args:
        self.enroll (ndarray)
        '''
        super(AS2TL2Loss, self).__init__()
        self.enroll = enroll_utters
        self.threshold = threshold
        # self.embed_loss = self.embed_loss_l2
        self.order_labels = list(enroll_utters.keys())
        self.order_labels.sort()
        self.enroll_matrix = torch.from_numpy(np.array([enroll_utters[i] for i in self.order_labels]))
        self.size_enroll = len(self.enroll_matrix)
        self.labels = labels
        print("Threshold: ", threshold)


    def calc_cosine_sim(self, dvecs):
        '''
        Make the cosine similarity matrix with dims (N,M,N)
        '''
        # if not dvecs.shape[0] == 1: dim = 1
        sims = nn.CosineSimilarity(dim=1)
        cos_sim_matrix = []
        for dvec in enumerate(dvecs):
            # Calculate the cosine similarity matrix
            cs_row = sims(dvec, self.enroll_matrix)
            cos_sim_matrix.append(cs_row)
        return torch.stack(cos_sim_matrix)

    def embed_loss_softmax(self, dvecs, cos_sim_matrix):
        '''
        Calculates the loss on each embedding $L(e_{ji})$ by taking softmax
        '''
        N, M, _ = dvecs.shape
        L = []
        for j in range(N):
            L_row = []
            for i in range(M):
                L_row.append(-F.log_softmax(cos_sim_matrix[j, i], 0)[j])
            L_row = torch.stack(L_row)
            L.append(L_row)
        return torch.stack(L)

    def embed_loss_l2(self, dvecs, cos_sim_matrix, gt_labels):
        '''
         Calculates the loss on each embedding $L(e_{ji})$ by l2 loss
         l2(x,t) = max{threshold,S(x)_s} -max i in G, i!=s [S(x)]i
         dvecs(tensor): current vector batch
         cos_sim_matirx(tensor): matrix similarty, each [i,j]cell include similarty between, i vector in dvec and j vector in src
         gt_labels(tensor): labels of current batch
        '''
        # TODO CHANGE LIST OF SPEAKERS TO DICT WITH KEY:SPK_ID,  VAL: NUMBER
        N, M, _ = dvecs.shape

        L = []
        for i in range(N):

            src_ind = gt_labels.tolist()[i]
            print("src_ind: ", src_ind)
            S_src, S_i, max_ind = -1, -1, -1
            L_row = []
            for j in range(self.size_enroll):
                print("j: ", j)
                if j == src_ind:
                    # print("cos sim matrix" , cos_sim_matrix[i,j])
                    print("cos sim matrix src_ind: ", cos_sim_matrix[i * j + j])
                    print(f"i: {i} , j: {j} ")
                    S_src = cos_sim_matrix[i * self.size_enroll + j]
                elif cos_sim_matrix[i * self.size_enroll + j] > S_i:
                    S_i = cos_sim_matrix[i * self.size_enroll + j]
                    max_ind = j
                print("cos sim matrix max_ind: ", S_i)
            print(f"end: {i} , max_ind: {max_ind} ")
            print("cos sim matrix src_ind: ", cos_sim_matrix[i * self.size_enroll + src_ind])
            print("cos sim matrix B *S_i: ", B * S_i)
            print("loss_val: ", max(self.threshold, S_src) - S_i)
            L.append(A * max(self.threshold, S_src) - B * S_i)
        return torch.stack(L), A * S_src, B * S_i



    def S_x(self, dvecs, cos_sim_matrix, x, s):
        return s


    def forward(self, dvecs, labels):
        '''
        Calculates the AS2T l2 loss for an input of dimensions (num_speakers, num_utts_per_speaker, dvec_feats)
        '''

        print(f"\nlabels ind_ {labels}labels: {self.order_labels[labels]}")
        print("order_labels: ", self.order_labels)
        sims = nn.CosineSimilarity(dim=1)
        cos_sim_matrix = sims(torch.squeeze(dvecs, 0), torch.squeeze(self.enroll_matrix, 1))

        cos_sim_matrix = (cos_sim_matrix + 1) / 2  # shift cosinesim to range 0,1
        print("\ncos_sim_matrix_after: ", cos_sim_matrix)

        L, A_loss, B_loss = self.embed_loss_l2(dvecs, cos_sim_matrix, labels)

        return L.sum(), A_loss, B_loss





def custom_clip(x, x_adv, eps,device='cpu'):
    x_add_epsilon = torch.add(x, eps).to(device)
    print(device)
    x_sub_epsilon = torch.sub(x, eps).to(device)
    part_a = torch.minimum(x_add_epsilon, torch.Tensor([1]).to(device))
    part_b = torch.maximum(x_adv.to(device), x_sub_epsilon).to(device)
    x_adv_clip = torch.minimum(part_a, torch.maximum(part_b, torch.Tensor([-1]).to(device))).to(device)
    return x_adv_clip
