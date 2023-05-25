# inspired by adversarial-robustness-toolbox (IBM)

import torch
from tqdm import tqdm
from utils.general import calculator_snr_direct


class CarliniWagnerL2:
    def __init__(self,
                 loss_function=None,
                 norm="inf",
                 eps=0.3,
                 eps_step=0.1,
                 decay=None,
                 max_iter=100,
                 targeted=False,
                 num_random_init=1,
                 device='cpu',
                 clip_values=(0, 1)) -> None:
        super().__init__()
        self.loss_function = loss_function
        self.norm = norm
        self.eps_step = torch.tensor(eps_step)
        self.eps = torch.tensor(eps, dtype=torch.float32, device=device)
        self.decay = decay
        self.max_iter = max_iter
        self.targeted = targeted
        self.num_random_init = num_random_init
        self.device = device
        self.clip_min = torch.tensor(clip_values[0], dtype=torch.float32, device=device)
        self.clip_max = torch.tensor(clip_values[1], dtype=torch.float32, device=device)

    def generate(self, inputs, targets, batch_info):
        return self._generate_batch(inputs, targets, batch_info)

    def _generate_batch(self, inputs, targets, batch_info):
        adv_x = inputs.clone()
        momentum = torch.zeros(inputs.shape)
        progress_bar = tqdm(range(self.max_iter),
                            total=self.max_iter,
                            desc='Batch {}/{}'.format(batch_info['cur'], batch_info['total']),
                            ncols=150)
        self.loss_values = []
        for _ in progress_bar:
            adv_x = self._compute(adv_x, inputs, targets, momentum)
            snr_score = calculator_snr_direct(src_array=inputs, adv_array=adv_x)
            progress_bar.set_postfix_str('Batch Loss: {:.6}, SNR: {:.6}'.format(self.loss_values[-1], snr_score))
        return adv_x

    def _compute(self, x_adv, x_init, targets, momentum):
        # handle random init
        perturbation = self._compute_perturbation(x_adv, targets, momentum)
        x_adv = self._apply_perturbation(x_adv, perturbation)
        perturbation = self._projection(x_adv - x_init)
        x_adv = perturbation + x_init
        return x_adv

    def _compute_perturbation(self, adv_x, targets, momentum):
        tol = 10e-8
        grad, loss_value = self.loss_function(adv_x, targets)
        self.loss_values.append(loss_value)
        grad = grad * (1 - 2 * int(self.targeted))
        if torch.any(grad.isnan()):
            grad[grad.isnan()] = 0.0

        # Apply momentum
        if self.decay is not None:
            ind = tuple(range(1, len(adv_x.shape)))
            grad = grad / (torch.sum(grad.abs(), dim=ind, keepdims=True) + tol)  # type: ignore
            grad = self.decay * momentum + grad
            # Accumulate the gradient for the next iter
            momentum += grad

        # Apply norm
        if self.norm == "inf":
            grad = grad.sign()
        elif self.norm == 1:
            ind = tuple(range(1, len(adv_x.shape)))
            grad = grad / (torch.sum(grad.abs(), dim=ind, keepdim=True) + tol)
        elif self.norm == 2:
            ind = tuple(range(1, len(adv_x.shape)))
            grad = grad / (torch.sqrt(torch.sum(grad * grad, dim=ind, keepdim=True)) + tol)

        return grad

    def _apply_perturbation(self, adv_x, perturbation):
        perturbation_step = self.eps_step * perturbation
        perturbation_step[torch.isnan(perturbation_step)] = 0
        adv_x = adv_x + perturbation_step
        if self.clip_min is not None and self.clip_max is not None:
            adv_x = torch.max(
                torch.min(adv_x, self.clip_max),
                self.clip_min,
            )
        return adv_x

    def _projection(self, values):
        tol = 10e-8
        values_tmp = values.reshape(values.shape[0], -1)
        if self.norm == 2:
            values_tmp = (values_tmp *
                          torch.min(
                              torch.tensor([1.0], dtype=torch.float32).to(self.device),
                              self.eps / (torch.norm(values_tmp, p=2, dim=1) + tol),
                              ).unsqueeze_(-1)
                          )
        elif self.norm == 1:
            values_tmp = (values_tmp *
                          torch.min(
                              torch.tensor([1.0], dtype=torch.float32).to(self.device),
                              self.eps / (torch.norm(values_tmp, p=1, dim=1) + tol),
                              ).unsqueeze_(-1)
                          )
        elif self.norm == 'inf':
            values_tmp = values_tmp.sign() * torch.min(values_tmp.abs(), self.eps)
        values = values_tmp.reshape(values.shape)
        return values

#
# class PGD:
#     def __init__(self):
#         self.num_steps =None
#         self.eps = None
#
#
#     def generate_attack(self, x, y, signal, num_steps=40, eps=0.3):
#         # self.delta.requires_grad = True
#         """Performs the projected gradient descent attack on a batch of images."""
#         # print("x:" ,x)
#         x_adv = x.clone().detach().requires_grad_(True).to(x.device)
#         print("x_adv in pgd AttackSRModelCustomLossWithSNR : ", x_adv)
#         print("self.delta: ", self.delta)
#         for i in tqdm(range(num_steps), desc="Attack step:"):
#             # print(f"Step number: {i} out of {num_steps}")
#             print("i in pgd : ", i)
#             _x_adv = x_adv.clone().detach().requires_grad_(True)
#             print("_x_adv: ", _x_adv)
#
#             prediction = self.classifier_class.encode_batch(_x_adv)  # self.forward_clean(_x_adv,1)
#             # print("prediction:" ,prediction.shape) $$$
#             # print("y:" ,y) $$$
#             loss, A_loss, B_loss = self.loss(prediction, y)  # torch.squeeze(y,0)) #CLOSS
#             print("A loss:  ", A_loss)
#             print("B loss:  ", B_loss)
#             print("loss:  ", loss)
#             print("_x_adv: ", _x_adv)
#             # if i ==0:
#             #   loss = loss + ADD_05
#             #   print("loss add 0.5:  ",loss)
#
#             # loss = loss + ADD_05/num_steps
#             # print("loss add 0.5/num steps:  ",loss)
#             print("num_steps:  ", num_steps)
#
#             # loss = 1-loss
#             # A_loss = 1 - A_loss
#             # B_loss = 1 - B_loss
#             # print("A loss 1-loss: ", A_loss)
#
#             self.A.append(A_loss.detach().clone().numpy())
#
#             # print("type Aloss: ", type(A_loss))
#             self.B.append(B_loss.detach().clone().numpy())
#             # print("B loss 1-loss: ", B_loss)
#             # print("loss in pgd 1-loss: ", loss)
#             new_input = signal + self.delta.clone() * self.eps  # + SNR_smooth * self.eps
#             input_clip = torch.clamp(new_input, -2 ** 15, 2 ** 15 - 1)
#             snr_direct = calculator_snr_direct(signal, input_clip)
#             # loss = self.loss(torch.squeeze(prediction,0), y,Variable(torch.Tensor([1]))) #COS
#             print("snr_direct: ", snr_direct)
#             print("loss: ", loss)
#             print("input_clip: ", input_clip)
#             # snr_square = np.square(snr_direct / 150)
#             # print("snr_square: ", snr_square)
#
#             snr_square = snr_direct / 150
#             # snr_square = (50 -snr_direct)/100
#             print("snr_norm: ", snr_square)
#             snr_part = S * snr_square
#             loss = loss + snr_part  # (snr_direct / 100 )
#             print("loss IN PGD: ", loss)
#             # loss = 1 - loss
#
#             self.loss_pgd.append(loss.detach().clone().numpy())
#             self.snr_loss.append(snr_direct)
#
#             loss_score = loss
#             loss.backward()
#             print("loss IN PGD after backward: ", loss)
#             print("\nself.delta after backward: ", self.delta)
#             self.apply_sign_optimizer.step()
#             print("self.delta after opt: ", self.delta)
#
#             with torch.no_grad():
#                 # Force the gradient step to be a fixed size in a certain norm
#                 # if step_norm == 'inf':
#                 gradients = _x_adv.grad.sign()  # * self.eps
#                 self.delta = self.delta + gradients
#
#                 # Untargeted: Gradient ascent on the loss of the correct label w.r.t.
#                 # the model parameters
#                 x_adv += self.delta
#
#             # Project back into l_norm ball and correct range
#             # if eps_norm == 'inf':
#             # Workaround as PyTorch doesn't have elementwise clip
#
#             # x_adv = torch.max(torch.min(x_adv, x + eps), x - eps) # check the diff between
#             # x_adv = custom_clip(x,x_adv,self.eps) custom clip, shoham
#             x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
#             print("self.delta after torch.no_grad() : ", self.delta)
#
#             x_adv = x_adv.clamp(-2 ** 15, 2 ** 15 - 1)  # del regular clip, adding new clip shoham
#             print("x_adv after torch.no_grad() : ", x_adv)
#             # x_adv = custom_clip(x,x_adv,eps)
#             # x_adv = x_adv.clamp(-1,1)
#         pertubed_embeddings_batch = self.classifier_class.encode_batch(x_adv)
#         print("custom ################################################8")
#         self.plot_loss_by_parts_A_B()
#         print(" plot")
#         return pertubed_embeddings_batch, loss_score  # x_adv
