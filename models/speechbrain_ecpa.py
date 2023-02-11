from speechbrain.pretrained import EncoderClassifier
from speechbrain.lobes.features import Fbank
import torch
import numpy as np
from torch.autograd import Variable
import pandas as pd
import torchaudio
import matplotlib as plt
import seaborn as sns
from tqdm import tqdm


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
        # fs a.k.a sample-rate - 16kHz
        self.max_audio_len = int(fs * aud_length)
        # delta - UAP
        # learning_rate - the gradient step size, decreasing gradually from 10 to 1

        # self.delta = torch.tensor(np.zeros((self.pertb_length * fs * 3), dtype=np.float32), requires_grad=True) #PGD
        # # self.delta = torch.tensor(np.random.rand((self.pertb_length * fs)), requires_grad=True) #FGSM
        # self.loss = torch.nn.CosineEmbeddingLoss()
        # # self.loss = torch.nn.CrossEntropyLoss()
        # # self.compute_loss_optimizer = tf.train.AdamOptimizer(1e-3)

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
        # DON'T mess with the UAP length and the tiling, because it's overfitting to our usecase - 3 second UAP, tiled
        # 3 times to 9 seconds audio files

        # print(self.aud_length/ self.pertb_length )
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
        # cost = self.loss(torch.squeeze(embeddings_batch,0),
        # torch.squeeze(target_embeddings_batch,0))
        self.model.zero_grad()

        cost.backward(retain_graph=True)

        # we tiling the UAP three times
        # DON'T mess with the UAP length and the tiling, because it's overfitting to our usecase - 3 second UAP, tiled
        # 3 times to 9 seconds audio files
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


    def plot_loss_by_parts_A_B(self):
        # visualize loss
        # sns.lineplot(data= s, color = "red")
        # sns.lineplot(data= t, color ="blue")

        # df_loss = pd.DataFrame(self.A, columns =['sims'])
        # df_loss = df_loss.reset_index()
        # sns.lineplot(data=df_loss, x="index", y="sims")
        # # self.B = []
        # del df_loss
        # plt.title("A Sims vs Utterences")
        # plt.ylim(0, 1)
        # plt.show()

        # df_loss = pd.DataFrame([1 - x.detach().numpy() for x in self.A], columns =['loss'])
        # print( " in plot loss  $$$$ ")
        df_loss = pd.DataFrame(self.A, columns=['loss'])
        print("minimize anc vs neg similartiy ")
        print("df loss=1-sims self.A:  ", df_loss)
        # df_loss2 = pd.DataFrame(self.B, columns =['sims'])

        # print(df_loss)
        # print(df_loss2)
        df_loss = df_loss.reset_index()
        # df_loss2 = df_loss2.reset_index()

        sns.lineplot(data=df_loss, x="index", y="loss")
        # sns.lineplot(data=df_loss2, x="index", y="sims")

        plt.title("Regular loss vs PGD Steps")
        plt.ylim(-1, 2)
        plt.legend('A', ncol=2, loc='upper left');
        self.A = []
        # self.B = []
        del df_loss
        # del df_loss2
        plt.show()

        # df_loss3 = pd.DataFrame(self.snr_loss, columns =['sims'])
        # df_loss3 = df_loss3.reset_index()
        # sns.lineplot(data=df_loss3, x="index", y="norm snr")
        # plt.title("norm snr vs steps")
        # plt.ylim(0, 1)
        # plt.legend('S', ncol=2, loc='upper left');
        # self.snr_loss = []
        # del df_loss3
        # plt.show()

        df_loss = pd.DataFrame(self.loss_pgd, columns=['loss'])
        df_loss2 = pd.DataFrame(self.B, columns=['loss'])
        df_loss = df_loss.reset_index()
        df_loss2 = df_loss2.reset_index()
        sns.lineplot(data=df_loss, x="index", y="loss")
        sns.lineplot(data=df_loss2, x="index", y="loss")
        plt.title("Regular Loss vs PGD Steps")
        plt.legend('L1', ncol=2, loc='upper left');

        print("df loss=1-sims self.B:  ", df_loss2)
        self.B = []

        del df_loss
        self.loss_pgd = []  # DEL TO CALC LOSS FOR ALL
        plt.show()


    def projected_gradient_descent(self, x, y, num_steps=40, eps=0.3):
        # self.delta.requires_grad = True
        """Performs the projected gradient descent attack on a batch of images."""
        x_adv = x.clone().detach().requires_grad_(True).to(x.device)
        x_adv_src = x.clone().detach().requires_grad_(True).to(x.device)
        for i in tqdm(range(num_steps), desc="Attack step:"):
            # print(f"Step number: {i} out of {num_steps}")
            _x_adv = x_adv.clone().detach().requires_grad_(True)

            prediction = self.classifier_class.encode_batch(_x_adv)  # self.forward_clean(_x_adv,1)
            prediction_src = self.classifier_class.encode_batch(x_adv_src)  # self.forward_clean(_x_adv,1)
            loss = self.loss(torch.squeeze(prediction, 0), torch.squeeze(y, 0), Variable(torch.Tensor([1])))  # COS
            print("\nloss: ", loss)
            print("\nloss type: ", type(loss))

            cos_sim = nn.CosineSimilarity(dim=1)
            sims = (cos_sim(torch.squeeze(prediction, 0), torch.squeeze(y, 0)) + 1) / 2  # y instead _x_adv
            sims_reg = cos_sim(torch.squeeze(prediction, 0), torch.squeeze(y, 0))  # y instead _x_adv
            # cosine_similarity(embd_np, embd_np)[0][0] + 1 )/ 2
            sims_x = cos_sim(torch.squeeze(prediction, 0), torch.squeeze(prediction_src, 0))
            print("\nsims AttackSRmodel anc vs neg  : ", sims)
            print("\nsims_reg AttackSRmodel anc vs neg  : ", sims_reg)
            print("\nsims AttackSRmodel anc vs x_adv  : ", sims_x)
            print("loss AttackSRmodel 1-sims(anc,neg) A : ", loss)
            self.loss_pgd.append(loss.detach().clone().numpy())
            self.A.append(loss.detach().clone().numpy())
            loss = ((1 - loss) + 1) / 2  # shift sims to range [0,1] loss = 1-loss
            print("loss 1-loss: ", loss)
            print("loss type: ", type(loss))
            self.B.append(loss.detach().clone().numpy())
            print("1- loss AttackSRmodel B : ", loss)

            loss_score = loss
            print("self.delta before backward: ", self.delta)
            loss.backward()
            print("\nself.delta after backward: ", self.delta)
            self.apply_sign_optimizer.step()
            print("self.delta after opt: ", self.delta)

            # loss_score = loss # src change by shoham
            # loss.backward() # src change by shoham

            with torch.no_grad():
                # Force the gradient step to be a fixed size in a certain norm
                # if step_norm == 'inf':
                gradients = _x_adv.grad.sign()  # * self.eps
                self.delta = self.delta + gradients

                # Untargeted: Gradient ascent on the loss of the correct label w.r.t.
                # the model parameters
                x_adv += self.delta

            # Project back into l_norm ball and correct range
            # if eps_norm == 'inf':
            # Workaround as PyTorch doesn't have elementwise clip
            x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)

            x_adv = x_adv.clamp(-2 ** 15, 2 ** 15 - 1)
        pertubed_embeddings_batch = self.classifier_class.encode_batch(x_adv)
        self.plot_loss_by_parts_A_B()
        print("loss score AttackSRmodel : ", loss_score)
        return pertubed_embeddings_batch, loss_score  # x_adv


class AttackSRModelCustomLossWithSNR:

    def __init__(self, enroll_utters, batch_size=32,
                 aud_length=3, pertb_length=1, fs=16000, eps=0.0001,
                 attack_name='PGD', loss_name='CLossSNR',
                 const_speaker=False):  # 1#3
        self.classifier_class = SRModel.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
        self.model = SRModel.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb").mods.embedding_model

        self.batch_size = batch_size
        self.aud_length = aud_length
        self.pertb_length = pertb_length
        self.fs = fs
        # fs a.k.a sample-rate - 16kHz
        self.max_audio_len = int(fs * aud_length)

        self.delta = {"PGD": torch.tensor(np.zeros((self.pertb_length * fs * 3), dtype=np.float32), requires_grad=True),
                      "FGSM": torch.tensor(np.random.rand((self.pertb_length * fs)), requires_grad=True),
                      "RPGD": (torch.randn(3 * 16000) / 100000).requires_grad_()}[attack_name]
        self.loss_name = attack_name

        self.loss = {'COSE': torch.nn.CosineEmbeddingLoss(),
                     'CRE': torch.nn.CrossEntropyLoss(),
                     'CLOSS': AS2TL2Loss(enroll_utters=enroll_utters),
                     'CLossSNR': AS2TL2LossSNR(enroll_utters=enroll_utters, const_speaker=const_speaker),
                     }[loss_name]
        self.loss_name = loss_name

        params_to_train = [self.delta]
        params_to_train += list(self.classifier_class.mods.embedding_model.parameters())
        self.apply_sign_optimizer = torch.optim.SGD(params_to_train, lr=0.01)
        self.eps = eps
        self.A = []
        self.B = []
        self.loss_pgd = []
        self.loss_all = []
        self.snr_loss = []

    def forward(self, audio_batch, batch_size):
        #################################################
        ### Configuring the UAP with the legit audios ###
        #################################################

        # we tiling the UAP three times
        # DON'T mess with the UAP length and the tiling, because it's overfitting to our usecase - 3 second UAP, tiled
        # 3 times to 9 seconds audio files

        # print(self.aud_length/ self.pertb_length )
        apply_delta = torch.stack([self.delta.clone() for _ in range(int(self.aud_length / self.pertb_length))])  # 3
        apply_delta = torch.reshape(apply_delta, [-1])

        # copying the UAP batch size times, so we can patch it for each audio file in out training
        apply_delta = torch.broadcast_to(apply_delta, [batch_size, apply_delta.shape[0]])
        new_input = audio_batch + self.eps * apply_delta

        # pass_in = torch.clamp(new_input, -2 ** 15, 2 ** 15 - 1) # del regular clip, adding new clip shoham
        pass_in = custom_clip(audio_batch, new_input, self.eps)
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
        # cost = self.loss(torch.squeeze(embeddings_batch,0),
        # torch.squeeze(target_embeddings_batch,0))
        self.model.zero_grad()

        cost.backward(retain_graph=True)

        # we tiling the UAP three times
        # DON'T mess with the UAP length and the tiling, because it's overfitting to our usecase - 3 second UAP, tiled
        # 3 times to 9 seconds audio files
        signed_delta = self.delta.grad.data.sign()
        apply_delta = torch.stack([signed_delta.clone() for _ in range(int(self.aud_length / self.pertb_length))])  # 3
        apply_delta = torch.reshape(apply_delta, [-1])

        # copying the UAP batch size times, so we can patch it for each audio file in out training
        apply_delta = torch.broadcast_to(apply_delta, [batch_size, apply_delta.shape[0]])
        new_input = audio_batch + apply_delta  # * self.eps

        # new_input = torch.max(torch.min(new_input, audio_batch + eps), audio_batch - eps)

        # pass_in = torch.clamp(new_input, -2 ** 15, 2 ** 15 - 1)  # del regular clip, adding new clip shoham
        pass_in = custom_clip(audio_batch, new_input, self.eps)

        # pass_in = new_input
        pertubed_embeddings_batch = self.classifier_class.encode_batch(pass_in)
        return pertubed_embeddings_batch

    def calc__SNR(self, audios_path, uap_path):
        audios = []
        for audio_path in audios_path:
            audio, fs = torchaudio.load(audio_path)  # wavfile.read(os.path.join(audios_path, audio_path))
            audios.append(np.mean(np.abs(audio.detach().numpy())))

        audios_mean_db = np.mean(audios)

        uap, _ = torchaudio.load(uap_path)
        uap_mean_dB = np.mean(np.abs(uap.detach().numpy()))

        SNR = 10 * np.log10(audios_mean_db / uap_mean_dB)
        print(f'{SNR:.2f}')
        return SNR

    def plot_loss_by_parts_A_B(self):
        # visualize loss
        # sns.lineplot(data= s, color = "red")
        # sns.lineplot(data= t, color ="blue")

        # df_loss = pd.DataFrame(self.A, columns =['sims'])
        # df_loss = df_loss.reset_index()
        # sns.lineplot(data=df_loss, x="index", y="sims")
        # # self.B = []
        # del df_loss
        # plt.title("A Sims vs Utterences")
        # plt.ylim(0, 1)
        # plt.show()

        # df_loss = pd.DataFrame([1 - x.detach().numpy() for x in self.A], columns =['loss'])
        # print( " in plot loss  $$$$ ")
        df_loss = pd.DataFrame(self.A, columns=['sims'])
        print("df loss A: ", df_loss)
        df_loss2 = pd.DataFrame(self.B, columns=['sims'])

        # print(df_loss)
        # print(df_loss2)
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

        # # df_loss = pd.DataFrame([1 - x.detach().numpy() for x in self.B], columns =['loss'])
        # df_loss = pd.DataFrame(self.B, columns =['sims'])
        # df_loss = df_loss.reset_index()
        # sns.lineplot(data=df_loss, x="index", y="sims")
        # self.B = []
        # del df_loss
        # plt.title("B Sims vs Utterences")
        # plt.ylim(0, 1)
        # plt.show()

        # sns.lineplot(x="timepoint", y="signal",
        #      hue="region", style="event",
        #      data=fmri)

    # eps=0.3 epx = 0.0001
    # specific attack
    def projected_gradient_descent(self, x, y, signal, num_steps=40, eps=0.3):
        # self.delta.requires_grad = True
        """Performs the projected gradient descent attack on a batch of images."""
        # print("x:" ,x)
        x_adv = x.clone().detach().requires_grad_(True).to(x.device)
        print("x_adv in pgd AttackSRModelCustomLossWithSNR : ", x_adv)
        print("self.delta: ", self.delta)
        for i in tqdm(range(num_steps), desc="Attack step:"):
            # print(f"Step number: {i} out of {num_steps}")
            print("i in pgd : ", i)
            _x_adv = x_adv.clone().detach().requires_grad_(True)
            print("_x_adv: ", _x_adv)

            prediction = self.classifier_class.encode_batch(_x_adv)  # self.forward_clean(_x_adv,1)
            # print("prediction:" ,prediction.shape) $$$
            # print("y:" ,y) $$$
            loss, A_loss, B_loss = self.loss(prediction, y)  # torch.squeeze(y,0)) #CLOSS
            print("A loss:  ", A_loss)
            print("B loss:  ", B_loss)
            print("loss:  ", loss)
            print("_x_adv: ", _x_adv)
            # if i ==0:
            #   loss = loss + ADD_05
            #   print("loss add 0.5:  ",loss)

            # loss = loss + ADD_05/num_steps
            # print("loss add 0.5/num steps:  ",loss)
            print("num_steps:  ", num_steps)

            # loss = 1-loss
            # A_loss = 1 - A_loss
            # B_loss = 1 - B_loss
            # print("A loss 1-loss: ", A_loss)

            self.A.append(A_loss.detach().clone().numpy())

            # print("type Aloss: ", type(A_loss))
            self.B.append(B_loss.detach().clone().numpy())
            # print("B loss 1-loss: ", B_loss)
            # print("loss in pgd 1-loss: ", loss)
            new_input = signal + self.delta.clone() * self.eps  # + SNR_smooth * self.eps
            input_clip = torch.clamp(new_input, -2 ** 15, 2 ** 15 - 1)
            snr_direct = calculator_snr_direct(signal, input_clip)
            # loss = self.loss(torch.squeeze(prediction,0), y,Variable(torch.Tensor([1]))) #COS
            print("snr_direct: ", snr_direct)
            print("loss: ", loss)
            print("input_clip: ", input_clip)
            # snr_square = np.square(snr_direct / 150)
            # print("snr_square: ", snr_square)

            snr_square = snr_direct / 150
            # snr_square = (50 -snr_direct)/100
            print("snr_norm: ", snr_square)
            snr_part = S * snr_square
            loss = loss + snr_part  # (snr_direct / 100 )
            print("loss IN PGD: ", loss)
            # loss = 1 - loss

            self.loss_pgd.append(loss.detach().clone().numpy())
            self.snr_loss.append(snr_direct)

            loss_score = loss
            loss.backward()
            print("loss IN PGD after backward: ", loss)
            print("\nself.delta after backward: ", self.delta)
            self.apply_sign_optimizer.step()
            print("self.delta after opt: ", self.delta)

            with torch.no_grad():
                # Force the gradient step to be a fixed size in a certain norm
                # if step_norm == 'inf':
                gradients = _x_adv.grad.sign()  # * self.eps
                self.delta = self.delta + gradients

                # Untargeted: Gradient ascent on the loss of the correct label w.r.t.
                # the model parameters
                x_adv += self.delta

            # Project back into l_norm ball and correct range
            # if eps_norm == 'inf':
            # Workaround as PyTorch doesn't have elementwise clip

            # x_adv = torch.max(torch.min(x_adv, x + eps), x - eps) # check the diff between
            # x_adv = custom_clip(x,x_adv,self.eps) custom clip, shoham
            x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
            print("self.delta after torch.no_grad() : ", self.delta)

            x_adv = x_adv.clamp(-2 ** 15, 2 ** 15 - 1)  # del regular clip, adding new clip shoham
            print("x_adv after torch.no_grad() : ", x_adv)
            # x_adv = custom_clip(x,x_adv,eps)
            # x_adv = x_adv.clamp(-1,1)
        pertubed_embeddings_batch = self.classifier_class.encode_batch(x_adv)
        print("custom ################################################8")
        self.plot_loss_by_parts_A_B()
        print(" plot")
        return pertubed_embeddings_batch, loss_score  # x_adv



