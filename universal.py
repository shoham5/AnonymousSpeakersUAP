import torch
from tqdm import tqdm

from configs.attacks_config import config_dict
from utils.model_utils import get_speaker_model
from utils.general import get_instance, save_config_to_file, calculator_snr_direct, get_pert
from utils.data_utils import get_embeddings, get_loaders


class UniversalAttack:
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.cfg.attack_type = self.__class__.__name__  # for logging purposes when inheriting from another class
        self.model = get_speaker_model(cfg)

        self.train_loader, self.val_loader, _ = get_loaders(self.cfg['loader_params'],
                                                            self.cfg['dataset_config'],
                                                            splits_to_load=['train', 'validation'])

        self.speaker_embeddings = get_embeddings(self.model,
                                                 self.train_loader,
                                                 self.cfg['dataset_config']['speaker_labels_mapper'].keys(),
                                                 self.cfg['device'])

        # Use different distance metrics
        self.loss_fns = []
        for i, (loss_cfg, loss_func_params) in enumerate(zip(self.cfg['losses_config'], self.cfg['loss_func_params'].values())):
            loss_func = get_instance(loss_cfg['module_name'],
                                     loss_cfg['class_name'])(**loss_func_params)
            self.loss_fns.append((loss_func, self.cfg['loss_params']['weights'][i]))

        save_config_to_file(self.cfg, self.cfg['current_dir'])

    def generate(self):
        adv_pert = get_pert(self.cfg['init_pert_type'], size=self.cfg['fs'] * self.cfg['num_of_seconds'])
        optimizer = torch.optim.Adam([adv_pert], lr=self.cfg.start_learning_rate, amsgrad=True)
        scheduler = self.cfg.scheduler_factory(optimizer)
        for epoch in range(self.cfg.epochs):
            running_loss = 0.0
            running_snr = 0.0
            progress_bar = tqdm(enumerate(self.train_loader), desc=f'Epoch {epoch}', total=len(self.train_loader), ncols=150)
            prog_bar_desc = 'Batch Loss: {:.6}, SNR: {:.6}'
            for i_batch, (cropped_signal_batch, person_ids_batch) in progress_bar:
                loss, running_loss = self.forward_step(adv_pert, cropped_signal_batch, person_ids_batch, running_loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                adv_pert.clamp(-1, 1)

                adv_snr = calculator_snr_direct(src_array=cropped_signal_batch, adv_array=self.apply_perturbation(cropped_signal_batch, adv_pert, clip=False))
                running_snr += adv_snr

                progress_bar.set_postfix_str(prog_bar_desc.format(running_loss / (i_batch + 1),
                                                                  running_snr / (i_batch + 1)))

            val_loss = self.evaluate(adv_pert)
            scheduler.step(val_loss)

    def apply_perturbation(self, input, adv_pert, clip=True):
        adv = input + adv_pert  # multiply by epsilon?
        if clip:
            adv.data.clamp_(-1, 1)  # clip?
        return adv

    def forward_step(self, adv_pert, cropped_signal_batch, person_ids_batch, running_loss):
        cropped_signal_batch = cropped_signal_batch.to(self.cfg['device'])
        person_ids_batch = person_ids_batch.to(self.cfg['device'])
        adv_pert = adv_pert.to(self.cfg['device'])

        adv_batch = self.apply_perturbation(cropped_signal_batch, adv_pert)
        adv_embs = self.model.encode_batch(adv_batch)
        loss = self.loss_fn(adv_embs, person_ids_batch)
        running_loss += loss.item()
        return loss, running_loss

    def loss_fn(self, adv_embs, speaker_ids):
        gt_embeddings = torch.index_select(self.speaker_embeddings, index=speaker_ids, dim=0)  # .squeeze(-2)
        loss = torch.zeros(1, device=self.cfg['device'])
        for loss_fn, loss_weight in self.loss_fns:
            loss += loss_weight * loss_fn(adv_embs, gt_embeddings).squeeze().mean()
        return loss

    def evaluate(self, adv_pert):
        running_loss = 0.0
        running_snr = 0.0
        progress_bar = tqdm(enumerate(self.val_loader), desc=f'Eval', total=len(self.val_loader), ncols=150)
        prog_bar_desc = 'Batch Loss: {:.6}, SNR: {:.6}'
        for i_batch, (cropped_signal_batch, person_ids_batch) in progress_bar:
            loss, running_loss = self.forward_step(adv_pert, cropped_signal_batch, person_ids_batch, running_loss)
            adv_snr = calculator_snr_direct(src_array=cropped_signal_batch,
                                            adv_array=self.apply_perturbation(cropped_signal_batch, adv_pert, clip=False))
            running_snr += adv_snr
            progress_bar.set_postfix_str(prog_bar_desc.format(running_loss / (i_batch + 1),
                                                              running_snr / (i_batch + 1)))

        return running_loss / len(self.val_loader)


def main():
    config_type = 'Universal'
    cfg = config_dict[config_type]()
    attack = UniversalAttack(cfg)
    attack.generate()


if __name__ == '__main__':
    main()
