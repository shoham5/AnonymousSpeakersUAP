import pandas as pd
import os
import torch
# import cv2
from utils.model_utils import get_speaker_model
from utils.general import get_instance, save_config_to_file, crop_images
# from losses import Loss
from utils.general import preplot, process_imgs, auroc_aupr_scores
import numpy as np
from speechbrain.pretrained import EncoderClassifier


class Attack:
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.cfg.attack_type = self.__class__.__name__

        self.model = get_speaker_model(cfg)

        loss_funcs = []
        s = ''
        for loss_cfg, loss_func_params in zip(self.cfg['losses_config'], self.cfg['loss_func_params'].values()):
            s += loss_cfg['class_name'] + ' '
            loss_func = get_instance(loss_cfg['module_name'],
                                     loss_cfg['class_name'])(**loss_func_params)
            loss_funcs.append(loss_func)
        print('Using the following distance functions in the loss:' + s)
        # loss_func = get_instance(self.cfg['losses_config']['module_name'],
        #                          self.cfg['losses_config']['class_name'])(**self.cfg['loss_func_params'])
        self.postprocess = get_instance(self.cfg['model_config']['model_path'].replace('/', '.') + '.utils', 'postprocess')
        self.loss = Loss(self.model, loss_funcs, self.postprocess, **self.cfg['loss_params'])
        self.cfg['attack_params']['loss_function'] = self.loss.loss_gradient
        self.attack = get_instance(self.cfg['attack_config']['module_name'],
                                   self.cfg['attack_config']['class_name'])(**self.cfg['attack_params'])
        self.estimator = get_classification_model(self.cfg['estimator_config'])
        save_config_to_file(self.cfg, self.cfg['current_dir'])

        if 'transforms' in self.cfg['estimator_config']:
            from torchvision import transforms
            self.cfg['estimator_config']['transforms'] = eval(self.cfg['estimator_config']['transforms'])

        self.gts = {i: [] for i in range(0, self.cfg['estimator_config']['num_of_classes'])}
        self.preds_clean = {i: [] for i in range(0, self.cfg['estimator_config']['num_of_classes'])}
        self.preds_adv = {i: [] for i in range(0, self.cfg['estimator_config']['num_of_classes'])}

        pd.DataFrame(columns=['ground_truth', 'prediction']).\
            to_csv(os.path.join(self.cfg['current_dir'], 'class_results.csv'), index=False)

        pd.DataFrame(columns=[f'iter {str(iteration)}' for iteration in range(self.cfg['attack_params']['max_iter'])]).\
            to_csv(os.path.join(self.cfg['current_dir'], 'loss_results.csv'))

        pd.DataFrame(columns=['clean_acc', 'ut_adv_acc', 't_adv_acc']).\
            to_csv(os.path.join(self.cfg['current_dir'], 'acc_results.csv'))

    def register_loss_values(self, batch_id):
        batch_result = pd.Series([f'batch{batch_id}'] + self.attack.loss_values)
        batch_result.to_frame().T.to_csv(os.path.join(self.cfg['current_dir'], 'loss_results.csv'), mode='a', header=False, index=False)

    @torch.no_grad()
    def compute_success(self, x_clean, x_adv, source_labels, target_labels, batch_id, data):
        rgb_clean = self.model(x_clean)
        rgb_adv = self.model(x_adv)

        rgb_clean = self.postprocess(rgb_clean, **self.cfg['dataset_config']['active_output_indices'], crop=False)
        rgb_adv = self.postprocess(rgb_adv, **self.cfg['dataset_config']['active_output_indices'], crop=False)

        if 'transforms' in self.cfg['estimator_config']:
            rgb_clean = self.cfg['estimator_config']['transforms'](rgb_clean)
            rgb_adv = self.cfg['estimator_config']['transforms'](rgb_adv)

        y_clean = self.estimator(rgb_clean).detach().cpu()
        y_adv = self.estimator(rgb_adv).detach().cpu()

        clean_pred = torch.nn.Softmax(dim=1)(y_clean)
        adv_pred = torch.nn.Softmax(dim=1)(y_adv)
        clean_acc = (clean_pred.argmax(dim=1).type(torch.long) == source_labels).sum().item() / len(y_clean)
        ut_adv_acc = (adv_pred.argmax(dim=1).type(torch.long) == source_labels).sum().item() / len(y_adv)
        t_adv_acc = (adv_pred.argmax(dim=1).type(torch.long) == target_labels).sum().item() / len(y_adv)

        for label in range(0, self.cfg['estimator_config']['num_of_classes']):
            self.gts[label].extend(torch.nn.functional.one_hot(source_labels, num_classes=self.cfg['estimator_config']['num_of_classes'])[:, label].detach().cpu().numpy())
            self.preds_clean[label].extend(clean_pred[:, label].detach().cpu().numpy())
            self.preds_adv[label].extend(adv_pred[:, label].detach().cpu().numpy())

        # with open(os.path.join(self.cfg['current_dir'], 'raw_clean_scores.txt'), 'a') as f:
        #     f.write('\n'.join([str(tuple(arr)) for arr in torch.hstack([torch.stack(clean_pred.max(dim=1), dim=1), source_labels.unsqueeze(-1)]).numpy()]))
        # with open(os.path.join(self.cfg['current_dir'], 'raw_untargeted_scores.txt'), 'a') as f:
        #     f.write('\n'.join([str(tuple(arr)) for arr in torch.hstack([torch.stack(adv_pred.max(dim=1), dim=1), source_labels.unsqueeze(-1)]).numpy()]))
        # with open(os.path.join(self.cfg['current_dir'], 'raw_targeted_scores.txt'), 'a') as f:
        #     f.write('\n'.join([str(tuple(arr)) for arr in torch.hstack([torch.stack(adv_pred.max(dim=1), dim=1), target_labels.unsqueeze(-1)]).numpy()]))
        class_result = pd.DataFrame({'ground_truth': clean_pred.argmax(dim=1).type(torch.long).T.detach().cpu().numpy(),
                                     'prediction': adv_pred.argmax(dim=1).type(torch.long).T.detach().cpu().numpy()})
        class_result.to_csv(os.path.join(self.cfg['current_dir'], 'class_results.csv'), mode='a', index=False, header=False)

        batch_result = pd.Series([f'batch{batch_id}'] + [clean_acc, ut_adv_acc, t_adv_acc])
        batch_result.to_frame().T.to_csv(os.path.join(self.cfg['current_dir'], 'acc_results.csv'),
                                         mode='a',
                                         index=False,
                                         header=False)
        print(f'Clean Accuracy: {round(clean_acc * 100, 2)} | '
              f'Source Label Accuracy: {round(ut_adv_acc * 100, 2)} | '
              f'Target Label Accuracy: {round(t_adv_acc * 100, 2)}')

    def calculate_final_metrics(self):
        metrics_df = pd.read_csv(os.path.join(self.cfg['current_dir'], 'acc_results.csv'),
                                 header=0,
                                 index_col=0)
        clean_acc_mean = metrics_df['clean_acc'].mean()
        ut_acc_mean = metrics_df['ut_adv_acc'].mean()
        t_acc_mean = metrics_df['t_adv_acc'].mean()
        auroc_clean = auroc_aupr_scores(np.stack(list(self.gts.values()), axis=1),
                                        np.stack(list(self.preds_clean.values()), axis=1),
                                        average_types=['macro'])['macro']
        auroc_adv = auroc_aupr_scores(np.stack(list(self.gts.values()), axis=1),
                                      np.stack(list(self.preds_adv.values()), axis=1),
                                      average_types=['macro'])['macro']

        with open(os.path.join(self.cfg['current_dir'], 'final_results.txt'), 'w') as f:
            f.write('Clean Accuracy,Source Label Accuracy,Target Label Accuracy,Clean AuROC,Adv Auroc\n')
            f.write(f'{clean_acc_mean},{ut_acc_mean},{t_acc_mean},{auroc_clean},{auroc_adv}')

        return clean_acc_mean, ut_acc_mean, t_acc_mean, auroc_clean, auroc_adv

    def print_metrics(self, clean_acc_mean, ut_acc_mean, t_acc_mean, auroc_clean, auroc_adv):
        print('Average accuracy for clean images with source labels: {}%'.format(
            str(round(clean_acc_mean * 100, 2))))
        print('Average accuracy for adv images with source labels: {}%'.format(
            str(round(ut_acc_mean * 100, 2))))
        print('Average accuracy for adv images with target labels: {}%'.format(
            str(round(t_acc_mean * 100, 2))))
        print('AuROC for clean images with source labels: {}%'.format(
            str(round(auroc_clean * 100, 2))))
        print('AuROC for adv images with source labels: {}%'.format(
            str(round(auroc_adv * 100, 2))))
