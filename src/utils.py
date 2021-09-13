import numpy as np

import torch
import torchvision

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from . import pytorch_utils as ptu
from . import mean_ap


class_names = ['', 'NoMask', 'BadMask', 'Mask']
class_colors = [(0, 0, 0), (255, 0, 0), (0, 0, 255), (0, 255, 0)]
class_colors_str = ['w', 'r', 'b', 'g']


class Config:
    """ a simple class for managing experiment setup """
    def __call__(self):
        return vars(self)

    def __repr__(self):
        return str(self())

    def __str__(self):
        return self.__repr__()


def plot_image(img, annotation):
    fig, ax = plt.subplots(1)
    plt.imshow(img.to(torch.uint8).cpu().permute(1, 2, 0))
    if 'scores' in annotation.keys():
        scores = annotation["scores"].cpu()
    else:
        scores = np.ones(annotation["labels"].shape)
    for box, label, score in zip(annotation["boxes"].cpu(), annotation["labels"].cpu(), scores):
        if score < 0.5:
            continue
        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=2, edgecolor=class_colors_str[label], facecolor='none')
        ax.add_patch(rect)
    plt.show()


class MyCheckpoint(ptu.Checkpoint):
    def batch_pass(self,
                   device,
                   batch,
                   *args, **kwargs):

        results = {}
        pbar_postfix = {}

        imgs, targets = batch

        imgs = [i.to(device) for i in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        self.batch_size = len(imgs)

        if self.model.training:
            losses = self.model(imgs, targets)
            loss = sum(losses.values())
            
            loss_vals = {k: float(v.data) for k, v in losses.items()}
            
            results.update(loss_vals)
            pbar_postfix.update(loss_vals)
        else:
            outs = self.model(imgs, targets)

            for out in outs:
                for k, v in out.items():
                    if k not in results:
                        results[k] = []
                    results[k].append(v.detach().cpu())

            for target in targets:
                for k, v in target.items():
                    k = k + '_true'
                    if k not in results:
                        results[k] = []
                    results[k].append(v.detach().cpu())

            maPs = []
            for target, out in zip(targets, outs):
                maPs.append(float(mean_ap.calculate_map(target['boxes'].detach().cpu(), out['boxes'].detach().cpu(), out['scores'].detach().cpu())))

            loss = torch.tensor(float('nan'))
            score = sum(maPs) / len(maPs)

            results['score'] = score

            pbar_postfix['score'] = score
            if len(self.raw_results) > 0:
                pbar_postfix['avg_score'] = np.array(self.raw_results['score']).mean()

        return loss, results, pbar_postfix

    def agg_results(self, results):
        single_num_score = None
        additional_metrics = {}
        single_num_score = 0.0

        if self.model.training:
            for k, v in results.items():
                additional_metrics[k] = np.array(results[k]).mean()
        else:
            single_num_score = np.array(results['score']).mean()

        return single_num_score, additional_metrics

