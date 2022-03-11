import torch
import torch.nn as nn

from lib.dataset.utils import get_data_config, get_imb_num

from .feature_queue import FeatureQueue
from .losses import Triplet_MI_loss
from .semi_model import SemiModel


def euclidean_dist(x, y):
    """
    Args:
      x, y: pytorch Variable, with shape [m, d] and [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    # proto: feats: (B, D), (K, D) -> (B, K)
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


class USADTMModel(SemiModel):

    def __init__(self, cfg):
        super(USADTMModel, self).__init__(cfg)
        self.device = cfg.GPU_ID

        # per-dataset config
        _cfg = get_data_config(cfg)
        imb_num = get_imb_num(
            _cfg.NUM_LABELED_HEAD, _cfg.IMB_FACTOR_L, num_classes=self.num_classes
        )
        self.queue = FeatureQueue(cfg, classwise_max_size=imb_num, bal_queue=False)

        self.dtm_thres = cfg.ALGORITHM.USADTM.DTM_THRES
        self.similarity_fn = nn.CosineSimilarity(dim=2)
        self.uc_loss_weight = cfg.ALGORITHM.USADTM.UC_LOSS_WEIGHT

        self.cfg = cfg
        self.pretraining = True

    def forward(self, x, is_train=False, return_features=False, **kwargs):
        if return_features:
            return self.encoder(x)
        if is_train:
            return self.forward_train(x, **kwargs)
        else:
            return self.forward_test(x)

    def forward_train(self, x, labels, ema_model, dist_logger=None, ul_loss=None, UL_LABELS=None):
        assert ul_loss is not None
        assert UL_LABELS is not None
        pred_class = None
        confidence = None

        loss_dict = {}
        # x: Tensor([l, ul_weak, ul_strong])
        num_labels = labels.size(0)

        # feature vectors
        x = self.encoder(x)
        with torch.no_grad():
            l_feats = x[:num_labels]
            self.queue.enqueue(l_feats.clone().detach(), labels.clone().detach())

        # similarity-based pseudo-label generation
        pred_class = torch.Tensor([-1 for _ in range(len(UL_LABELS))]).long().to(self.device)
        confidence = torch.Tensor([-1 for _ in range(len(UL_LABELS))]).long().to(self.device)
        if not self.pretraining:
            prototypes = self.queue.prototypes  # (K, D)
            _, feats_weak, _ = x[num_labels:].chunk(3)

            with torch.no_grad():
                sim_weak = self.similarity_fn(feats_weak.unsqueeze(1), prototypes.unsqueeze(0))
                confidence, pred_class = torch.max(sim_weak, dim=1)

                dist_weak = euclidean_dist(feats_weak, prototypes)  # (B, K)
                _, dist_pred = torch.min(dist_weak, dim=1)

                loss_weight = torch.eq(pred_class, dist_pred).float()

        logits_concat = self.classifier(x)  # (l, u_i, u_w, u_s)
        l_logits = logits_concat[:num_labels]

        loss_cls = self.l_loss(l_logits, labels)
        loss_dict.update({"loss_cls": loss_cls})

        logits_identity, logits_weak, logits_strong = logits_concat[num_labels:].chunk(3)
        _, lin_pred = torch.max(logits_weak, dim=1)

        if self.uc_loss_weight > 0.0:
            uc_loss = self.uc_loss_weight * Triplet_MI_loss(
                logits_identity, logits_weak, logits_strong
            )
        else:
            uc_loss = torch.Tensor([0]).float().to(self.device)
        loss_dict.update({"uc_loss": uc_loss})

        if self.pretraining:
            loss_cons = torch.Tensor([0]).float().to(self.device)
            # only compute labeled loss
        else:
            loss_weight *= confidence.ge(self.dtm_thres).float()
            loss_cons = ul_loss(
                logits_strong, pred_class, weight=loss_weight, avg_factor=pred_class.size(0)
            )
        loss_dict.update({"loss_cons": loss_cons})
        return loss_dict

    def forward_test(self, x):
        return self.classifier(self.encoder(x))
