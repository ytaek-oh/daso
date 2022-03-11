import torch
import torch.nn as nn
import torch.nn.functional as F

from .classifier import Classifier
from .feature_queue import FeatureQueue
from .semi_model import SemiModel


class DASOModel(SemiModel):

    def __init__(self, cfg):
        super(DASOModel, self).__init__(cfg)
        self.device = cfg.GPU_ID

        # balanced queue
        self.queue = FeatureQueue(cfg, classwise_max_size=None, bal_queue=True)
        self.conf_thres = cfg.ALGORITHM.CONFIDENCE_THRESHOLD

        self.similarity_fn = nn.CosineSimilarity(dim=2)

        self.T_proto = cfg.ALGORITHM.DASO.PROTO_TEMP
        self.pretraining = True
        self.psa_loss_weight = cfg.ALGORITHM.DASO.PSA_LOSS_WEIGHT

        self.T_dist = cfg.ALGORITHM.DASO.DIST_TEMP
        self.with_dist_aware = cfg.ALGORITHM.DASO.WITH_DIST_AWARE
        self.interp_alpha = cfg.ALGORITHM.DASO.INTERP_ALPHA

        self.cfg = cfg

        # logit adjustment
        self.with_la = cfg.ALGORITHM.LOGIT_ADJUST.APPLY  # train la
        self.tau = cfg.ALGORITHM.LOGIT_ADJUST.TAU

        self.num_classes = cfg.MODEL.NUM_CLASSES

        # ABC options
        self.with_abc = cfg.ALGORITHM.ABC.APPLY
        if self.with_abc:
            self.abc_classifier = Classifier(self.out_features, self.num_classes)
            self.with_daso_pl = cfg.ALGORITHM.ABC.DASO_PSEUDO_LABEL

    def forward(self, x, is_train=False, return_features=False, **kwargs):
        if return_features:
            return self.encoder(x)
        if is_train:
            return self.forward_train(x, **kwargs)
        else:
            return self.forward_test(x)

    def forward_train(
        self, x, labels=None, ema_model=None, dist_logger=None, ul_loss=None, UL_LABELS=None
    ):
        assert dist_logger is not None
        assert ul_loss is not None
        assert UL_LABELS is not None

        pred_class = None

        loss_dict = {}
        logger_dict = {"gt_labels": labels, "ul_labels": UL_LABELS}  # initial log
        num_labels = labels.size(0)

        # push memory queue
        with torch.no_grad():
            l_feats = ema_model(x[:num_labels], return_features=True)
            self.queue.enqueue(l_feats.clone().detach(), labels.clone().detach())

        # feature vectors
        x = self.encoder(x)

        # initial empty assignment
        assignment = torch.Tensor([-1 for _ in range(len(UL_LABELS))]).float().to(self.device)
        if not self.pretraining:
            prototypes = self.queue.prototypes  # (K, D)
            feats_weak, feats_strong = x[num_labels:].chunk(2)  # (B, D)

            with torch.no_grad():
                # similarity between weak features and prototypes  (B, K)
                sim_weak = self.similarity_fn(
                    feats_weak.unsqueeze(1), prototypes.unsqueeze(0)
                ) / self.T_proto
                soft_target = sim_weak.softmax(dim=1)
                assign_confidence, assignment = torch.max(soft_target.detach(), dim=1)

            # soft loss
            if self.psa_loss_weight > 0:
                # similarity between strong features and prototypes  (B, K)
                sim_strong = self.similarity_fn(
                    feats_strong.unsqueeze(1), prototypes.unsqueeze(0)
                ) / self.T_proto

                loss_assign = -1 * torch.sum(soft_target * F.log_softmax(sim_strong, dim=1),
                                             dim=1).sum() / sim_weak.size(0)
                loss_dict.update({"loss_assign": self.psa_loss_weight * loss_assign})
        logger_dict.update({"sem_pl": assignment})  # semantic pl

        if self.with_abc:
            # abc classification pipeline
            # logits for ABC
            logits_concat_abc = self.abc_classifier(x)
            l_logits_abc = logits_concat_abc[:num_labels]

            l_mask_abc = torch.bernoulli(self.bal_param[labels].detach()).float()
            cls_loss_abc = self.l_loss(l_logits_abc, labels, weight=l_mask_abc)

            # unlabeled data part
            logits_weak_abc, logits_strong_abc = logits_concat_abc[num_labels:].chunk(2)
            p_abc = logits_weak_abc.detach_().softmax(dim=1)  # soft pseudo labels
            # mix with DASO?
            if self.with_daso_pl:
                with torch.no_grad():
                    if not self.pretraining:
                        conf_abc, pred_class_abc = torch.max(p_abc, dim=1)

                        current_pl_dist = dist_logger.get_pl_dist().to(self.device)  # (1, C)
                        current_pl_dist = current_pl_dist**(1. / self.T_dist)
                        current_pl_dist = current_pl_dist / current_pl_dist.sum()
                        current_pl_dist = current_pl_dist / current_pl_dist.max()  # MIXUP

                        pred_to_dist = current_pl_dist[pred_class_abc].view(-1, 1)  # (B, )

                        # pl mixup
                        p_abc = (1. - pred_to_dist) * p_abc + pred_to_dist * soft_target

            conf_abc, pred_class_abc = torch.max(p_abc, dim=1)
            loss_weight_abc = conf_abc.ge(self.conf_thres).float()

            # mask generation
            current_epoch = int(self.iter / 500)  # 0~499
            gradual_bal_param = 1.0 - (current_epoch / 500) * (1.0 - self.bal_param)
            ul_mask_abc = torch.bernoulli(gradual_bal_param[pred_class_abc].detach()).float()

            # mask consistency loss with soft pseudo-label
            abc_mask = loss_weight_abc * ul_mask_abc
            cons_loss_abc = -1 * torch.mean(
                abc_mask * torch.sum(p_abc * F.log_softmax(logits_strong_abc, dim=1), dim=1)
            )

            abc_loss = cls_loss_abc + cons_loss_abc
            loss_dict.update({"loss_abc": abc_loss})

        # fixmatch pipelines
        logits_concat = self.classifier(x)
        l_logits = logits_concat[:num_labels]

        # logit_adjust for train-time.
        if self.with_la:
            assert self.target_dist is not None
            l_logits += (self.tau * self.target_dist.view(1, -1).log())

        # loss computation
        loss_cls = self.l_loss(l_logits, labels)
        loss_dict.update({"loss_cls": loss_cls})

        logits_weak, logits_strong = logits_concat[num_labels:].chunk(2)
        with torch.no_grad():
            # compute pseudo-label
            p = logits_weak.softmax(dim=1)  # soft pseudo labels
            confidence, pred_class = torch.max(p.detach(), dim=1)  # (B, 1)
            logger_dict.update({"linear_pl": pred_class})  # linear pl

            if not self.pretraining:
                current_pl_dist = dist_logger.get_pl_dist().to(self.device)  # (1, C)
                current_pl_dist = current_pl_dist**(1. / self.T_dist)
                current_pl_dist = current_pl_dist / current_pl_dist.sum()
                current_pl_dist = current_pl_dist / current_pl_dist.max()  # MIXUP

                pred_to_dist = current_pl_dist[pred_class].view(-1, 1)  # (B, )
                if not self.with_dist_aware:
                    pred_to_dist = self.interp_alpha  # override to fixed constant

                # pl mixup
                p = (1. - pred_to_dist) * p + pred_to_dist * soft_target

            confidence, pred_class = torch.max(p.detach(), dim=1)  # final pl
            logger_dict.update({"pseudo_labels": pred_class, "pl_confidence": confidence})
            dist_logger.accumulate(logger_dict)
            dist_logger.push_pl_list(pred_class)

            loss_weight = confidence.ge(self.conf_thres).float()

        loss_cons = ul_loss(
            logits_strong, pred_class, weight=loss_weight, avg_factor=pred_class.size(0)
        )
        loss_dict.update({"loss_cons": loss_cons})

        return loss_dict

    def forward_test(self, x):
        classifier = self.abc_classifier if self.with_abc else self.classifier
        return classifier(self.encoder(x))
