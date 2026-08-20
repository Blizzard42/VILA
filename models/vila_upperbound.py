"""e_48 vila_upperbound — "VILA but it's cheating": keep VILA's task-0
adapter training and frozen ViT+CLIP feature pipeline untouched, but replace
the analytic head training (_cls_align ridge solve + _IL_align RLS) with a
pluggable TestModel head trained by minibatch SGD (MSE to one-hot) on ALL
seen data — features cached once per task from a single train-mode
augmentation draw per sample, the same information the analytic solve saw.

Head interface (acil_upperbound convention, adapted to the two-branch net):
    __init__(in_features, args, device)
    preprocess(network, images, clip_images) -> features   [no_grad, cached]
    append_task(num_new_classes)                           [zero-init growth]
    forward(features) -> {"logits": ..., "buffer_feature": ...}
Heads are registered in HEADS and selected via the "head_model" config key.
forward() must accept the concatenated normalized (backbone, CLIP) features,
because eval flows through VILA.forward -> ac_model(features) unchanged.

v0 "vila-mimic" is AC_Linear with the solve swapped for SGD: frozen random
Linear(d, Hidden) -> ReLU -> growing bias-free Linear (the only trainable
part, exactly the weights VILA sets analytically).

"relu" and "agalu" are the e_47 acil_upperbound 1-hidden-layer heads ported
to this interface: relu -> h = relu(W1 x), agalu -> h = 1[Bg x>0] (o) (W1 x)
(Bg frozen); both size the hidden layer with the existing "Hidden" key and
keep W1 across tasks (fresh optimizer per task, as always).

New config keys (all optional): head_model, head_lr, head_epochs,
head_schedule, head_momentum, head_opt, head_wd, head_batch_size,
head_eval_every, head_dropout, head_input_noise (train-time gaussian noise on
the cached features, sigma in per-dim std units of the seen cache).
"""
import logging

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from models.vila import Learner as VilaLearner, num_workers


# --------------------------------------------------------------------------
# Heads
# --------------------------------------------------------------------------


class VilaMimicHead(nn.Module):
    """AC_Linear trained by SGD: fc0 is the same random frozen expansion
    (nn.Linear default init, VILA never trains it — update_fc only carries it
    forward), the last layer grows per task with zero-init new rows (mirrors
    update_fc's zero-padding); optional dropout (head_dropout) on h before
    the growing head, same placement as the other heads."""

    def __init__(self, in_features, args, device):
        super().__init__()
        self.fc0 = nn.Linear(in_features, args["Hidden"], bias=False, device=device)
        self.fc0.weight.requires_grad_(False)
        drop = args.get("head_dropout", 0.0)
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()
        self.head = nn.Linear(args["Hidden"], 0, bias=False, device=device)

    @torch.no_grad()
    def preprocess(self, network, images, clip_images):
        return network(images, clip_images)["features"]

    @torch.no_grad()
    def append_task(self, num_new_classes):
        old = self.head.weight
        new = nn.Linear(self.head.in_features, old.shape[0] + num_new_classes,
                        bias=False, device=old.device)
        new.weight.zero_()
        new.weight[: old.shape[0]] = old
        self.head = new

    def forward(self, features):
        h = F.relu(self.fc0(features))
        return {"buffer_feature": h, "logits": self.head(self.drop(h))}


class ReLUHead(nn.Module):
    """e_47 acil_upperbound ReLU port: 1 trainable hidden layer,
    h = relu(W1 x), W1 ~ N(0, 1/d), width = args["Hidden"]; optional dropout
    (head_dropout) before the growing bias-free head."""

    def __init__(self, in_features, args, device):
        super().__init__()
        kh = args["Hidden"]
        self.W1 = nn.Parameter(
            torch.randn(kh, in_features, device=device) / in_features ** 0.5)
        drop = args.get("head_dropout", 0.0)
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()
        self.head = nn.Linear(kh, 0, bias=False, device=device)

    @torch.no_grad()
    def preprocess(self, network, images, clip_images):
        return network(images, clip_images)["features"]

    @torch.no_grad()
    def append_task(self, num_new_classes):
        old = self.head.weight
        new = nn.Linear(self.head.in_features, old.shape[0] + num_new_classes,
                        bias=False, device=old.device)
        new.weight.zero_()
        new.weight[: old.shape[0]] = old
        self.head = new

    def forward(self, features):
        h = F.relu(features @ self.W1.T)
        return {"buffer_feature": h, "logits": self.head(self.drop(h))}


class AGaLUHead(nn.Module):
    """e_47 acil_upperbound BasicAGaLU port: 1 gated hidden layer,
    h = 1[Bg x > 0] (o) (W1 x) with Bg frozen ~ N(0, 1/d) and W1 trainable
    ~ N(0, 1/d), width = args["Hidden"]; optional dropout (head_dropout)."""

    def __init__(self, in_features, args, device):
        super().__init__()
        kh = args["Hidden"]
        self.register_buffer(
            "Bg", torch.randn(kh, in_features, device=device) / in_features ** 0.5)
        self.W1 = nn.Parameter(
            torch.randn(kh, in_features, device=device) / in_features ** 0.5)
        drop = args.get("head_dropout", 0.0)
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()
        self.head = nn.Linear(kh, 0, bias=False, device=device)

    @torch.no_grad()
    def preprocess(self, network, images, clip_images):
        return network(images, clip_images)["features"]

    @torch.no_grad()
    def append_task(self, num_new_classes):
        old = self.head.weight
        new = nn.Linear(self.head.in_features, old.shape[0] + num_new_classes,
                        bias=False, device=old.device)
        new.weight.zero_()
        new.weight[: old.shape[0]] = old
        self.head = new

    def forward(self, features):
        h = (features @ self.Bg.T > 0).to(features.dtype) * (features @ self.W1.T)
        return {"buffer_feature": h, "logits": self.head(self.drop(h))}


class AGaLU3Head(nn.Module):
    """e_47 acil_upperbound AGaLU3 port: 3-layer AGaLU, both hidden layers
    width kh = args["Hidden"], both gates anchored to the TRUE input:
        h1 = 1[Bg1 x > 0] (o) (W1 x)
        h2 = 1[Bg2 x > 0] (o) (W2 h1)
    Bg1/Bg2 frozen ~ N(0, 1/d); W1 ~ N(0, 1/d), W2 ~ N(0, 1/kh) trainable
    (variance-preserving for their input dims); optional dropout before the
    growing bias-free head."""

    def __init__(self, in_features, args, device):
        super().__init__()
        kh = args["Hidden"]
        self.register_buffer(
            "Bg1", torch.randn(kh, in_features, device=device) / in_features ** 0.5)
        self.register_buffer(
            "Bg2", torch.randn(kh, in_features, device=device) / in_features ** 0.5)
        self.W1 = nn.Parameter(
            torch.randn(kh, in_features, device=device) / in_features ** 0.5)
        self.W2 = nn.Parameter(
            torch.randn(kh, kh, device=device) / kh ** 0.5)
        drop = args.get("head_dropout", 0.0)
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()
        self.head = nn.Linear(kh, 0, bias=False, device=device)

    @torch.no_grad()
    def preprocess(self, network, images, clip_images):
        return network(images, clip_images)["features"]

    @torch.no_grad()
    def append_task(self, num_new_classes):
        old = self.head.weight
        new = nn.Linear(self.head.in_features, old.shape[0] + num_new_classes,
                        bias=False, device=old.device)
        new.weight.zero_()
        new.weight[: old.shape[0]] = old
        self.head = new

    def forward(self, features):
        h1 = (features @ self.Bg1.T > 0).to(features.dtype) * (features @ self.W1.T)
        h2 = (features @ self.Bg2.T > 0).to(features.dtype) * (h1 @ self.W2.T)
        return {"buffer_feature": h2, "logits": self.head(self.drop(h2))}


class AGaLU5Head(nn.Module):
    """AGaLU3 extended to depth 5: 4 gated hidden layers, all width
    kh = args["Hidden"], every gate anchored to the TRUE input:
        h1 = 1[Bg1 x > 0] (o) (W1 x)
        h_i = 1[Bg_i x > 0] (o) (W_i h_{i-1})   i = 2..4
    Bg1..Bg4 frozen ~ N(0, 1/d); W1 ~ N(0, 1/d), W2..W4 ~ N(0, 1/kh)
    trainable (variance-preserving for their input dims); optional dropout
    before the growing bias-free head (the 5th layer)."""

    def __init__(self, in_features, args, device):
        super().__init__()
        kh = args["Hidden"]
        for i in range(1, 5):
            self.register_buffer(
                f"Bg{i}",
                torch.randn(kh, in_features, device=device) / in_features ** 0.5)
        self.W1 = nn.Parameter(
            torch.randn(kh, in_features, device=device) / in_features ** 0.5)
        for i in range(2, 5):
            setattr(self, f"W{i}", nn.Parameter(
                torch.randn(kh, kh, device=device) / kh ** 0.5))
        drop = args.get("head_dropout", 0.0)
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()
        self.head = nn.Linear(kh, 0, bias=False, device=device)

    @torch.no_grad()
    def preprocess(self, network, images, clip_images):
        return network(images, clip_images)["features"]

    @torch.no_grad()
    def append_task(self, num_new_classes):
        old = self.head.weight
        new = nn.Linear(self.head.in_features, old.shape[0] + num_new_classes,
                        bias=False, device=old.device)
        new.weight.zero_()
        new.weight[: old.shape[0]] = old
        self.head = new

    def forward(self, features):
        h = (features @ self.Bg1.T > 0).to(features.dtype) * (features @ self.W1.T)
        for i in range(2, 5):
            gate = (features @ getattr(self, f"Bg{i}").T > 0).to(features.dtype)
            h = gate * (h @ getattr(self, f"W{i}").T)
        return {"buffer_feature": h, "logits": self.head(self.drop(h))}


HEADS = {"vila-mimic": VilaMimicHead, "relu": ReLUHead, "agalu": AGaLUHead,
         "agalu3": AGaLU3Head, "agalu5": AGaLU5Head}


# --------------------------------------------------------------------------
# Learner
# --------------------------------------------------------------------------


class Learner(VilaLearner):

    def __init__(self, args):
        super().__init__(args)
        assert len(self._multiple_gpus) == 1, "vila_upperbound is single-GPU"
        self.head = None
        self._seen_feats = self._seen_labels = None
        self._test_feats = self._test_labels = None
        self.head_lr = args.get("head_lr", 1e-3)
        self.head_epochs = args.get("head_epochs", 100)
        self.head_schedule = args.get("head_schedule", "constant")
        self.head_momentum = args.get("head_momentum", 0.9)
        self.head_opt = args.get("head_opt", "sgd")
        self.head_wd = args.get("head_wd", 0.0)
        self.head_batch_size = args.get("head_batch_size", 4096)
        self.head_eval_every = args.get("head_eval_every", 0)
        # CSVs land next to trainer.py's log file, same naming scheme
        init_cls = 0 if args["init_cls"] == args["increment"] else args["init_cls"]
        stem = "logs/{}/{}/{}/{}/{}_{}_{}".format(
            args["model_name"], args["dataset"], init_cls, args["increment"],
            args["prefix"], args["seed"], args["backbone_type"])
        self._head_log = open(stem + "_head_curves.csv", "w", buffering=1)
        print("task", "epoch", "lr", "train_loss", "train_acc", "test_acc",
              file=self._head_log, sep=",")

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train", mode="train")
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                       shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size,
                                      shuffle=False, num_workers=num_workers)

        if self._cur_task == 0:
            self._init_train(self.train_loader, self.test_loader)
            in_features = self.feature_dim + self._network.clip.out_dim
            self.head = HEADS[self.args.get("head_model", "vila-mimic")](
                in_features, self.args, self._device)
            self._network.ac_model = self.head  # eval flows through it unchanged
        self.head.append_task(self._total_classes - self._known_classes)
        self._cache_task_features()
        self._train_head()

    @torch.no_grad()
    def _cache_task_features(self):
        """One train-mode (augmented) draw per new-task sample -> seen cache;
        clean test features of the new classes -> test cache (for the cheap
        per-epoch eval)."""
        self._network.to(self._device)
        self._network.eval()
        feats, labels = [], []
        for _, data, clip_data, label in self.train_loader:
            feats.append(self.head.preprocess(
                self._network, data.to(self._device), clip_data.to(self._device)))
            labels.append(label.to(self._device))
        feats, labels = torch.cat(feats), torch.cat(labels)
        self._seen_feats = (feats if self._seen_feats is None
                            else torch.cat([self._seen_feats, feats]))
        self._seen_labels = (labels if self._seen_labels is None
                             else torch.cat([self._seen_labels, labels]))
        if self.head_eval_every > 0:
            new_test = self.data_manager.get_dataset(
                np.arange(self._known_classes, self._total_classes),
                source="test", mode="test")
            loader = DataLoader(new_test, batch_size=self.batch_size,
                                shuffle=False, num_workers=num_workers)
            feats, labels = [], []
            for _, data, clip_data, label in loader:
                feats.append(self.head.preprocess(
                    self._network, data.to(self._device), clip_data.to(self._device)))
                labels.append(label.to(self._device))
            feats, labels = torch.cat(feats), torch.cat(labels)
            self._test_feats = (feats if self._test_feats is None
                                else torch.cat([self._test_feats, feats]))
            self._test_labels = (labels if self._test_labels is None
                                 else torch.cat([self._test_labels, labels]))

    @torch.no_grad()
    def _eval_cached(self, text_features):
        """_eval_cnn's metric (0.8 adpt + 0.2 clip-rerank) from the cached
        clean test features; the CLIP branch is the last clip.out_dim dims of
        the concatenated feature."""
        clip_dim = self._network.clip.out_dim
        hits = 0
        for i in range(0, self._test_feats.shape[0], self.head_batch_size):
            f = self._test_feats[i:i + self.head_batch_size]
            adpt_logits = self.head(f)["logits"]
            clip_logits = f[:, -clip_dim:] @ text_features.T
            rerank_logits = self.clip_rerank(adpt_logits, clip_logits,
                                             topk=self.args["rerank_topk"])
            logits = adpt_logits * 0.8 + rerank_logits * 0.2
            hits += (logits.argmax(dim=1)
                     == self._test_labels[i:i + self.head_batch_size]).sum().item()
        return hits / self._test_feats.shape[0]

    def _train_head(self):
        """The cheat: minibatch SGD, MSE to one-hot, over the FULL seen
        feature cache, fresh optimizer per task (constant or cosine lr)."""
        params = [p for p in self.head.parameters() if p.requires_grad]
        if self.head_opt == "adamw":
            opt = optim.AdamW(params, lr=self.head_lr, weight_decay=self.head_wd)
        else:
            opt = optim.SGD(params, lr=self.head_lr, momentum=self.head_momentum,
                            weight_decay=self.head_wd)
        sched = (optim.lr_scheduler.CosineAnnealingLR(
                     opt, T_max=self.head_epochs, eta_min=0.0)
                 if self.head_schedule == "cosine" else None)
        text_features = None
        if self.head_eval_every > 0:
            text_features = self._feat_from_temp()
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        n = self._seen_feats.shape[0]
        # train-time gaussian input noise, sigma in per-dim std units of the
        # (growing) seen cache; resampled every minibatch
        in_noise = self.args.get("head_input_noise", 0.0)
        feat_std = self._seen_feats.std(dim=0) if in_noise > 0 else None
        self.head.train()  # dropout active only while fitting
        for epoch in range(1, self.head_epochs + 1):
            lr_now = opt.param_groups[0]["lr"]
            perm = torch.randperm(n, device=self._device)
            loss_sum = correct = 0
            for i in range(0, n, self.head_batch_size):
                idx = perm[i:i + self.head_batch_size]
                X, y = self._seen_feats[idx], self._seen_labels[idx]
                if in_noise > 0:
                    X = X + in_noise * feat_std * torch.randn_like(X)
                opt.zero_grad(set_to_none=True)
                logits = self.head(X)["logits"]
                loss = F.mse_loss(logits, F.one_hot(
                    y, self._total_classes).to(logits.dtype))
                loss.backward()
                opt.step()
                loss_sum += loss.item() * y.numel()
                correct += (logits.argmax(dim=1) == y).sum().item()
            if sched is not None:
                sched.step()
            train_loss, train_acc = loss_sum / n, correct / n
            test_acc = ""
            if self.head_eval_every > 0 and (epoch % self.head_eval_every == 0
                                             or epoch == self.head_epochs):
                self.head.eval()
                test_acc = self._eval_cached(text_features)
                self.head.train()
            print(self._cur_task, epoch, lr_now, train_loss, train_acc, test_acc,
                  file=self._head_log, sep=",")
            if epoch % max(1, self.head_epochs // 5) == 0 or epoch == self.head_epochs:
                logging.info(
                    "task {} head epoch {}/{} lr {:.5f} train_mse {:.6f} "
                    "train_acc {:.4f} test_acc {}".format(
                        self._cur_task, epoch, self.head_epochs, lr_now,
                        train_loss, train_acc, test_acc))
        self.head.eval()
