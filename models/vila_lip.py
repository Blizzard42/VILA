"""e_48 vila_lip -- vila_upperbound + a fixed-size compressed memory: the head
never trains on the accumulated feature cache; at every task boundary the
seen history is re-compressed to m points and the head trains on THOSE alone.
Max storage at any instant = m memory points + ONE task's feature cache.

    task 0: cache task-0 features; train head on the full cache (verbatim
            vila_upperbound, B.5); then compress cache -> memory_0, discard.
    task t: cache the new task's features; build memory_t from
                target = w_old * moments(memory_{t-1}) + w_new * moments(new)
                (w = N_part / N_seen; moments are linear in the distribution)
            discard the cache; train head on memory_t alone (fresh optimizer,
            warm-started head weights, unchanged loop).

Two memory modes ("memory_mode"):
  lip      free-atom (M, V) moment distillation in the head's own AGaLU
           feature space (models/lip_fit.py, the exp_1 canonical math with a
           weighted target).  Atoms warm-start at the previous memory (the
           first fit inits at a plain uniform draw of m task-0 rows); targets
           y are free fp32 vectors in plain one-hot space (vila convention),
           zero-padded as classes grow -- mirroring the head's zero-init row
           growth.
  coreset  the selection control: proportional draw, round(w_old*m) points
           kept uniformly from the old memory + the rest drawn uniformly from
           the new rows.  Real rows, integer labels.

STEP-MATCHED training (B.7): the head runs the same number of gradient steps
the full-cache run would --  head_epochs * ceil(N_seen / bs)  -- re-expressed
as epochs over the m-point memory (m <= bs makes each epoch one full-batch
step).  Task 0 reduces to the parent budget exactly.  head_input_noise sigma
is in per-dim std units of whatever the head trains on (here: the memory).

New config keys: memory_mode (lip|coreset), memory_m, lip_fit_steps,
lip_fit_lr (RELATIVE), lip_lambda_mv, lip_fit_eval_every, lip_fit_adam_eps.
LIP fit curves land in <stem>_lip_curves.csv beside the head curves.
"""
import logging

import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from models.vila import num_workers
from models.vila_upperbound import HEADS, Learner as UpperboundLearner
from models.lip_fit import make_target, fit_lip, LIP_CURVE


class Learner(UpperboundLearner):

    def __init__(self, args):
        super().__init__(args)
        self.memory_mode = args.get("memory_mode", "lip")
        assert self.memory_mode in ("lip", "coreset", "joint"), self.memory_mode
        if self.memory_mode in ("lip", "joint"):
            assert args.get("head_model") == "agalu", \
                "lip memory prices the AGaLU feature space (needs head.Bg)"
        self.final_fresh_retrain = args.get("final_fresh_retrain", False)
        self.memory_m = args.get("memory_m", 500)
        self.lip_fit_steps = args.get("lip_fit_steps", 48000)
        self.lip_fit_lr = args.get("lip_fit_lr", 1e-3)
        self.lip_lambda_mv = args.get("lip_lambda_mv", 0.5)
        self.lip_fit_eval_every = args.get("lip_fit_eval_every", 800)
        self.lip_fit_adam_eps = args.get("lip_fit_adam_eps", 1e-8)
        # wave-4 fit knobs (defaults reproduce waves 1-3 exactly)
        self.lip_fit_mb = args.get("lip_fit_mb", 0)
        self.lip_fit_jitter = args.get("lip_fit_jitter", 0.0)
        self.lip_fit_snapshot_best = args.get("lip_fit_snapshot_best", False)
        self.lip_fit_init = args.get("lip_fit_init", "warm")
        assert self.lip_fit_init in ("warm", "rand"), self.lip_fit_init
        # wave-7 knobs (defaults reproduce waves 1-6 exactly)
        self.lip_fit_sched = args.get("lip_fit_sched", "const")
        self.lip_centred_y = args.get("lip_centred_y", False)
        ss = args.get("lip_snap_steps", []) or []
        if isinstance(ss, str):                # "12000,24000" (glob-safe form)
            ss = [x for x in ss.split(",") if x]
        self.lip_snap_steps = [int(x) for x in ss]
        self.memory_save = args.get("memory_save", False)
        self.head_from_memory = args.get("head_from_memory", "")
        self.head_from_snap = int(args.get("head_from_snap", 0))  # 0 = final
        self.head_budget_mult = int(args.get("head_budget_mult", 1))
        if self.memory_mode == "joint":
            assert self.lip_fit_mb > 0, \
                "joint mode fits a full-dataset target: mb estimator required"
        self._seen_count = 0            # real rows seen so far (the weights)
        self._mem_X = self._mem_Y = None  # Y: fp32 [m, C] (lip) | int labels
        self._lip_log = None
        if self.memory_mode in ("lip", "joint"):
            init_cls = 0 if args["init_cls"] == args["increment"] else args["init_cls"]
            stem = "logs/{}/{}/{}/{}/{}_{}_{}".format(
                args["model_name"], args["dataset"], init_cls, args["increment"],
                args["prefix"], args["seed"], args["backbone_type"])
            self._stem = stem
            self._lip_log = open(stem + "_lip_curves.csv", "w", buffering=1)
            print("task", *LIP_CURVE, file=self._lip_log, sep=",")

    # ---------------------------------------------------------------- flow
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
            self._network.ac_model = self.head
        self.head.append_task(self._total_classes - self._known_classes)

        last = self._cur_task == data_manager.nb_tasks - 1
        if self.memory_mode == "joint":
            # STEP-ABLATION (wave 6): keep the WHOLE cache (parent append);
            # the head never trains until the final task, where the full seen
            # data is distilled in ONE fit and a never-trained head trains on
            # the memory alone, step-matched to the recursive final task.
            # Wave 7 additions: lip_centred_y (target = one-hot - pi, the
            # exp_1 convention; balanced classes make argmax invariant),
            # lip_snap_steps (mid-fit memories, each trained on a FRESH head,
            # tasks 71,72,... in the head CSV), memory_save (payload .pt),
            # head_from_memory (stage B: no cache, no fit -- load a payload
            # and train the head alone).
            if self.head_from_memory:
                self._cache_test_features()   # eval set only, no train cache
                if last:
                    self._head_from_memory()
                return
            self._cache_task_features()
            self._seen_count = self._seen_feats.shape[0]
            if last:
                m, C = self.memory_m, self._total_classes
                X = self._seen_feats
                Y = F.one_hot(self._seen_labels, C).double()
                pi = None
                if self.lip_centred_y:
                    pi = Y.mean(dim=0)         # class priors of the seen rows
                    Y = Y - pi
                draw = torch.randperm(X.shape[0], device=self._device)[:m]
                fit = self._fit_memory([(X.double(), Y, 1.0)],
                                       X[draw], Y[draw].float(), "joint")
                if self.memory_save:
                    allsn = fit["snaps"] + [dict(step=self.lip_fit_steps,
                                                 J=fit["final_J"], B=fit["B"],
                                                 Y=fit["Yat"])]
                    torch.save(dict(Bg=self.head.Bg.detach().cpu(),
                                    seen_count=int(self._seen_count),
                                    centred=bool(self.lip_centred_y),
                                    pi=None if pi is None else pi.cpu(),
                                    snaps=[dict(step=s["step"], J=s["J"],
                                                B=s["B"].cpu(), Y=s["Y"].cpu())
                                           for s in allsn]),
                               self._stem + "_memory.pt")
                    logging.info("memory payload ({} snaps) -> {}".format(
                        len(allsn), self._stem + "_memory.pt"))
                for i, s in enumerate(fit["snaps"]):
                    self._train_snap_head(s["B"].to(self._device),
                                          s["Y"].to(self._device), 71 + i)
                self._seen_feats, self._seen_labels = self._mem_X, self._mem_Y
                self._train_head()
            return

        # capture ONLY the new task's rows (the cache; parent concatenates)
        self._seen_feats = self._seen_labels = None
        self._cache_task_features()
        new_X, new_y = self._seen_feats, self._seen_labels
        N_prev, n_new = self._seen_count, new_X.shape[0]
        self._seen_count = N_prev + n_new

        if self._cur_task == 0:
            self._train_head()                       # full task-0 data (B.5)
            self._update_memory(new_X, new_y, N_prev)  # then compress (Q1a)
        else:
            self._update_memory(new_X, new_y, N_prev)  # compress FIRST (B.4)
            self._seen_feats, self._seen_labels = self._mem_X, self._mem_Y
            self._train_head()                       # memory alone
            if last and self.final_fresh_retrain:
                self._fresh_retrain()
        self._seen_feats = self._seen_labels = None  # the cache is gone

    def _cache_test_features(self):
        """The TEST half of the parent's _cache_task_features, alone:
        head_from_memory runs skip the train cache but the per-epoch eval
        (_eval_cached) still needs the accumulated clean test features."""
        if self.head_eval_every <= 0:
            return
        self._network.to(self._device)
        self._network.eval()
        new_test = self.data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="test", mode="test")
        loader = DataLoader(new_test, batch_size=self.batch_size,
                            shuffle=False, num_workers=num_workers)
        feats, labels = [], []
        for _, data, clip_data, label in loader:
            feats.append(self.head.preprocess(
                self._network, data.to(self._device),
                clip_data.to(self._device)))
            labels.append(label.to(self._device))
        feats, labels = torch.cat(feats), torch.cat(labels)
        self._test_feats = (feats if self._test_feats is None
                            else torch.cat([self._test_feats, feats]))
        self._test_labels = (labels if self._test_labels is None
                             else torch.cat([self._test_labels, labels]))

    def _train_snap_head(self, X, Y, marker):
        """Train a FRESH head (new W1 draw, zeroed output, SAME Bg -- the
        memory lives in this run's gate space) on one mid-fit memory
        snapshot, logged under `marker` in the head CSV; then restore the
        original never-trained head for the final memory's own training."""
        logging.info("snapshot head training (task marker {})".format(marker))
        orig, ct = self.head, self._cur_task
        in_features = self.feature_dim + self._network.clip.out_dim
        head = HEADS[self.args.get("head_model", "vila-mimic")](
            in_features, self.args, self._device)
        head.Bg.copy_(orig.Bg)
        head.append_task(self._total_classes)
        self.head = self._network.ac_model = head
        self._seen_feats, self._seen_labels, self._cur_task = X, Y, marker
        try:
            self._train_head()
        finally:
            self.head, self._network.ac_model, self._cur_task = orig, orig, ct

    def _head_from_memory(self):
        """Stage B (wave 7): load a saved memory payload, adopt its Bg (the
        atoms were distilled against it), pick one snapshot by step
        (head_from_snap, 0 = the final one), and train the head on it with
        the step-matched budget x head_budget_mult.  No caching, no fit."""
        pay = torch.load(self.head_from_memory, map_location="cpu")
        snaps = {int(s["step"]): s for s in pay["snaps"]}
        step = self.head_from_snap or max(snaps)
        s = snaps[step]
        with torch.no_grad():
            self.head.Bg.copy_(pay["Bg"].to(self._device))
        self._seen_count = int(pay["seen_count"])
        self._seen_feats = s["B"].to(self._device)
        self._seen_labels = s["Y"].to(self._device)
        logging.info("head_from_memory: {} snap step {} ({} rows, centred={},"
                     " budget x{})".format(self.head_from_memory, step,
                                           self._seen_feats.shape[0],
                                           pay.get("centred"),
                                           self.head_budget_mult))
        self._train_head()

    def _fresh_retrain(self):
        """Control (wave 6): rebuild the head -- fresh W1 draw, zeroed output
        -- but COPY Bg (the memory was distilled in the old head's gate
        space), then train it on the final memory with the same step-matched
        budget.  Logged as task 99 in the head curves CSV."""
        logging.info("final fresh-head retrain on the memory (task 99)")
        old_Bg = self.head.Bg
        in_features = self.feature_dim + self._network.clip.out_dim
        self.head = HEADS[self.args.get("head_model", "vila-mimic")](
            in_features, self.args, self._device)
        self.head.Bg.copy_(old_Bg)
        self.head.append_task(self._total_classes)
        self._network.ac_model = self.head
        ct, self._cur_task = self._cur_task, 99
        try:
            self._train_head()
        finally:
            self._cur_task = ct

    # ---------------------------------------------------------------- memory
    def _update_memory(self, new_X, new_y, N_prev):
        """memory <- compress(w_old * memory + w_new * new rows), m points."""
        m, C = self.memory_m, self._total_classes
        w_old = N_prev / (N_prev + new_X.shape[0])
        if self.memory_mode == "coreset":
            n_old = int(round(w_old * m))
            keep = torch.randperm(self._mem_X.shape[0],
                                  device=self._device)[:n_old] \
                if n_old else torch.empty(0, dtype=torch.long, device=self._device)
            draw = torch.randperm(new_X.shape[0], device=self._device)[:m - n_old]
            self._mem_X = (torch.cat([self._mem_X[keep], new_X[draw]])
                           if n_old else new_X[draw].clone())
            self._mem_Y = (torch.cat([self._mem_Y[keep], new_y[draw]])
                           if n_old else new_y[draw].clone())
            logging.info("task {} coreset memory: kept {} old + drew {} new"
                         .format(self._cur_task, int(n_old), m - n_old))
            return
        # ---- lip ----
        # verbatim until full (wave 3): while N_seen <= m the union of seen
        # rows IS an exact m'-point memory (int labels, no fit -- lossless by
        # construction).  The first real fit happens once N_seen > m.
        if N_prev + new_X.shape[0] <= m:
            self._mem_X = (new_X.clone() if self._mem_X is None
                           else torch.cat([self._mem_X, new_X]))
            self._mem_Y = (new_y.clone() if self._mem_Y is None
                           else torch.cat([self._mem_Y, new_y]))
            logging.info("task {} lip memory: verbatim ({} rows <= m={}, "
                         "no fit)".format(self._cur_task,
                                          self._mem_X.shape[0], m))
            return
        Y_new = F.one_hot(new_y, C).double()         # plain one-hots (c.9)
        parts = [(new_X.double(), Y_new, 1.0 - w_old)]
        if self._mem_X is not None:
            Y_old = (F.one_hot(self._mem_Y, C).float()   # verbatim -> one-hot
                     if not self._mem_Y.dtype.is_floating_point
                     else F.pad(self._mem_Y, (0, C - self._mem_Y.shape[1])))
            parts.insert(0, (self._mem_X.double(), Y_old.double(), w_old))
            B0, Y0 = self._mem_X, Y_old              # warm start (Q2)
            if B0.shape[0] < m:      # verbatim memory smaller than m: top up
                d = torch.randperm(new_X.shape[0],   # from new rows (mirrors
                                   device=self._device)[:m - B0.shape[0]]
                B0 = torch.cat([B0, new_X[d]])       # the first-fit init rule)
                Y0 = torch.cat([Y0, Y_new[d].float()])
        else:                                        # first fit: uniform draw
            draw = torch.randperm(new_X.shape[0], device=self._device)[:m]
            B0, Y0 = new_X[draw], Y_new[draw].float()
        if self.lip_fit_init == "rand":              # from-scratch every fit:
            allZ = torch.cat([p[0] for p in parts]).float()   # moment-matched
            B0 = (allZ.mean(0) + allZ.std(0)                  # Gaussian atoms
                  * torch.randn(m, allZ.shape[1], device=self._device))
            Y0 = 0.01 * torch.randn(m, C, device=self._device)
        self._fit_memory(parts, B0, Y0, "w_old={:.4f}".format(w_old))

    def _fit_memory(self, parts, B0, Y0, note):
        TAR = make_target(parts, self.head.Bg)
        logging.info("task {} lip fit: target {} rows ({}), m={}, {} steps"
                     .format(self._cur_task, TAR["Z"].shape[0], note,
                             self.memory_m, self.lip_fit_steps))
        fit = fit_lip(TAR, B0, Y0, self.head.Bg, steps=self.lip_fit_steps,
                      fit_lr=self.lip_fit_lr, lam=self.lip_lambda_mv,
                      eval_every=self.lip_fit_eval_every,
                      adam_eps=self.lip_fit_adam_eps,
                      mb=self.lip_fit_mb, jitter=self.lip_fit_jitter,
                      snapshot_best=self.lip_fit_snapshot_best,
                      sched=self.lip_fit_sched,
                      snap_steps=self.lip_snap_steps,
                      wm_chunk=2048 if TAR["Z"].shape[0] > 20000 else 8192,
                      verbose=lambda s: logging.info(s.strip()))
        if fit["status"] != "ok":
            raise RuntimeError("lip fit diverged at task {} step {}".format(
                self._cur_task, fit["diverged_at"]))
        self._mem_X, self._mem_Y = fit["B"], fit["Yat"]
        for i in range(len(fit["curve"]["step"])):
            print(self._cur_task, *[fit["curve"][k][i] for k in LIP_CURVE],
                  file=self._lip_log, sep=",")
        return fit

    # ---------------------------------------------------------------- train
    def _train_head(self):
        """Parent loop with two changes: (1) step-matched epoch budget --
        head_epochs * ceil(N_seen/bs) total steps, re-expressed as epochs over
        the actual training set; (2) targets may be soft fp32 rows (lip)."""
        n = self._seen_feats.shape[0]
        bs = self.head_batch_size
        steps_target = (self.head_budget_mult * self.head_epochs
                        * ((self._seen_count + bs - 1) // bs))
        steps_per_epoch = (n + bs - 1) // bs
        epochs = (steps_target + steps_per_epoch - 1) // steps_per_epoch
        soft = self._seen_labels.dtype.is_floating_point
        params = [p for p in self.head.parameters() if p.requires_grad]
        if self.head_opt == "adamw":
            opt = optim.AdamW(params, lr=self.head_lr, weight_decay=self.head_wd)
        else:
            opt = optim.SGD(params, lr=self.head_lr, momentum=self.head_momentum,
                            weight_decay=self.head_wd)
        sched = (optim.lr_scheduler.CosineAnnealingLR(
                     opt, T_max=epochs, eta_min=0.0)
                 if self.head_schedule == "cosine" else None)
        text_features = None
        if self.head_eval_every > 0:
            text_features = self._feat_from_temp()
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        in_noise = self.args.get("head_input_noise", 0.0)
        feat_std = self._seen_feats.std(dim=0) if in_noise > 0 else None
        logging.info("task {} head training: n={} epochs={} (step-matched to "
                     "N_seen={})".format(self._cur_task, n, epochs,
                                         self._seen_count))
        self.head.train()
        for epoch in range(1, epochs + 1):
            lr_now = opt.param_groups[0]["lr"]
            perm = torch.randperm(n, device=self._device)
            loss_sum = correct = 0
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                X, y = self._seen_feats[idx], self._seen_labels[idx]
                if in_noise > 0:
                    X = X + in_noise * feat_std * torch.randn_like(X)
                opt.zero_grad(set_to_none=True)
                logits = self.head(X)["logits"]
                tgt = (y.to(logits.dtype) if soft
                       else F.one_hot(y, self._total_classes).to(logits.dtype))
                loss = F.mse_loss(logits, tgt)
                loss.backward()
                opt.step()
                loss_sum += loss.item() * idx.numel()
                y_idx = y.argmax(dim=1) if soft else y
                correct += (logits.argmax(dim=1) == y_idx).sum().item()
            if sched is not None:
                sched.step()
            train_loss, train_acc = loss_sum / n, correct / n
            test_acc = ""
            if self.head_eval_every > 0 and (epoch % self.head_eval_every == 0
                                             or epoch == epochs):
                self.head.eval()
                test_acc = self._eval_cached(text_features)
                self.head.train()
            print(self._cur_task, epoch, lr_now, train_loss, train_acc, test_acc,
                  file=self._head_log, sep=",")
            if epoch % max(1, epochs // 5) == 0 or epoch == epochs:
                logging.info(
                    "task {} head epoch {}/{} lr {:.5f} train_mse {:.6f} "
                    "train_acc {:.4f} test_acc {}".format(
                        self._cur_task, epoch, epochs, lr_now,
                        train_loss, train_acc, test_acc))
        self.head.eval()
