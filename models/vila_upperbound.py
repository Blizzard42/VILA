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
the cached features, sigma in per-dim std units of the seen cache), head_l2
(gamma of the analytic ridge: adds (gamma/(N_seen*C_seen))*sum||W||^2 over
ALL trainable head params to the mean-MSE loss, so the minimizer matches the
analytic solve's sum-SE + gamma*||W||^2 objective; CSV train_loss stays pure
MSE).

Wave 19 keys: head_solve ("sgd" default | "analytic": vila-mimic only,
replaces _train_head with VILA's exact math on the cached features — task-0
ridge (HtH + gamma I)^-1 HtY with gamma = head_l2 if > 0 else CV'd by the
vila.py 80/20 grid, later tasks RLS on the new-task cache carrying R; float64
throughout, one CSV row per task with the full-seen-cache train MSE/acc);
head_task0_scale (multiply the last-layer weight by this once after the
task-0 fit, both solve modes — VILA's hard-coded 0.9); head_proj_seed
(vila-mimic only: re-draw fc0 from its own CPU generator with this seed,
same U(-1/sqrt(d), 1/sqrt(d)) law as nn.Linear's default init; unset =
bit-identical to previous waves).

Wave 40 keys (multi-draw augmentation statistics):
  cache_draws (int, 1): cache k independent train-mode augmentation draws
    per task instead of one (k passes of the train loader; the CLIP branch
    is deterministic so its features repeat).  k > 1 caches are stored
    fp16 (fp32 compute per batch) and the SGD head budget is STEP-MATCHED
    to the 1-draw convention (head_epochs * ceil(n_real/bs) total steps,
    n_real = rows/k) — draws add data richness, not compute.
  cache_save (path): after the LAST task, write the whole per-task draw
    cache ({task: {"F": [k, n_t, D] fp16 cpu, "y": [k, n_t] int64}}) so
    one producer serves every arm.
  cache_load (path): skip the backbone forward for TRAIN caching and take
    the FIRST cache_draws draws per task from the file (test features are
    still cached from this run's own backbone — same seed, same adapter).
  head_solve additions: "none" (producer: no head fit at all);
    "analytic_gram" (vila-mimic only: accumulate G += HtH, Q += HtY in
    fp64 over every draw and task, solve W = (G + gamma I)^-1 Q per task —
    the RLS-without-forgetting equivalent that stays feasible at k=100;
    gamma = head_l2 if > 0 else the vila.py CV grid on the first 2500
    task-0 rows, one first-draw-sized subsample).
"""
import logging
import time
from contextlib import contextmanager

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from models.vila import Learner as VilaLearner, num_workers
from models.lip_fit import GATE_ACTS


class PhaseTimer:
    """Coarse wall-clock phase accumulator, reported as one '[time]' log
    line per task.  Phases NEST (head_eval runs inside head_train;
    snap_head calls head_train; chunked lip fits accumulate one lip_fit
    count per chunk): the summary prints RAW per-phase totals, so nested
    phases overlap and their sum can exceed the wall clock.  GPU work is
    not synchronized -- phase boundaries sit at the pipeline's natural
    .item()/logging sync points -- so these are coarse benchmarks, not
    profiles.  Untimed remainder = trainer-side eval + data loading."""

    def __init__(self):
        self.acc, self.n, self.t0 = {}, {}, time.time()

    @contextmanager
    def __call__(self, name):
        t = time.time()
        try:
            yield
        finally:
            self.acc[name] = self.acc.get(name, 0.0) + time.time() - t
            self.n[name] = self.n.get(name, 0) + 1

    def wrap(self, obj, name, meth):
        """Rebind obj.meth (MRO-resolved, so overrides are caught) to a
        version timed under `name`."""
        fn = getattr(obj, meth)

        def timed(*a, **k):
            with self(name):
                return fn(*a, **k)
        setattr(obj, meth, timed)

    def summary(self):
        return "[time] wall {:.1f}s; ".format(time.time() - self.t0) + " ".join(
            "{}={:.1f}s/{}".format(k, v, self.n[k])
            for k, v in sorted(self.acc.items(), key=lambda kv: -kv[1]))


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
        proj_seed = args.get("head_proj_seed", 0)
        if proj_seed:
            # nn.Linear default init is kaiming_uniform_(a=sqrt(5)) =
            # U(-1/sqrt(fan_in), 1/sqrt(fan_in)) for a bias-free layer;
            # redraw from an own-seed CPU generator so the draw is decoupled
            # from the global seed stream
            g = torch.Generator().manual_seed(int(proj_seed))
            bound = 1.0 / in_features ** 0.5
            w = torch.empty(args["Hidden"], in_features)
            w.uniform_(-bound, bound, generator=g)
            with torch.no_grad():
                self.fc0.weight.copy_(w)
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
    h = act(Bg x) (o) (W1 x) with Bg FROZEN ~ N(0, 1/d) (a random projection,
    never trained -- the net stays linear in its trainable weights for fixed
    gates) and W1 trainable ~ N(0, 1/d), width = args["Hidden"]; optional
    dropout (head_dropout).  Wave 10: gate_act picks the activation (default
    "step" = the classic 1[.>0], bit-identical); gate_scale is a run-level
    constant divisor (see set_gate_scale), 1.0 unless gate_norm is on."""

    def __init__(self, in_features, args, device):
        super().__init__()
        kh = args["Hidden"]
        gs = args.get("gate_seed", None)
        if gs in (None, ""):
            Bg0 = torch.randn(kh, in_features, device=device)
        else:
            # wave 17: reseed ONLY the frozen gate projection (the kernel).
            # CPU generator; also burn the device randn so every other
            # random event matches the unseeded run.
            torch.randn(kh, in_features, device=device)
            g = torch.Generator().manual_seed(int(gs))
            Bg0 = torch.randn(kh, in_features, generator=g).to(device)
        self.register_buffer("Bg", Bg0 / in_features ** 0.5)
        self.W1 = nn.Parameter(
            torch.randn(kh, in_features, device=device) / in_features ** 0.5)
        drop = args.get("head_dropout", 0.0)
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()
        self.head = nn.Linear(kh, 0, bias=False, device=device)
        self.gate_act = args.get("gate_act", "step")
        assert self.gate_act in GATE_ACTS, self.gate_act
        self.gate_alpha = float(args.get("gate_alpha", 1.0))
        self.register_buffer("gate_scale",
                             torch.ones((), dtype=torch.float32, device=device))

    @torch.no_grad()
    def set_gate_scale(self, feats):
        """gate_norm: divide gates by the rms of act(alpha Bg z) over the
        given rows (the task-0 cache) -- ONE run-level constant, shared with
        the LIP fit so memory and head live in the same feature space."""
        g = GATE_ACTS[self.gate_act](self.gate_alpha * (feats @ self.Bg.T))
        self.gate_scale.copy_(g.pow(2).mean().sqrt())

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

    def forward(self, features, g_override=None):
        # g_override (wave 36): untied per-row gates used verbatim (already
        # post-scale, exactly what gates() / the lip fit stores) -- memory
        # rows carry their gates as data; test rows stay tied (no override)
        g = (g_override if g_override is not None
             else GATE_ACTS[self.gate_act](
                 self.gate_alpha * (features @ self.Bg.T)) / self.gate_scale)
        h = g * (features @ self.W1.T)
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
        self.head_l2 = args.get("head_l2", 0.0)
        self.head_batch_size = args.get("head_batch_size", 4096)
        self.head_eval_every = args.get("head_eval_every", 0)
        self.head_solve = args.get("head_solve", "sgd")
        assert self.head_solve in ("sgd", "analytic", "analytic_gram", "none")
        self.head_task0_scale = args.get("head_task0_scale", 1.0)
        # wave 40: multi-draw augmentation caches (module docstring)
        self.cache_draws = int(args.get("cache_draws", 1) or 1)
        self.cache_save = args.get("cache_save", "")
        self.cache_load = args.get("cache_load", "")
        self._cache_file = None          # lazy torch.load of cache_load
        self._cache_dump = {}            # producer: per-task cpu tensors
        # CSVs land next to trainer.py's log file, same naming scheme
        init_cls = 0 if args["init_cls"] == args["increment"] else args["init_cls"]
        stem = "logs/{}/{}/{}/{}/{}_{}_{}".format(
            args["model_name"], args["dataset"], init_cls, args["increment"],
            args["prefix"], args["seed"], args["backbone_type"])
        self._head_log = open(stem + "_head_curves.csv", "w", buffering=1)
        print("task", "epoch", "lr", "train_loss", "train_acc", "test_acc",
              file=self._head_log, sep=",")
        # wave 31.1: coarse phase timing ('[time]' log lines, greppable)
        self._pt = PhaseTimer()
        for name, meth in (("init_backbone", "_init_train"),
                           ("cache_feats", "_cache_task_features"),
                           ("head_train", "_train_head"),
                           ("head_analytic", "_solve_head_analytic"),
                           ("head_analytic_gram", "_solve_head_analytic_gram"),
                           ("head_eval", "_eval_cached")):
            self._pt.wrap(self, name, meth)

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
        n_prev = 0 if self._seen_feats is None else self._seen_feats.shape[0]
        self._cache_task_features()
        if self.head_solve == "none":
            pass                          # wave 40 producer: cache only
        elif self.head_solve == "analytic":
            self._solve_head_analytic(n_prev)
        elif self.head_solve == "analytic_gram":
            self._solve_head_analytic_gram(n_prev)
        else:
            self._train_head()
            if self._cur_task == 0 and self.head_task0_scale != 1.0:
                with torch.no_grad():
                    self.head.head.weight.mul_(self.head_task0_scale)
        if (self.cache_save
                and self._cur_task == data_manager.nb_tasks - 1):
            torch.save(dict(draws=self.cache_draws, tasks=self._cache_dump),
                       self.cache_save)
            logging.info("draw cache ({} tasks x {} draws) -> {}".format(
                len(self._cache_dump), self.cache_draws, self.cache_save))
        logging.info(self._pt.summary())

    @torch.no_grad()
    def _cache_task_features(self):
        """cache_draws train-mode (augmented) draws per new-task sample ->
        seen cache (fp16 when draws > 1, or when loading a k>1-draw file);
        clean test features of the new classes -> test cache (for the cheap
        per-epoch eval), always from THIS run's backbone."""
        self._network.to(self._device)
        self._network.eval()
        k = self.cache_draws
        if self.cache_load:
            if self._cache_file is None:
                self._cache_file = torch.load(self.cache_load,
                                              map_location="cpu")
                stored = int(self._cache_file["draws"])
                assert stored >= k, (stored, k)
                logging.info("cache_load: {} ({} draws stored, using first "
                             "{})".format(self.cache_load, stored, k))
            ent = self._cache_file["tasks"][self._cur_task]
            feats = (ent["F"][:k].reshape(-1, ent["F"].shape[-1])
                     .to(self._device))
            labels = ent["y"][:k].reshape(-1).to(self._device)
            if k == 1:
                feats = feats.float()     # 1-draw runs stay fp32 end to end
        else:
            drawsF, drawsY = [], []
            for d in range(k):
                fs, ls = [], []
                for _, data, clip_data, label in self.train_loader:
                    fs.append(self.head.preprocess(
                        self._network, data.to(self._device),
                        clip_data.to(self._device)))
                    ls.append(label.to(self._device))
                fs, ls = torch.cat(fs), torch.cat(ls)
                drawsF.append(fs.half() if k > 1 else fs)
                drawsY.append(ls)
                if k > 1 and (d + 1) % 10 == 0:
                    logging.info("task {} cache draw {}/{}".format(
                        self._cur_task, d + 1, k))
            feats, labels = torch.cat(drawsF), torch.cat(drawsY)
            if self.cache_save:
                self._cache_dump[self._cur_task] = dict(
                    F=torch.stack([f.half().cpu() for f in drawsF]),
                    y=torch.stack([l.cpu() for l in drawsY]))
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
        # wave 10: gate_norm sets the run-level gate scale ONCE, from the
        # task-0 cache (this method runs for every learner that caches)
        if (self._cur_task == 0 and self.args.get("gate_norm", False)
                and hasattr(self.head, "set_gate_scale")):
            self.head.set_gate_scale(self._seen_feats)
            logging.info("gate_norm: gate_scale = {:.6f}".format(
                float(self.head.gate_scale)))

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
        feature cache, fresh optimizer per task (constant or cosine lr).
        cache_draws > 1: epochs re-expressed so TOTAL STEPS match the 1-draw
        budget (draws are data richness, not extra compute)."""
        n = self._seen_feats.shape[0]
        epochs = self.head_epochs
        if self.cache_draws > 1:
            bs = self.head_batch_size
            n_real = n // self.cache_draws
            steps_target = self.head_epochs * ((n_real + bs - 1) // bs)
            spe = (n + bs - 1) // bs
            epochs = max(1, (steps_target + spe - 1) // spe)
            logging.info("draw step-match: n={} ({} draws), epochs {} -> {} "
                         "({} steps vs {} at 1 draw)".format(
                             n, self.cache_draws, self.head_epochs, epochs,
                             epochs * spe, steps_target))
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
        # train-time gaussian input noise, sigma in per-dim std units of the
        # (growing) seen cache; resampled every minibatch
        in_noise = self.args.get("head_input_noise", 0.0)
        feat_std = self._feat_std() if in_noise > 0 else None
        # exact ridge match: analytic objective is sum-SE + gamma*||W||^2;
        # ours is mean-MSE, so the equivalent coefficient is gamma/(N*C)
        l2_coef = self.head_l2 / (n * self._total_classes)
        self.head.train()  # dropout active only while fitting
        for epoch in range(1, epochs + 1):
            lr_now = opt.param_groups[0]["lr"]
            perm = torch.randperm(n, device=self._device)
            loss_sum = correct = 0
            for i in range(0, n, self.head_batch_size):
                idx = perm[i:i + self.head_batch_size]
                X, y = self._seen_feats[idx].float(), self._seen_labels[idx]
                if in_noise > 0:
                    X = X + in_noise * feat_std * torch.randn_like(X)
                opt.zero_grad(set_to_none=True)
                logits = self.head(X)["logits"]
                mse = F.mse_loss(logits, F.one_hot(
                    y, self._total_classes).to(logits.dtype))
                loss = mse
                if l2_coef > 0:
                    loss = mse + l2_coef * sum(p.pow(2).sum() for p in params)
                loss.backward()
                opt.step()
                loss_sum += mse.item() * y.numel()
                correct += (logits.argmax(dim=1) == y).sum().item()
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

    @torch.no_grad()
    def _feat_std(self):
        """Per-dim std of the seen cache, fp64 accumulators, row-chunked (a
        5M-row fp16 cache neither materializes in fp32 nor sums stably in
        fp16); returns fp32."""
        n, s = self._seen_feats.shape[0], self.head_batch_size * 8
        acc = acc2 = 0.0
        for i in range(0, n, s):
            x = self._seen_feats[i:i + s].double()
            acc = acc + x.sum(dim=0)
            acc2 = acc2 + (x * x).sum(dim=0)
        mu = acc / n
        return (acc2 / n - mu * mu).clamp_min_(0).sqrt().float()

    @torch.no_grad()
    def _hidden(self, feats):
        """vila-mimic's frozen expansion h = relu(fc0 x), float64, batched."""
        out = []
        for i in range(0, feats.shape[0], self.head_batch_size):
            out.append(F.relu(self.head.fc0(
                feats[i:i + self.head_batch_size].float())).double())
        return torch.cat(out)

    @torch.no_grad()
    def _optimise_gamma(self, H, Y):
        """vila.py optimise_ridge_parameter port: fit on the first 80%,
        score MSE on the last 20%, grid 10^-8..10^8."""
        ridges = 10.0 ** np.arange(-8, 9)
        n_fit = int(H.shape[0] * 0.8)
        Q = H[:n_fit].T @ Y[:n_fit]
        G = H[:n_fit].T @ H[:n_fit]
        eye = torch.eye(G.shape[0], dtype=G.dtype, device=G.device)
        losses = []
        for ridge in ridges:
            Wo = torch.linalg.solve(G + ridge * eye, Q)
            losses.append(F.mse_loss(H[n_fit:] @ Wo, Y[n_fit:]).item())
        gamma = float(ridges[int(np.argmin(losses))])
        logging.info("analytic head: CV-selected gamma {}".format(gamma))
        return gamma

    @torch.no_grad()
    def _solve_head_analytic_gram(self, n_prev):
        """Wave 40: RLS-without-forgetting as one batch ridge.  Accumulate
        G += HtH, Q += HtY in fp64 over the NEW rows (all augmentation
        draws), solve W = (G + gamma I)^-1 Q each task.  Same fixed point as
        head_solve=analytic at the same gamma (RLS from a ridge-consistent
        init IS batch ridge) but stays feasible at k=100 draws, where the
        batched Woodbury recursion costs ~50 GPU-hours.  gamma: head_l2 if
        > 0, else the vila.py CV grid on the FIRST 2500 task-0 rows (one
        first-draw-sized subsample, so the selection problem matches k=1)."""
        assert isinstance(self.head, VilaMimicHead), \
            "head_solve=analytic_gram requires head_model=vila-mimic"
        kh = self.head.fc0.weight.shape[0]
        C, bs = self._total_classes, self.head_batch_size
        if self._cur_task == 0:
            self._an_G = torch.zeros(kh, kh, dtype=torch.float64,
                                     device=self._device)
            self._an_Q = torch.zeros(kh, C, dtype=torch.float64,
                                     device=self._device)
            if self.head_l2 > 0:
                self._an_gamma = float(self.head_l2)
            else:
                H0 = self._hidden(self._seen_feats[:2500])
                Y0 = F.one_hot(self._seen_labels[:2500], C).double()
                self._an_gamma = self._optimise_gamma(H0, Y0)
                del H0
        elif self._an_Q.shape[1] < C:
            self._an_Q = F.pad(self._an_Q, (0, C - self._an_Q.shape[1]))
        new_F = self._seen_feats[n_prev:]
        for i in range(0, new_F.shape[0], bs):
            Hb = F.relu(self.head.fc0(new_F[i:i + bs].float())).double()
            Yb = F.one_hot(self._seen_labels[n_prev + i:n_prev + i + bs],
                           C).double()
            self._an_G += Hb.T @ Hb
            self._an_Q += Hb.T @ Yb
        eye = torch.eye(kh, dtype=torch.float64, device=self._device)
        W = torch.linalg.solve(self._an_G + self._an_gamma * eye, self._an_Q)
        if self._cur_task == 0:
            W = self.head_task0_scale * W
        self.head.head.weight.copy_(W.t().float())
        # one CSV row per task: full-seen-cache metrics, batched (a 5M-row
        # cache never materializes its hidden expansion)
        n = self._seen_feats.shape[0]
        loss_sum, correct = 0.0, 0
        for i in range(0, n, bs):
            Hb = F.relu(self.head.fc0(
                self._seen_feats[i:i + bs].float())).double()
            lg = Hb @ W
            yb = self._seen_labels[i:i + bs]
            loss_sum += F.mse_loss(lg, F.one_hot(yb, C).double(),
                                   reduction="sum").item()
            correct += (lg.argmax(dim=1) == yb).sum().item()
        train_loss, train_acc = loss_sum / (n * C), correct / n
        test_acc = ""
        if self.head_eval_every > 0:
            text_features = self._feat_from_temp()
            text_features = text_features / text_features.norm(dim=-1,
                                                               keepdim=True)
            test_acc = self._eval_cached(text_features)
        print(self._cur_task, 0, 0.0, train_loss, train_acc, test_acc,
              file=self._head_log, sep=",")
        logging.info(
            "task {} analytic_gram solve gamma {} train_mse {:.6f} "
            "train_acc {:.4f} test_acc {}".format(
                self._cur_task, self._an_gamma, train_loss, train_acc,
                test_acc))

    @torch.no_grad()
    def _solve_head_analytic(self, n_prev):
        """VILA's exact head math on the cached features (vila-mimic only):
        task 0 = ridge solve (+ head_task0_scale), later tasks = RLS over the
        NEW task's cache carrying R — vila.py _cls_align/_IL_align, but in
        float64 and on the same single-augmentation-draw cache the SGD arms
        train on."""
        assert isinstance(self.head, VilaMimicHead), \
            "head_solve=analytic requires head_model=vila-mimic"
        new_H = self._hidden(self._seen_feats[n_prev:])
        new_Y = F.one_hot(self._seen_labels[n_prev:],
                          self._total_classes).double()
        k = new_H.shape[1]
        if self._cur_task == 0:
            gamma = self.head_l2 if self.head_l2 > 0 else \
                self._optimise_gamma(new_H, new_Y)
            self._an_gamma = gamma
            eye = torch.eye(k, dtype=new_H.dtype, device=new_H.device)
            R = torch.linalg.inv(new_H.T @ new_H + gamma * eye)
            W = self.head_task0_scale * (R @ (new_H.T @ new_Y))
        else:
            # head.append_task zero-padded the new-class rows, mirroring
            # VILA's update_fc; RLS updates W with new data only
            W = self.head.head.weight.detach().double().t()
            R = self._an_R
            for i in range(0, new_H.shape[0], self.head_batch_size):
                Hb = new_H[i:i + self.head_batch_size]
                Yb = new_Y[i:i + self.head_batch_size]
                eye_b = torch.eye(Hb.shape[0], dtype=Hb.dtype, device=Hb.device)
                R = R - R @ Hb.T @ torch.linalg.inv(
                    eye_b + Hb @ R @ Hb.T) @ Hb @ R
                W = W + R @ Hb.T @ (Yb - Hb @ W)
        self._an_R = R
        self.head.head.weight.copy_(W.t().float())
        # one CSV row per task: full-seen-cache metrics, epoch 0, lr 0
        H_all = self._hidden(self._seen_feats)
        logits = H_all @ W
        Y_all = F.one_hot(self._seen_labels, self._total_classes).double()
        train_loss = F.mse_loss(logits, Y_all).item()
        train_acc = (logits.argmax(dim=1)
                     == self._seen_labels).double().mean().item()
        test_acc = ""
        if self.head_eval_every > 0:
            text_features = self._feat_from_temp()
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            test_acc = self._eval_cached(text_features)
        print(self._cur_task, 0, 0.0, train_loss, train_acc, test_acc,
              file=self._head_log, sep=",")
        logging.info(
            "task {} analytic solve gamma {} train_mse {:.6f} train_acc "
            "{:.4f} test_acc {}".format(self._cur_task, self._an_gamma,
                                        train_loss, train_acc, test_acc))
