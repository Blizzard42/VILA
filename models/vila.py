import copy
import logging
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split

from utils.inc_net import SimpleCosineIncrementalNet
from utils.VILA import VILA
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy


import time

num_workers = 8


class Learner(BaseLearner):   
    def __init__(self, args):
        super().__init__(args)
        if 'resnet' in args['backbone_type']:
            self._network = SimpleCosineIncrementalNet(args, True)
            self.batch_size = 128
            self.init_lr = args["init_lr"] if args["init_lr"] is not None else 0.01
        else:
            self._network = VILA(args, True)
            self.batch_size = args["batch_size"]
            self.init_lr = args["init_lr"]
        
        self._network._device = self._device
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args['min_lr'] if args['min_lr'] is not None else 1e-8
        self.args = args
        self.R = None
        # e_48 experiment_2 feature-cache PRODUCER (opt-in, "feature_cache_save"
        # = path, "{seed}" substituted).  VILA's own protocol already makes
        # exactly one train-mode (augmented) pass over every task's rows in
        # _cls_align/_IL_align; this captures those 1280-d features (768 adapter
        # + 512 CLIP, both L2-normalised) by GLOBAL train-row index, the clean
        # test features of ALL classes, the CLIP text features of ALL classes
        # (class-order space), and the trained adapter+mlp0 state, so later
        # runs skip the backbone entirely.  Adapter weights depend only on
        # (dataset, seed, init_cls, backbone_type, ffn_num, adpt_epoch,
        # init_lr, weight_decay, min_lr, optimizer, batch_size, num_workers).
        # When enabled, eval_task scores from the cached test features
        # (_fc_eval: same 0.8 adpt + 0.2 clip-rerank math as _eval_cnn, no
        # per-task re-forward of the test set).
        self.fc_save = args.get("feature_cache_save", "") or ""
        if self.fc_save:
            self.fc_save = self.fc_save.format(seed=args["seed"])
        self._fc = None

        logging.info('Parameter information at initialization stage.')
        total_params = sum(p.numel() for p in self._network.backbone.parameters())
        total_trainable_params = sum(p.numel() for p in self._network.backbone.parameters() if p.requires_grad)
        logging.info(f'Adapter branch info: {total_params:,} total parameters, {total_trainable_params:,} training parameters.')
        
        total_params = sum(p.numel() for p in self._network.clip.parameters())
        total_trainable_params = sum(p.numel() for p in self._network.clip.parameters() if p.requires_grad)
        logging.info(f'CLIP branch info: {total_params:,} total parameters, {total_trainable_params:,} training parameters.')
        
    def after_task(self):
        self._known_classes = self._total_classes
    
    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train", mode="train")
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        if self.fc_save:
            # get_dataset order: class-major, each class in global row order
            y = data_manager._train_targets
            self._fc_task_gidx = np.concatenate(
                [np.where(y == c)[0]
                 for c in range(self._known_classes, self._total_classes)])
        if self._cur_task == 0:
            self._init_train(self.train_loader, self.test_loader)
            self._network.update_fc(self.args['Hidden'], self._total_classes)
            if self.fc_save:
                # RNG state restored so the augmentation draw _cls_align sees
                # is bit-identical to a plain (non-caching) VILA run
                rng = torch.get_rng_state()
                self._fc_init(data_manager)
                torch.set_rng_state(rng)
            self._cls_align(self.train_loader, self._network)
        else:
            self._network.update_fc(self.args['Hidden'], self._total_classes)
            for param in self._network.ac_model.parameters():
                param.requires_grad = False
            self._IL_align(self.train_loader, self._network)
        
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
        if self.fc_save and self._cur_task == data_manager.nb_tasks - 1:
            self._fc_finish()

    def _init_train(self, train_loader, test_loader):
        total_params = sum(p.numel() for p in self._network.backbone.parameters())
        total_trainable_params = sum(p.numel() for p in self._network.backbone.parameters() if p.requires_grad)
        logging.info(f'Parameter information at training ViT stage: {total_params:,} total parameters, {total_trainable_params:,} training parameters.')
        
        assert self._known_classes == 0
        self._network.to(self._device)

        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(self._network.backbone.parameters(), momentum=0.9, lr=self.init_lr, weight_decay=self.weight_decay)
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.AdamW(self._network.backbone.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args['adpt_epoch'], eta_min=self.min_lr)
        
        prog_bar = tqdm(range(self.args['adpt_epoch']))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, _, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
            
                features = self._network.backbone(inputs)
                logits = self._network.backbone.mlp0(features)

                loss = F.cross_entropy(logits, targets.long())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(self._cur_task,epoch + 1,self.args['adpt_epoch'],losses / len(train_loader),train_acc,)
            prog_bar.set_description(info)

        logging.info(info)

    def _cls_align(self, loader, model):
        if hasattr(model, 'module'):
            model = model.module
        else:
            model = model

        embedding_list = []
        label_list = []
        feature_list = []

        # AL training process
        model.to(self._device)
        model = model.eval()

        auto_cor = torch.zeros(model.ac_model.fc[-1].weight.size(1), model.ac_model.fc[-1].weight.size(1)).to(self._device)
        crs_cor = torch.zeros(model.ac_model.fc[-1].weight.size(1), self._total_classes).to(self._device)

        with torch.no_grad():
            pbar = tqdm(enumerate(loader), desc='Alignment', total=len(loader), unit='batch')
            for i, batch in pbar:
                (bidx, data, clip_data, label) = batch
                images, clip_images, target = data.to(self._device), clip_data.to(self._device), label.to(self._device)

                feature = model(images, clip_images)["features"]
                if self.fc_save:
                    self._fc_store(bidx, feature)
                new_activation = model.ac_model.fc[:2](feature)

                label_onehot = F.one_hot(target, self._total_classes).float()
                auto_cor += torch.t(new_activation) @ new_activation
                crs_cor += torch.t(new_activation) @ (label_onehot)

                embedding_list.append(new_activation.cpu())
                label_list.append(target.cpu())
                feature_list.append(feature.cpu())

        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)
        feature_list = torch.cat(feature_list, dim=0)
        Y = target2onehot(label_list, self._total_classes)

        ridge = self.optimise_ridge_parameter(embedding_list, Y)
        logging.info("gamma {}".format(ridge))

        print('numpy inverse')
        R = np.mat(auto_cor.cpu().numpy() + ridge * np.eye(model.ac_model.fc[-1].weight.size(1))).I
        R = torch.tensor(R).float().to(self._device)
        Delta = R @ crs_cor
        model.ac_model.fc[-1].weight = torch.nn.parameter.Parameter(torch.t(0.9 * Delta.float()))
        self.R = R
        del R

    def optimise_ridge_parameter(self, Features, Y):
        ridges = 10.0 ** np.arange(-8, 9)
        num_val_samples = int(Features.shape[0] * 0.8)
        losses = []
        Q_val = Features[0:num_val_samples, :].T @ Y[0:num_val_samples, :]
        G_val = Features[0:num_val_samples, :].T @ Features[0:num_val_samples, :]
        for ridge in ridges:
            Wo = torch.linalg.solve(G_val + ridge*torch.eye(G_val.size(dim=0)), Q_val).T #better nmerical stability than .inv
            Y_train_pred = Features[num_val_samples::,:] @ Wo.T
            losses.append(F.mse_loss(Y_train_pred, Y[num_val_samples::, :]))
        ridge = ridges[np.argmin(np.array(losses))]
        print('selected lambda =',ridge)
        return ridge

    def _IL_align(self, loader, model):
        if hasattr(model, 'module'):
            model = model.module
        else:
            model = model

        label_list = []
        feature_list = []

        # AL training process
        model.to(self._device)
        model = model.eval()

        W = (model.ac_model.fc[-1].weight.t()).float()
        R = copy.deepcopy(self.R.float())

        with torch.no_grad():
            pbar = tqdm(enumerate(loader), desc='Alignment', total=len(loader), unit='batch')
            for i, batch in pbar:
                (bidx, data, clip_data, label) = batch
                images, clip_images, target = data.to(self._device), clip_data.to(self._device), label.to(self._device)

                feature = model(images, clip_images)["features"]
                if self.fc_save:
                    self._fc_store(bidx, feature)
                new_activation = model.ac_model.fc[:2](feature)

                R = R - R @ new_activation.t() @ torch.pinverse(
                    torch.eye(new_activation.size(0)).to(self._device) +
                    new_activation @ R @ new_activation.t()) @ new_activation @ R

                label_onehot = F.one_hot(target, self._total_classes).float()
                W = W + R @ new_activation.t() @ (label_onehot - new_activation @ W)

                label_list.append(target.cpu())
                feature_list.append(feature.cpu())
        label_list = torch.cat(label_list, dim=0)
        feature_list = torch.cat(feature_list, dim=0)

        print('numpy inverse')
        model.ac_model.fc[-1].weight = torch.nn.parameter.Parameter(torch.t(W.float()))
        self.R = R
        del R

    def _feat_from_temp(self):
        text_features = []
        total_labels=self.data_manager._class_to_label[:self._total_classes] # mask all known classes
        with torch.no_grad():
            for l in total_labels:
                texts = [t.format(l) for t in self.data_manager._data_to_prompt]
                texts = self._network.tokenizer(texts).to(self._device)
                class_embeddings = self._network.clip_encode_text(texts)
                class_embeddings = class_embeddings.mean(dim=0)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                text_features.append(class_embeddings)
            text_features = torch.stack(text_features, dim=0)
        return text_features

    def _eval_cnn(self, loader):
        self._network.to(self._device)
        self._network.eval()

        text_features = self._feat_from_temp()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        y_pred, y_true = [], []
        for _, (_, inputs, clip_inputs, targets) in enumerate(loader):
            inputs, clip_inputs = inputs.to(self._device), clip_inputs.to(self._device)
            with torch.no_grad():
                adpt_logits = self._network(inputs, clip_inputs)['logits']
                clip_logits = self._network.clip_encode_image(clip_inputs) @ text_features.T
                # Aggregate logits
                rerank_logits = self.clip_rerank(adpt_logits, clip_logits, topk=self.args['rerank_topk'])
                logits = adpt_logits * 0.8 + rerank_logits * 0.2

            predicts = torch.topk(
                logits, k=self.topk, dim=1, largest=True, sorted=True
            )[1]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
            
        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]
        
    # ---- experiment_2 feature-cache producer -------------------------------
    def eval_task(self):
        if not self.fc_save:
            return super().eval_task()
        y_pred, y_true = self._fc_eval()
        return self._evaluate(y_pred, y_true), None

    @torch.no_grad()
    def _fc_init(self, data_manager):
        """After task-0 adapter training: adapter+mlp0 state, clean test
        features of ALL classes, CLIP text features of ALL classes; write the
        small task-0 file immediately."""
        import os
        self._network.to(self._device)
        self._network.eval()
        n_train = len(data_manager._train_targets)
        dim = self.feature_dim + self._network.clip.out_dim
        test_set = data_manager.get_dataset(
            np.arange(0, data_manager.nb_classes), source="test", mode="test")
        loader = DataLoader(test_set, batch_size=self.batch_size,
                            shuffle=False, num_workers=num_workers)
        feats, labels = [], []
        for _, data, clip_data, label in loader:
            feats.append(self._network(data.to(self._device),
                                       clip_data.to(self._device))["features"].cpu())
            labels.append(label)
        text = []
        for l in data_manager._class_to_label:      # class-order space
            t = self._network.tokenizer(
                [tp.format(l) for tp in data_manager._data_to_prompt]).to(self._device)
            e = self._network.clip_encode_text(t).mean(dim=0)
            text.append(e / e.norm())
        self._fc = dict(
            recipe={k: self.args.get(k) for k in (
                "dataset", "seed", "init_cls", "increment", "backbone_type",
                "ffn_num", "adpt_epoch", "init_lr", "weight_decay", "min_lr",
                "optimizer", "batch_size", "shuffle", "Hidden", "rerank_topk")},
            num_workers=num_workers, feature_dim=dim,
            clip_dim=self._network.clip.out_dim,
            class_order=list(data_manager._class_order),
            class_to_label=list(data_manager._class_to_label),
            adapter_state={n: p.detach().cpu().clone() for n, p in
                           self._network.backbone.named_parameters() if p.requires_grad},
            test_F=torch.cat(feats).float(), test_y=torch.cat(labels).long(),
            text_F=torch.stack(text).float().cpu(),
            train_paths=list(data_manager._train_data),
            train_y=torch.as_tensor(data_manager._train_targets).long(),
            train_F=torch.zeros(n_train, dim), train_filled=torch.zeros(n_train, dtype=torch.bool))
        os.makedirs(os.path.dirname(self.fc_save) or ".", exist_ok=True)
        small = {k: v for k, v in self._fc.items()
                 if k not in ("train_F", "train_filled", "train_paths", "train_y")}
        torch.save(small, self.fc_save.replace(".pt", "_task0.pt"))
        self._fc_test_dev = self._fc["test_F"].to(self._device)
        logging.info("feature cache: task-0 file written ({} test rows, {} text rows, "
                     "{} adapter tensors); train_F [{} x {}] fp32 allocated".format(
                         self._fc["test_F"].shape[0], self._fc["text_F"].shape[0],
                         len(self._fc["adapter_state"]), n_train, dim))

    def _fc_store(self, bidx, feature):
        g = torch.as_tensor(self._fc_task_gidx[bidx.numpy()])
        self._fc["train_F"][g] = feature.detach().float().cpu()
        self._fc["train_filled"][g] = True

    def _fc_finish(self):
        n_ok = int(self._fc["train_filled"].sum())
        if self.args.get("limit_classes"):        # smoke: partial by design
            logging.warning("feature cache: PARTIAL ({} / {} rows, limit_classes)"
                            .format(n_ok, self._fc["train_F"].shape[0]))
        else:
            assert n_ok == self._fc["train_F"].shape[0], (n_ok, self._fc["train_F"].shape)
            del self._fc["train_filled"]
        torch.save(self._fc, self.fc_save)
        logging.info("feature cache: {} train rows -> {}".format(n_ok, self.fc_save))

    @torch.no_grad()
    def _fc_eval(self):
        """_eval_cnn's metric from the cached clean test features restricted
        to the classes seen so far (bit-for-bit the same math)."""
        text_features = self._feat_from_temp()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        clip_dim = self._network.clip.out_dim
        y = self._fc["test_y"]
        mask = y < self._total_classes
        Fm, ym = self._fc_test_dev[mask.to(self._device)], y[mask]
        y_pred = []
        for i in range(0, Fm.shape[0], 512):
            f = Fm[i:i + 512]
            adpt_logits = self._network.ac_model(f)["logits"]
            clip_logits = f[:, -clip_dim:] @ text_features.T
            rerank_logits = self.clip_rerank(adpt_logits, clip_logits,
                                             topk=self.args["rerank_topk"])
            logits = adpt_logits * 0.8 + rerank_logits * 0.2
            y_pred.append(torch.topk(logits, k=self.topk, dim=1, largest=True,
                                     sorted=True)[1].cpu().numpy())
        return np.concatenate(y_pred), ym.numpy()

    def clip_rerank(self, outputs, clip_logits, topk=5):
        if topk > outputs.shape[1]:
            topk = outputs.shape[1]
        with torch.no_grad():
            top5_predict_indices = outputs.topk(topk, 1, True, True)[1]
            # top5_predict_labels = [[self.data_manager._class_to_label[int(label)] for label in pred] for pred in top5_predict_indices]
            new_logits = torch.zeros_like(outputs)
            for i in range(outputs.shape[0]):
                new_logits[i, top5_predict_indices[i]] = clip_logits[i, top5_predict_indices[i]]
            return new_logits
        