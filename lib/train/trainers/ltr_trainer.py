import os
import datetime
import json
from contextlib import nullcontext
from collections import Counter, OrderedDict

from lib.train.data.wandb_logger import WandbWriter
from lib.train.trainers import BaseTrainer
from lib.train.admin import AverageMeter, StatValue
from lib.train.admin import TensorboardWriter
import torch
import time
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

from lib.utils.misc import get_world_size


class LTRTrainer(BaseTrainer):
    def __init__(self, actor, loaders, optimizer, settings, lr_scheduler=None, use_amp=False):
        """
        args:
            actor - The actor for training the network
            loaders - list of dataset loaders, e.g. [train_loader, val_loader]. In each epoch, the trainer runs one
                        epoch for each loader.
            optimizer - The optimizer used for training, e.g. Adam
            settings - Training settings
            lr_scheduler - Learning rate scheduler
        """
        super().__init__(actor, loaders, optimizer, settings, lr_scheduler)

        self._set_default_settings()

        # Initialize statistics variables
        self.stats = OrderedDict({loader.name: None for loader in self.loaders})

        # Initialize tensorboard and wandb
        self.wandb_writer = None
        if settings.local_rank in [-1, 0]:
            tensorboard_writer_dir = os.path.join(self.settings.env.tensorboard_dir, self.settings.project_path)
            if not os.path.exists(tensorboard_writer_dir):
                os.makedirs(tensorboard_writer_dir)
            self.tensorboard_writer = TensorboardWriter(tensorboard_writer_dir, [l.name for l in loaders])

            if settings.use_wandb:
                world_size = get_world_size()
                cur_train_samples = self.loaders[0].dataset.samples_per_epoch * max(0, self.epoch - 1)
                interval = (world_size * settings.batchsize)  # * interval
                self.wandb_writer = WandbWriter(settings.project_path[6:], {}, tensorboard_writer_dir, cur_train_samples, interval)

        self.move_data_to_gpu = getattr(settings, 'move_data_to_gpu', True)
        self.settings = settings
        self.use_amp = use_amp
        if use_amp:
            self.scaler = GradScaler()

    def _set_default_settings(self):
        # Dict of all default values
        default = {'print_interval': 10,
                   'print_stats': None,
                   'description': ''}

        for param, default_value in default.items():
            if getattr(self.settings, param, None) is None:
                setattr(self.settings, param, default_value)

    def cycle_dataset(self, loader):
        """Do a cycle of training or validation."""

        self.actor.train(loader.training)
        torch.set_grad_enabled(loader.training)

        # if loader.training:
        # # #     # 1. 遍历网络的所有参数
        # #     # 1. 暴力冻结：先剥夺【所有】参数的梯度
        #     for param in self.actor.net.parameters():
        #         param.requires_grad = False

        # # # 2. 白名单精准解冻：放行生成器、注入器、以及预测头
        #     print("====== 正在配置白名单参数 ======")
        #     for name, param in self.actor.net.named_parameters():
        #         if 'mlf_trm' in name or 'prompt_blocks' in name or 'box_head' in name:
        #             param.requires_grad = True
        #             #print(name)

        self._init_timing()
        multiview_counts = Counter()
        target_view_counts = Counter()
        multiview_groups = 0
        validation_manifest_rows = []

        for i, data in enumerate(loader, 1):
            self.data_read_done_time = time.time()
            # get inputs
            if self.move_data_to_gpu:
                data = data.to(self.device)

            self.data_to_gpu_time = time.time()

            data['epoch'] = self.epoch
            data['settings'] = self.settings
            actor_cfg = getattr(self.actor, "cfg", None)
            diagnostics_enabled = bool(getattr(
                getattr(getattr(actor_cfg, "TRAIN", None), "MULTIVIEW", None),
                "DIAGNOSTICS_ENABLED", False))
            if diagnostics_enabled:
                images = data.get("template_images", None)
                if not torch.is_tensor(images) or images.dim() != 5:
                    raise RuntimeError(
                        "Multiview diagnostics expected [V,B,C,H,W] images")
                num_views, group_batch = int(images.shape[0]), int(images.shape[1])
                view_ids = data.get("view_ids", None)
                target_ids = data.get("target_id", None)
                if view_ids is None or target_ids is None:
                    raise RuntimeError("Multiview target/view metadata is missing")
                if len(view_ids) != num_views or len(target_ids) != group_batch:
                    raise RuntimeError("Multiview metadata shape mismatch")
                for view_index in range(num_views):
                    labels = view_ids[view_index]
                    for batch_index in range(group_batch):
                        view_label = str(labels[batch_index])
                        target_id = str(target_ids[batch_index])
                        multiview_counts[view_label] += 1
                        target_view_counts[(target_id, view_label)] += 1
                        save_val_manifest = bool(getattr(
                            getattr(actor_cfg.TRAIN, "MULTIVIEW", None),
                            "SAVE_VAL_MANIFEST", False))
                        if save_val_manifest and not loader.training:
                            template_ids = data.get("template_frame_ids", [])
                            search_ids = data.get("search_frame_ids", [])
                            template_frame = int(template_ids[0][batch_index])
                            search_frame = int(search_ids[0][batch_index])
                            validation_manifest_rows.append({
                                "target_id": target_id,
                                "view_id": view_label,
                                "template_frame": template_frame,
                                "search_frame": search_frame,
                            })
                multiview_groups += group_batch
            paired_training = bool(
                loader.training
                and getattr(self.actor, "paired_supervision_enabled", False)
            )
            if paired_training:
                stats = self._paired_training_step(data, i)
                loss = None
            # forward pass
            elif not self.use_amp:
                loss, stats = self.actor(data)
            else:
                with autocast():
                    loss, stats = self.actor(data)

            # backward pass and update weights
            def print_grad_norms():
                    print("\n[Gradient Norms]:")
                    for name, param in self.actor.net.named_parameters():
                        if param.grad is not None:
                            # 计算 L2 范数
                            grad_norm = param.grad.norm().item()
                            # 过滤掉梯度非常小的层，避免刷屏 (可选)
                            if grad_norm > 0: 
                                print(f"{name}: {grad_norm:.6f}")
                        else:
                            print(f"{name}: None")
                    print("-" * 20)
            if loader.training and not paired_training:
                self.optimizer.zero_grad()
                if not self.use_amp:
                    # for name, param in self.actor.net.named_parameters():
                    #     if not param.requires_grad:
                    #         continue
                    #     def make_hook(n):
                    #         def hook(grad):
                    #             print(f"{n}: grad_norm={grad.norm().item():.6e}, grad_mean={grad.mean().item():.6e}")
                    #         return hook
                    #     param.register_hook(make_hook(name))
                    loss.backward()
                    # if i % 80 == 0:  # 每 100 个 iter 打印一次，防止刷屏
                    #     print_grad_norms()
                    # backbone_params = []
                    # atp_params = []

                    # for name, param in self.actor.net.named_parameters():
                    #     if 'atp' in name:
                    #         atp_params.append(param)
                    #     else:
                    #         backbone_params.append(param)

                    # torch.nn.utils.clip_grad_norm_(backbone_params, 0.1)
                    # torch.nn.utils.clip_grad_norm_(atp_params, 0.1) 
                    if self.settings.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.actor.net.parameters(), self.settings.grad_clip_norm)
                    self.optimizer.step()
                else:
                    self.scaler.scale(loss).backward()
                    if self.settings.grad_clip_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.actor.net.parameters(), self.settings.grad_clip_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

            # update statistics
            batch_size = data['template_images'].shape[loader.stack_dim]
            self._update_stats(stats, batch_size, loader)

            # print statistics
            self._print_stats(i, loader, batch_size)

            # update wandb status
            if self.wandb_writer is not None and i % self.settings.print_interval == 0:
                if self.settings.local_rank in [-1, 0]:
                    self.wandb_writer.write_log(self.stats, self.epoch)

            train_cfg = getattr(actor_cfg, "TRAIN", None)
            max_iterations = int(getattr(train_cfg, "MAX_ITERS_PER_EPOCH", 0))
            if max_iterations > 0 and i >= max_iterations:
                break

        if multiview_counts:
            payload = {
                "loader": loader.name,
                "epoch": int(self.epoch),
                "group_samples": int(multiview_groups),
                "local_samples": int(sum(multiview_counts.values())),
                "view_counts": dict(sorted(multiview_counts.items())),
                "target_id_x_view": {
                    "%s x %s" % key: int(value)
                    for key, value in sorted(target_view_counts.items())
                },
            }
            line = "[MultiviewEpoch] " + json.dumps(
                payload, sort_keys=True, ensure_ascii=True)
            print(line)
            if self.settings.local_rank in [-1, 0]:
                with open(self.settings.log_file, "a") as log_file:
                    log_file.write(line + "\n")
                if validation_manifest_rows:
                    manifest_path = os.path.join(
                        os.path.dirname(self.settings.log_file),
                        "validation_sampling_epoch_%04d.jsonl" % self.epoch,
                    )
                    with open(manifest_path, "w") as manifest_file:
                        for row in validation_manifest_rows:
                            manifest_file.write(json.dumps(
                                row, sort_keys=True, ensure_ascii=True) + "\n")
                    manifest_line = "[ValidationManifest] path=%s rows=%d" % (
                        manifest_path, len(validation_manifest_rows))
                    print(manifest_line)
                    with open(self.settings.log_file, "a") as log_file:
                        log_file.write(manifest_line + "\n")

        # calculate ETA after every epoch
        epoch_time = self.prev_time - self.start_time
        print("Epoch Time: " + str(datetime.timedelta(seconds=epoch_time)))
        print("Avg Data Time: %.5f" % (self.avg_date_time / self.num_frames * batch_size))
        print("Avg GPU Trans Time: %.5f" % (self.avg_gpu_trans_time / self.num_frames * batch_size))
        print("Avg Forward Time: %.5f" % (self.avg_forward_time / self.num_frames * batch_size))

    def _paired_training_step(self, data, iteration):
        diagnostics_interval = int(self.actor._get_cfg_value(
            "TRAIN.PCUM.DIAGNOSTICS_INTERVAL", 50))
        diagnostics_active = bool(
            self.actor._get_cfg_value("TRAIN.PCUM.DIAGNOSTICS_ENABLED", False)
            and diagnostics_interval > 0
            and iteration % diagnostics_interval == 0
        )
        self.actor.begin_paired_iteration(data, diagnostics_active=diagnostics_active)
        if diagnostics_active and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
        self.optimizer.zero_grad()

        no_sync_context = (
            self.actor.net.no_sync()
            if hasattr(self.actor.net, "no_sync")
            else nullcontext()
        )
        with no_sync_context:
            if self.use_amp:
                with autocast():
                    local_loss, cache = self.actor.paired_local_stage(data)
                self.scaler.scale(local_loss).backward()
            else:
                local_loss, cache = self.actor.paired_local_stage(data)
                local_loss.backward()
        del local_loss

        if self.use_amp:
            with autocast():
                collaborative_loss, stats = self.actor.paired_collaborative_stage(
                    data, cache)
            self.scaler.scale(collaborative_loss).backward()
            self.scaler.unscale_(self.optimizer)
        else:
            collaborative_loss, stats = self.actor.paired_collaborative_stage(
                data, cache)
            collaborative_loss.backward()
        del collaborative_loss, cache

        stats.update(self.actor.collect_gradient_diagnostics())
        if diagnostics_active and torch.cuda.is_available():
            stats["Memory/max_allocated_mb"] = float(
                torch.cuda.max_memory_allocated(self.device) / (1024.0 ** 2)
            )

        if self.settings.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.actor.net.parameters(), self.settings.grad_clip_norm)

        if self.use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        return stats

    def train_epoch(self):
        """Do one epoch for each loader."""
        for loader in self.loaders:
            if self.epoch % loader.epoch_interval == 0:
                # 2021.1.10 Set epoch
                if isinstance(loader.sampler, DistributedSampler):
                    loader.sampler.set_epoch(self.epoch)
                self.cycle_dataset(loader)

        self._stats_new_epoch()
        if self.settings.local_rank in [-1, 0]:
            self._write_tensorboard()

    def _init_timing(self):
        self.num_frames = 0
        self.start_time = time.time()
        self.prev_time = self.start_time
        self.avg_date_time = 0
        self.avg_gpu_trans_time = 0
        self.avg_forward_time = 0

    def _update_stats(self, new_stats: OrderedDict, batch_size, loader):
        # Initialize stats if not initialized yet
        if loader.name not in self.stats.keys() or self.stats[loader.name] is None:
            self.stats[loader.name] = OrderedDict({name: AverageMeter() for name in new_stats.keys()})

        # add lr state
        if loader.training:
            lr_list = self.lr_scheduler.get_last_lr()
            for i, lr in enumerate(lr_list):
                var_name = 'LearningRate/group{}'.format(i)
                if var_name not in self.stats[loader.name].keys():
                    self.stats[loader.name][var_name] = StatValue()
                self.stats[loader.name][var_name].update(lr)

        for name, val in new_stats.items():
            if name not in self.stats[loader.name].keys():
                self.stats[loader.name][name] = AverageMeter()
            self.stats[loader.name][name].update(val, batch_size)

    def _print_stats(self, i, loader, batch_size):
        self.num_frames += batch_size
        current_time = time.time()
        batch_fps = batch_size / (current_time - self.prev_time)
        average_fps = self.num_frames / (current_time - self.start_time)
        prev_frame_time_backup = self.prev_time
        self.prev_time = current_time

        self.avg_date_time += (self.data_read_done_time - prev_frame_time_backup)
        self.avg_gpu_trans_time += (self.data_to_gpu_time - self.data_read_done_time)
        self.avg_forward_time += current_time - self.data_to_gpu_time

        if i % self.settings.print_interval == 0 or i == loader.__len__():
            print_str = '[%s: %d, %d / %d] ' % (loader.name, self.epoch, i, loader.__len__())
            print_str += 'FPS: %.1f (%.1f)  ,  ' % (average_fps, batch_fps)

            # 2021.12.14 add data time print
            print_str += 'DataTime: %.3f (%.3f)  ,  ' % (self.avg_date_time / self.num_frames * batch_size, self.avg_gpu_trans_time / self.num_frames * batch_size)
            print_str += 'ForwardTime: %.3f  ,  ' % (self.avg_forward_time / self.num_frames * batch_size)
            print_str += 'TotalTime: %.3f  ,  ' % ((current_time - self.start_time) / self.num_frames * batch_size)
            # print_str += 'DataTime: %.3f (%.3f)  ,  ' % (self.data_read_done_time - prev_frame_time_backup, self.data_to_gpu_time - self.data_read_done_time)
            # print_str += 'ForwardTime: %.3f  ,  ' % (current_time - self.data_to_gpu_time)
            # print_str += 'TotalTime: %.3f  ,  ' % (current_time - prev_frame_time_backup)

            for name, val in self.stats[loader.name].items():
                if (self.settings.print_stats is None or name in self.settings.print_stats):
                    if hasattr(val, 'avg'):
                        print_str += '%s: %.5f  ,  ' % (name, val.avg)
                    elif hasattr(val, 'val'):
                        print_str += '%s: %.8g  ,  ' % (name, val.val)

            print(print_str[:-5])
            log_str = print_str[:-5] + '\n'
            with open(self.settings.log_file, 'a') as f:
                f.write(log_str)

    def _stats_new_epoch(self):
        # Record learning rate
        for loader in self.loaders:
            if loader.training:
                try:
                    lr_list = self.lr_scheduler.get_last_lr()
                except:
                    lr_list = self.lr_scheduler._get_lr(self.epoch)
                for i, lr in enumerate(lr_list):
                    var_name = 'LearningRate/group{}'.format(i)
                    if var_name not in self.stats[loader.name].keys():
                        self.stats[loader.name][var_name] = StatValue()
                    self.stats[loader.name][var_name].update(lr)

        for loader_stats in self.stats.values():
            if loader_stats is None:
                continue
            for stat_value in loader_stats.values():
                if hasattr(stat_value, 'new_epoch'):
                    stat_value.new_epoch()

    def _write_tensorboard(self):
        if self.epoch == 1:
            self.tensorboard_writer.write_info(self.settings.script_name, self.settings.description)

        self.tensorboard_writer.write_epoch(self.stats, self.epoch)


# The FCVC specialization shares the canonical trainer module namespace while
# retaining a separate file to keep the legacy LTRTrainer readable.
from .fcvc_ltr_trainer import FCVCLTRTrainer  # noqa: E402,F401
