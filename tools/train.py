import _init_path
import argparse
import datetime
import glob
import os
from pathlib import Path
from test import repeat_eval_ckpt
from eval_utils import eval_utils

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator
from pcdet.utils import common_utils
from train_utils.optimization import build_optimizer, build_scheduler, get_finetune_groups_summary
from train_utils.wandb_utils import set_quick_dashboard_keys, log_run_basics, record_quick_summary
from train_utils.train_utils import train_model


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')
    parser.add_argument('--local_rank', type=int, default=None, help='local rank for distributed training')
    parser.add_argument('--max_ckpt_save_num', type=int, default=30, help='max number of saved checkpoint')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--num_epochs_to_eval', type=int, default=0, help='number of checkpoints to be evaluated')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    
    parser.add_argument('--use_tqdm_to_record', action='store_true', default=False, help='if True, the intermediate losses will not be logged to file, only tqdm will be used')
    parser.add_argument('--logger_iter_interval', type=int, default=50, help='')
    parser.add_argument('--ckpt_save_time_interval', type=int, default=300, help='in terms of seconds')
    parser.add_argument('--wo_gpu_stat', action='store_true', help='')
    parser.add_argument('--use_amp', action='store_true', help='use mix precision training')
    

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'
    
    args.use_amp = args.use_amp or cfg.OPTIMIZATION.get('USE_AMP', False)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    # Allow overriding CLI workers from YAML: prefer OPTIMIZATION.WORKERS, fallback to TRAIN.WORKERS
    try:
        yaml_workers = cfg.OPTIMIZATION.get('WORKERS', None)
        if yaml_workers is None:
            yaml_workers = cfg.get('TRAIN', {}).get('WORKERS', None)
        if yaml_workers is not None and args.workers == parser.get_default('workers'):
            args.workers = int(yaml_workers)
    except Exception:
        pass

    return args, cfg


def main():
    args, cfg = parse_config()
    if args.launcher == 'none':
        dist_train = False
        total_gpus = 1
    else:
        if args.local_rank is None:
            args.local_rank = int(os.environ.get('LOCAL_RANK', '0'))
            
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_train = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs

    if args.fix_random_seed:
        common_utils.set_random_seed(666 + cfg.LOCAL_RANK)

    # If user didn't set extra_tag, inherit WANDB autogenerated name (single-process only)
    wandb_run = None
    if cfg.get('WANDB', {}).get('USE', False) and args.launcher == 'none' and args.extra_tag == 'default':
        if wandb is not None:
            try:
                tmp_run = wandb.init(project=cfg.WANDB.PROJECT)
                if getattr(tmp_run, 'name', None):
                    args.extra_tag = tmp_run.name
                wandb_run = tmp_run
            except Exception:
                pass

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    ckpt_dir = output_dir / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('train_%s.log' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_train:
        logger.info('Training in distributed mode : total_batch_size: %d' % (total_gpus * args.batch_size))
    else:
        logger.info('Training with a single process')
        
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)
    if cfg.LOCAL_RANK == 0:
        os.system('cp %s %s' % (args.cfg_file, output_dir))

    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None

    logger.info("----------- Create dataloader & network & optimizer -----------")
    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        total_epochs=args.epochs,
        seed=666 if args.fix_random_seed else None
    )

    # Optional: build validation dataloader if evaluating during training
    eval_every_n = int(cfg.OPTIMIZATION.get('EVAL_EVERY_N_EPOCHS', 0) or 0)
    test_set = test_loader = None
    eval_output_dir = None
    if eval_every_n > 0:
        test_set, test_loader, _ = build_dataloader(
            dataset_cfg=cfg.DATA_CONFIG,
            class_names=cfg.CLASS_NAMES,
            batch_size=args.batch_size,
            dist=dist_train, workers=args.workers, logger=logger, training=False
        )
        eval_output_dir = (cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG /
                           args.extra_tag / 'eval' / 'during_train')
        eval_output_dir.mkdir(parents=True, exist_ok=True)

    if cfg.get('WANDB', {}).get('USE', False) and cfg.LOCAL_RANK == 0:
        if wandb is None:
            logger.warning('wandb is not installed, skipping wandb logging')
        else:
            def edict_to_dict(e):
                from easydict import EasyDict
                if isinstance(e, EasyDict) or isinstance(e, dict):
                    return {k: edict_to_dict(v) for k, v in e.items()}
                if isinstance(e, list):
                    return [edict_to_dict(v) for v in e]
                return e

            cfg_dict = edict_to_dict(cfg)
            if wandb_run is None:
                # Initialize now; if user set extra_tag, name run accordingly
                if args.extra_tag != 'default':
                    wandb_run = wandb.init(project=cfg.WANDB.PROJECT, name=f"{cfg.TAG}/{args.extra_tag}", config=cfg_dict)
                else:
                    wandb_run = wandb.init(project=cfg.WANDB.PROJECT, config=cfg_dict)
            else:
                # Reuse early-initialized run and attach full config
                try:
                    wandb_run.config.update(cfg_dict, allow_val_change=True)
                except Exception:
                    pass

            # Quick dashboard keys and basic metadata
            set_quick_dashboard_keys(wandb_run)
            log_run_basics(wandb_run, cfg, train_set, per_gpu_batch_size=args.batch_size, total_gpus=total_gpus)

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    optimizer = build_optimizer(model, cfg.OPTIMIZATION)

    # load checkpoint if it is possible
    start_epoch = it = 0
    last_epoch = -1

    # Allow config-driven checkpoints
    if args.pretrained_model is None:
        args.pretrained_model = cfg.get('PRETRAINED_MODEL', None)
    if args.ckpt is None:
        args.ckpt = cfg.get('CKPT', cfg.get('RESUME_FROM', None))

    # Verbose: show resolved checkpoint sources
    if args.pretrained_model is not None:
        logger.info(f"Resolved PRETRAINED_MODEL from config: {args.pretrained_model}")
    if args.ckpt is not None:
        logger.info(f"Resolved CKPT/RESUME_FROM from config: {args.ckpt}")

    if args.pretrained_model is not None:
        model.load_params_from_file(filename=args.pretrained_model, to_cpu=dist_train, logger=logger)

    if args.ckpt is not None:
        it, start_epoch = model.load_params_with_optimizer(args.ckpt, to_cpu=dist_train, optimizer=optimizer, logger=logger)
        last_epoch = start_epoch + 1
    else:
        auto_resume = cfg.get('TRAIN', {}).get('AUTO_RESUME', True)
        if auto_resume:
            ckpt_list = glob.glob(str(ckpt_dir / '*.pth'))
            if len(ckpt_list) > 0:
                ckpt_list.sort(key=os.path.getmtime)
                while len(ckpt_list) > 0:
                    try:
                        it, start_epoch = model.load_params_with_optimizer(
                            ckpt_list[-1], to_cpu=dist_train, optimizer=optimizer, logger=logger
                        )
                        last_epoch = start_epoch + 1
                        # If the checkpoint already completed all target epochs, skip resuming
                        if start_epoch >= args.epochs:
                            logger.info(f"Found checkpoint at epoch {start_epoch} >= target epochs {args.epochs}. Starting a fresh run without resuming.")
                            # Reset optimizer to clear loaded state
                            optimizer = build_optimizer(model, cfg.OPTIMIZATION)
                            start_epoch = 0
                            it = 0
                            last_epoch = -1
                        break
                    except:
                        ckpt_list = ckpt_list[:-1]
        else:
            logger.info('AUTO_RESUME disabled by config; not scanning for existing checkpoints.')

    model.train()  # before wrap to DistributedDataParallel to support fixed some parameters
    if dist_train:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()])
    logger.info(f'----------- Model {cfg.MODEL.NAME} created, param count: {sum([m.numel() for m in model.parameters()])} -----------')
    logger.info(model)

    # Log fine-tune group summary to wandb (lightweight, config-level)
    try:
        if wandb_run is not None:
            ft_summary = get_finetune_groups_summary(optimizer, cfg.OPTIMIZATION)
            wandb_run.config.update(ft_summary, allow_val_change=True)
    except Exception:
        pass

    # Verbose console logging of finetune settings
    try:
        ft_summary = get_finetune_groups_summary(optimizer, cfg.OPTIMIZATION)
        freeze_modules = ft_summary['finetune'].get('freeze_modules', [])
        bn_eval = ft_summary['finetune'].get('bn_eval_during_train', False)
        bn_affine = ft_summary['finetune'].get('bn_freeze_affine', False)
        logger.info(f"[Finetune] Freeze modules: {freeze_modules if freeze_modules else '[]'}")
        logger.info(f"[Finetune] BN eval during train: {bn_eval} (freeze affine: {bn_affine})")

        # Print logical group settings
        groups = ft_summary['finetune'].get('groups', [])
        if groups:
            logger.info("[Finetune] Logical groups (max_lr, init_lr, wd, counts):")
            for g in groups:
                logger.info(
                    f"  - {g['name']}: max_lr={g['max_lr']:.3e}, init_lr={g['init_lr']:.3e}, "
                    f"wd={g.get('weight_decay')}, params={g['params_count']}, "
                    f"decay={g['decay_count']}, nodecay={g['nodecay_count']}"
                )

        # Print actual optimizer param_groups order
        logger.info("[Finetune] Optimizer param_groups order (lr_init, wd, size):")
        for i, pg in enumerate(getattr(optimizer, 'param_groups', [])):
            gname = pg.get('group_name', f'group_{i}')
            n_params = sum(p.numel() for p in pg.get('params', []))
            lr_init = pg.get('lr', 0.0)
            wd = pg.get('weight_decay', 0.0)
            logger.info(f"  [{i}] {gname}: lr_init={lr_init:.3e}, wd={wd}, params={n_params}")
    except Exception:
        pass

    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer, total_iters_each_epoch=len(train_loader), total_epochs=args.epochs,
        last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION
    )

    # -----------------------start training---------------------------
    logger.info('**********************Start training %s/%s(%s)**********************'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    train_model(
        model,
        optimizer,
        train_loader,
        model_func=model_fn_decorator(),
        lr_scheduler=lr_scheduler,
        optim_cfg=cfg.OPTIMIZATION,
        start_epoch=start_epoch,
        total_epochs=args.epochs,
        start_iter=it,
        rank=cfg.LOCAL_RANK,
        tb_log=tb_log,
        ckpt_save_dir=ckpt_dir,
        train_sampler=train_sampler,
        lr_warmup_scheduler=lr_warmup_scheduler,
        ckpt_save_interval=args.ckpt_save_interval,
        max_ckpt_save_num=args.max_ckpt_save_num,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch, 
        logger=logger, 
        logger_iter_interval=args.logger_iter_interval,
        ckpt_save_time_interval=args.ckpt_save_time_interval,
        use_logger_to_record=not args.use_tqdm_to_record, 
        show_gpu_stat=not args.wo_gpu_stat,
        use_amp=args.use_amp,
        cfg=cfg,
        wandb_run=wandb_run,
        # Eval-during-train options
        eval_every_n_epochs=eval_every_n,
        eval_loader=test_loader,
        eval_output_dir=eval_output_dir,
        dist_test=dist_train,
        args=args
    )

    if hasattr(train_set, 'use_shared_memory') and train_set.use_shared_memory:
        train_set.clean_shared_memory()

    logger.info('**********************End training %s/%s(%s)**********************\n\n\n'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    # If eval-during-train is enabled, skip the post-training repeat-eval to avoid duplicate evals
    if eval_every_n > 0:
        logger.info('Skip post-training repeat_eval_ckpt since EVAL_EVERY_N_EPOCHS is enabled.')
    else:
        logger.info('**********************Start evaluation %s/%s(%s)**********************' %
                    (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
        test_set, test_loader, sampler = build_dataloader(
            dataset_cfg=cfg.DATA_CONFIG,
            class_names=cfg.CLASS_NAMES,
            batch_size=args.batch_size,
            dist=dist_train, workers=args.workers, logger=logger, training=False
        )
        if wandb_run is not None:
            wandb_run.config.update({'dataset/val_samples': len(test_set)})
        eval_output_dir = output_dir / 'eval' / 'eval_with_train'
        eval_output_dir.mkdir(parents=True, exist_ok=True)
        # Only evaluate the last args.num_epochs_to_eval epochs; if 0 (unspecified), evaluate the latest checkpoint once
        if args.num_epochs_to_eval is None or args.num_epochs_to_eval <= 0:
            args.start_epoch = max(args.epochs - 1, 0)
        else:
            args.start_epoch = max(args.epochs - args.num_epochs_to_eval, 0)

        repeat_eval_ckpt(
            model.module if dist_train else model,
            test_loader, args, eval_output_dir, logger, ckpt_dir,
            dist_test=dist_train,
            wandb_run=wandb_run,
            best_key=cfg.get('WANDB', {}).get('BEST_KEY'),
            save_artifacts=cfg.get('WANDB', {}).get('SAVE_ARTIFACTS', True),
            wandb_phase='valid'
        )
        logger.info('**********************End evaluation %s/%s(%s)**********************' %
                    (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    if wandb_run is not None:
        # Record concise summary for cross-run comparison
        record_quick_summary(wandb_run, cfg, epochs=args.epochs, extra_tag=args.extra_tag)
        wandb_run.finish()


if __name__ == '__main__':
    main()
