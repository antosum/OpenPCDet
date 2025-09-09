import os

import torch
import torch.nn as nn
import tqdm
import time
import glob
from torch.nn.utils import clip_grad_norm_
from pcdet.utils import common_utils, commu_utils
from eval_utils import eval_utils


def set_bn_eval(model, freeze_affine: bool = False):
    """Set all BatchNorm modules to eval mode to freeze running stats.
    If freeze_affine is True, also freezes affine parameters.
    Works with plain and DDP-wrapped models.
    """
    module = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
            m.eval()
            if freeze_affine:
                if getattr(m, 'weight', None) is not None:
                    m.weight.requires_grad = False
                if getattr(m, 'bias', None) is not None:
                    m.bias.requires_grad = False


def train_one_epoch(model, optimizer, train_loader, model_func, lr_scheduler, accumulated_iter, optim_cfg,
                    rank, tbar, total_it_each_epoch, dataloader_iter, tb_log=None, leave_pbar=False,
                    use_logger_to_record=False, logger=None, logger_iter_interval=50, cur_epoch=None,
                    total_epochs=None, ckpt_save_dir=None, ckpt_save_time_interval=300, show_gpu_stat=False, use_amp=False,
                    wandb_run=None):
    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)

    ckpt_save_cnt = 1
    start_it = accumulated_iter % total_it_each_epoch

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp, init_scale=optim_cfg.get('LOSS_SCALE_FP16', 2.0**16))
    
    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)
        data_time = common_utils.AverageMeter()
        batch_time = common_utils.AverageMeter()
        forward_time = common_utils.AverageMeter()
        losses_m = common_utils.AverageMeter()
        # Accumulate total points to compute running avg throughput
        points_total = 0

    end = time.time()
    for cur_it in range(start_it, total_it_each_epoch):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)
            print('new iters')
        
        data_timer = time.time()
        cur_data_time = data_timer - end

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        if tb_log is not None:
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)

        model.train()
        # Re-freeze BN stats each iteration if requested
        if optim_cfg.get('BN_EVAL_DURING_TRAIN', False):
            set_bn_eval(model, freeze_affine=optim_cfg.get('BN_FREEZE_AFFINE', False))
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=use_amp):
            loss, tb_dict, disp_dict = model_func(model, batch)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        # Compute total gradient norm (pre-clipping) and apply clipping
        grad_norm = clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
        scaler.step(optimizer)
        scaler.update()

        # Advance LR scheduler after optimizer step to follow PyTorch guidance
        lr_scheduler.step(accumulated_iter, cur_epoch)

        accumulated_iter += 1
 
        cur_forward_time = time.time() - data_timer
        cur_batch_time = time.time() - end
        end = time.time()

        # average reduce
        avg_data_time = commu_utils.average_reduce_value(cur_data_time)
        avg_forward_time = commu_utils.average_reduce_value(cur_forward_time)
        avg_batch_time = commu_utils.average_reduce_value(cur_batch_time)

        # log to console and tensorboard
        if rank == 0:
            batch_size = batch.get('batch_size', None)

            # update meters
            data_time.update(avg_data_time)
            forward_time.update(avg_forward_time)
            batch_time.update(avg_batch_time)
            losses_m.update(loss.item(), batch_size)

            # points in this batch (collated across samples)
            cur_points = 0
            try:
                if isinstance(batch.get('points', None), torch.Tensor):
                    cur_points = int(batch['points'].shape[0])
                elif batch.get('points', None) is not None:
                    cur_points = int(getattr(batch['points'], 'shape', [0])[0])
            except Exception:
                cur_points = 0
            points_total += cur_points

            # instantaneous and running-average throughput (points/sec)
            time_past_this_epoch = pbar.format_dict.get('elapsed', 0.0)
            pts_per_sec_cur = (cur_points / max(avg_batch_time, 1e-6)) if cur_points > 0 else 0.0
            pts_per_sec_avg = (points_total / max(time_past_this_epoch, 1e-6)) if points_total > 0 else 0.0

            disp_dict.update({
                'loss': loss.item(), 'lr': cur_lr, 'd_time': f'{data_time.val:.2f}({data_time.avg:.2f})',
                'f_time': f'{forward_time.val:.2f}({forward_time.avg:.2f})', 'b_time': f'{batch_time.val:.2f}({batch_time.avg:.2f})',
                'pts/s': f'{pts_per_sec_cur:,.0f}({pts_per_sec_avg:,.0f})'
            })
            
            if use_logger_to_record:
                if accumulated_iter % logger_iter_interval == 0 or cur_it == start_it or cur_it + 1 == total_it_each_epoch:
                    trained_time_past_all = tbar.format_dict['elapsed']
                    second_each_iter = pbar.format_dict['elapsed'] / max(cur_it - start_it + 1, 1.0)

                    trained_time_each_epoch = pbar.format_dict['elapsed']
                    remaining_second_each_epoch = second_each_iter * (total_it_each_epoch - cur_it)
                    remaining_second_all = second_each_iter * ((total_epochs - cur_epoch) * total_it_each_epoch - cur_it)
                    
                    logger.info(
                        'Train: {:>4d}/{} ({:>3.0f}%) [{:>4d}/{} ({:>3.0f}%)]  '
                        'Loss: {loss.val:#.4g} ({loss.avg:#.3g})  '
                        'Pts/s: {pts_cur}({pts_avg})  '
                        'LR: {lr:.3e}  '
                        f'Time cost: {tbar.format_interval(trained_time_each_epoch)}/{tbar.format_interval(remaining_second_each_epoch)} ' 
                        f'[{tbar.format_interval(trained_time_past_all)}/{tbar.format_interval(remaining_second_all)}]  '
                        'Acc_iter {acc_iter:<10d}  '
                        'Data time: {data_time.val:.2f}({data_time.avg:.2f})  '
                        'Forward time: {forward_time.val:.2f}({forward_time.avg:.2f})  '
                        'Batch time: {batch_time.val:.2f}({batch_time.avg:.2f})'.format(
                            cur_epoch+1,total_epochs, 100. * (cur_epoch+1) / total_epochs,
                            cur_it,total_it_each_epoch, 100. * cur_it / total_it_each_epoch,
                            loss=losses_m,
                            pts_cur=f'{pts_per_sec_cur:,.0f}',
                            pts_avg=f'{pts_per_sec_avg:,.0f}',
                            lr=cur_lr,
                            acc_iter=accumulated_iter,
                            data_time=data_time,
                            forward_time=forward_time,
                            batch_time=batch_time
                            )
                    )
                    
                    if show_gpu_stat and accumulated_iter % (3 * logger_iter_interval) == 0:
                        # To show the GPU utilization, please install gpustat through "pip install gpustat"
                        gpu_info = os.popen('gpustat').read()
                        logger.info(gpu_info)
            else:                
                pbar.update()
                pbar.set_postfix(dict(total_it=accumulated_iter))
                tbar.set_postfix(disp_dict)
                # tbar.refresh()

            if tb_log is not None:
                tb_log.add_scalar('train/loss', loss, accumulated_iter)
                tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
                for key, val in tb_dict.items():
                    tb_log.add_scalar('train/' + key, val, accumulated_iter)
                # log throughput metrics
                tb_log.add_scalar('meta_data/points_in_batch', cur_points, accumulated_iter)
                tb_log.add_scalar('meta_data/points_per_sec', pts_per_sec_cur, accumulated_iter)
                tb_log.add_scalar('meta_data/points_per_sec_avg', pts_per_sec_avg, accumulated_iter)
                # log gradient norm
                try:
                    tb_log.add_scalar('meta_data/grad_norm', float(getattr(grad_norm, 'item', lambda: grad_norm)()), accumulated_iter)
                except Exception:
                    try:
                        tb_log.add_scalar('meta_data/grad_norm', float(grad_norm), accumulated_iter)
                    except Exception:
                        pass

            # Ensure only the main process performs W&B logging
            if rank == 0 and wandb_run is not None:
                # prepare wandb log dict
                log_dict = {
                    'train/loss': loss.item(),
                    'meta_data/learning_rate': cur_lr,
                    'meta_data/data_time': avg_data_time,
                    'meta_data/forward_time': avg_forward_time,
                    'meta_data/batch_time': avg_batch_time,
                    'meta_data/points_in_batch': cur_points,
                    'meta_data/points_per_sec': pts_per_sec_cur,
                    'meta_data/points_per_sec_avg': pts_per_sec_avg,
                }
                # add gradient norm if available
                try:
                    log_dict['meta_data/grad_norm'] = float(getattr(grad_norm, 'item', lambda: grad_norm)())
                except Exception:
                    try:
                        log_dict['meta_data/grad_norm'] = float(grad_norm)
                    except Exception:
                        pass
                for key, val in tb_dict.items():
                    log_dict[f'train/{key}'] = val
                wandb_run.log(log_dict, step=accumulated_iter)
            
            # save intermediate ckpt every {ckpt_save_time_interval} seconds         
            time_past_this_epoch = pbar.format_dict['elapsed']
            if time_past_this_epoch // ckpt_save_time_interval >= ckpt_save_cnt:
                ckpt_name = ckpt_save_dir / 'latest_model'
                save_checkpoint(
                    checkpoint_state(model, optimizer, cur_epoch, accumulated_iter), filename=ckpt_name,
                )
                logger.info(f'Save latest model to {ckpt_name}')
                ckpt_save_cnt += 1
                
    if rank == 0:
        pbar.close()
    return accumulated_iter


def train_model(model, optimizer, train_loader, model_func, lr_scheduler, optim_cfg,
                start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir, train_sampler=None,
                lr_warmup_scheduler=None, ckpt_save_interval=1, max_ckpt_save_num=50,
                merge_all_iters_to_one_epoch=False, use_amp=False,
                use_logger_to_record=False, logger=None, logger_iter_interval=None, ckpt_save_time_interval=None, show_gpu_stat=False, cfg=None,
                wandb_run=None,
                # Optional evaluation during training
                eval_every_n_epochs: int = 0,
                eval_loader=None,
                eval_output_dir=None,
                dist_test: bool = False,
                args=None):
    accumulated_iter = start_iter

    # use for disable data augmentation hook
    hook_config = cfg.get('HOOK', None) 
    augment_disable_flag = False

    # Track best metric during training-eval (rank 0 only saves)
    best_metric = float('-inf')
    best_epoch = None
    best_key = None
    try:
        best_key = cfg.get('WANDB', {}).get('BEST_KEY', None)
    except Exception:
        best_key = None

    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(train_loader)
        if merge_all_iters_to_one_epoch:
            assert hasattr(train_loader.dataset, 'merge_all_iters_to_one_epoch')
            train_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)
            total_it_each_epoch = len(train_loader) // max(total_epochs, 1)

        dataloader_iter = iter(train_loader)
        for cur_epoch in tbar:
            if train_sampler is not None:
                train_sampler.set_epoch(cur_epoch)

            # train one epoch
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler
            
            augment_disable_flag = disable_augmentation_hook(hook_config, dataloader_iter, total_epochs, cur_epoch, cfg, augment_disable_flag, logger)
            accumulated_iter = train_one_epoch(
                model, optimizer, train_loader, model_func,
                lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                rank=rank, tbar=tbar, tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch,
                dataloader_iter=dataloader_iter, 
                
                cur_epoch=cur_epoch, total_epochs=total_epochs,
                use_logger_to_record=use_logger_to_record, 
                logger=logger, logger_iter_interval=logger_iter_interval,
                ckpt_save_dir=ckpt_save_dir, ckpt_save_time_interval=ckpt_save_time_interval, 
                show_gpu_stat=show_gpu_stat,
                use_amp=use_amp,
                wandb_run=wandb_run
            )

            # save trained model
            trained_epoch = cur_epoch + 1
            if trained_epoch % ckpt_save_interval == 0 and rank == 0:

                ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                ckpt_list.sort(key=os.path.getmtime)

                if ckpt_list.__len__() >= max_ckpt_save_num:
                    for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                        os.remove(ckpt_list[cur_file_idx])

                ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)
                save_checkpoint(
                    checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
                )

            # Evaluate during training at the requested frequency
            if eval_every_n_epochs and eval_loader is not None and eval_output_dir is not None:
                if trained_epoch % int(eval_every_n_epochs) == 0:
                    # All ranks participate in evaluation for correct metrics when dist is enabled
                    try:
                        if logger is not None and rank == 0:
                            logger.info(f"[Eval] Running validation at epoch {trained_epoch} (every {eval_every_n_epochs} epochs)")

                        # Use underlying module for evaluation to avoid double-wrapping DDP
                        eval_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
                        split = cfg.DATA_CONFIG.DATA_SPLIT['test'] if hasattr(cfg.DATA_CONFIG, 'DATA_SPLIT') else 'val'
                        cur_result_dir = eval_output_dir / (f'epoch_{trained_epoch}') / split

                        # Run evaluation
                        tb_dict = eval_utils.eval_one_epoch(
                            cfg, args, eval_model, eval_loader, trained_epoch, logger if logger is not None else common_utils.create_logger(None),
                            dist_test=dist_test, result_dir=cur_result_dir
                        )

                        # Log summarized eval metrics to tb and wandb (rank 0 only)
                        if rank == 0:
                            if tb_log is not None:
                                for key, val in tb_dict.items():
                                    tb_log.add_scalar(key, val, trained_epoch)
                            if wandb_run is not None:
                                try:
                                    wandb_run.log({f'valid/{k}': v for k, v in tb_dict.items()}, step=accumulated_iter)
                                except Exception:
                                    pass

                            # Best-model tracking and saving
                            if isinstance(best_key, str) and best_key in tb_dict:
                                cur_metric = tb_dict[best_key]
                                if cur_metric > best_metric:
                                    best_metric = cur_metric
                                    best_epoch = trained_epoch
                                    # Save current model as best_model.pth
                                    best_ckpt_path = ckpt_save_dir / 'best_model'
                                    save_checkpoint(
                                        checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=best_ckpt_path,
                                    )
                                    if logger is not None:
                                        logger.info(f"[Eval] New best {best_key}={cur_metric:.6f} at epoch {trained_epoch}. Saved to {best_ckpt_path}.pth")
                                    # Optionally log artifact to wandb
                                    try:
                                        from importlib import import_module
                                        wandb_mod = import_module('wandb') if wandb_run is not None else None
                                        save_artifacts = getattr(getattr(cfg, 'WANDB', {}), 'SAVE_ARTIFACTS', True)
                                        if wandb_mod is not None and save_artifacts:
                                            art = wandb_mod.Artifact('best_model', type='model')
                                            art.add_file(str((best_ckpt_path.with_suffix('.pth'))))
                                            wandb_run.log_artifact(art)
                                        if wandb_run is not None:
                                            wandb_run.summary['best_epoch'] = int(best_epoch)
                                            wandb_run.summary['best_metric'] = float(best_metric)
                                    except Exception:
                                        pass
                    except Exception as e:
                        if logger is not None and rank == 0:
                            logger.warning(f"[Eval] Validation at epoch {trained_epoch} failed: {e}")


def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    try:
        import pcdet
        version = 'pcdet+' + pcdet.__version__
    except:
        version = 'none'

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state, 'version': version}


def save_checkpoint(state, filename='checkpoint'):
    if False and 'optimizer_state' in state:
        optimizer_state = state['optimizer_state']
        state.pop('optimizer_state', None)
        optimizer_filename = '{}_optim.pth'.format(filename)
        if torch.__version__ >= '1.4':
            torch.save({'optimizer_state': optimizer_state}, optimizer_filename, _use_new_zipfile_serialization=False)
        else:
            torch.save({'optimizer_state': optimizer_state}, optimizer_filename)

    filename = '{}.pth'.format(filename)
    if torch.__version__ >= '1.4':
        torch.save(state, filename, _use_new_zipfile_serialization=False)
    else:
        torch.save(state, filename)


def disable_augmentation_hook(hook_config, dataloader, total_epochs, cur_epoch, cfg, flag, logger):
    """
    This hook turns off the data augmentation during training.
    """
    if hook_config is not None:
        DisableAugmentationHook = hook_config.get('DisableAugmentationHook', None)
        if DisableAugmentationHook is not None:
            num_last_epochs = DisableAugmentationHook.NUM_LAST_EPOCHS
            if (total_epochs - num_last_epochs) <= cur_epoch and not flag:
                DISABLE_AUG_LIST = DisableAugmentationHook.DISABLE_AUG_LIST
                dataset_cfg=cfg.DATA_CONFIG
                logger.info(f'Disable augmentations: {DISABLE_AUG_LIST}')
                dataset_cfg.DATA_AUGMENTOR.DISABLE_AUG_LIST = DISABLE_AUG_LIST
                dataloader._dataset.data_augmentor.disable_augmentation(dataset_cfg.DATA_AUGMENTOR)
                flag = True
    return flag
