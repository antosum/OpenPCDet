from functools import partial

import re
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched

from .fastai_optim import OptimWrapper
from .learning_schedules_fastai import CosineWarmupLR, OneCycle, CosineAnnealing


# ---------- Utilities for fine-tuning parameter groups ----------
_BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)


def _resolve_module(model: nn.Module, path: str) -> nn.Module:
    """Resolve dotted/bracketed selector like 'backbone_2d.blocks[2]' to a submodule."""
    cur = model
    for token in path.split('.'):
        m = re.match(r"^(\w+)(\[(\d+)\])?$", token)
        if m is None:
            raise AttributeError(f"Invalid selector token: {token}")
        name, _, idx = m.groups()
        cur = getattr(cur, name)
        if idx is not None:
            cur = cur[int(idx)]
    return cur


def _collect_params(mod: nn.Module) -> set:
    return {p for p in mod.parameters(recurse=True)}


def _build_param_groups_adamw(model: nn.Module, optim_cfg):
    """
    Build AdamW parameter groups from OPTIMIZATION config with:
    - FREEZE_MODULES: list of selectors to fully freeze
    - PARAM_GROUPS: list of {name, select: [selectors|rest], max_lr, weight_decay?}
    Excludes WD for 1D params (bias + norm weights) by default.
    Returns (param_groups, max_lrs), where param_groups is ready for torch.optim.AdamW
    and max_lrs is a list aligned with logical groups (one per logical group).
    """
    freeze_selectors = list(optim_cfg.get('FREEZE_MODULES', []))
    groups_cfg = list(optim_cfg.get('PARAM_GROUPS', []))

    # Gather all named parameters for convenience
    all_named = list(model.named_parameters())
    all_params = [p for _, p in all_named]
    name_by_param = {p: n for n, p in all_named}

    # Freeze requested modules
    frozen_params = set()
    for sel in freeze_selectors:
        try:
            mod = _resolve_module(model, sel)
        except Exception:
            # Skip if module does not exist in this architecture
            continue
        for p in _collect_params(mod):
            p.requires_grad = False
            frozen_params.add(p)

    assigned = set()
    logical_groups = []  # list of dict(name, params)

    # Helper to add a logical group from selectors
    def add_group(name: str, selectors: list):
        params = set()
        for sel in selectors:
            try:
                mod = _resolve_module(model, sel)
            except Exception:
                continue
            params |= _collect_params(mod)
        # filter out frozen and already assigned
        params = [p for p in params if p.requires_grad and (p not in assigned)]
        for p in params:
            assigned.add(p)
        if params:
            logical_groups.append({
                'name': name,
                'params': params
            })

    # First, add explicitly selected groups in order
    for g in groups_cfg:
        sels = list(g.get('select', []))
        if len(sels) == 1 and sels[0] == 'rest':
            # Defer rest to the end
            continue
        add_group(g.get('name', 'group'), sels)

    # Handle 'rest' sentinel if present
    if any((len(g.get('select', [])) == 1 and g['select'][0] == 'rest') for g in groups_cfg):
        rest_name = next((g.get('name', 'rest') for g in groups_cfg if len(g.get('select', [])) == 1 and g['select'][0] == 'rest'), 'rest')
        rest_params = [p for p in all_params if p.requires_grad and (p not in assigned) and (p not in frozen_params)]
        if rest_params:
            logical_groups.append({
                'name': rest_name,
                'params': rest_params
            })

    # If no PARAM_GROUPS provided, fallback to one group with all trainable params
    if not groups_cfg:
        rest_params = [p for p in all_params if p.requires_grad]
        logical_groups.append({'name': 'all', 'params': rest_params})

    # Now split each logical group into decay/no_decay param_groups and collect max_lrs
    div_factor = float(optim_cfg.get('DIV_FACTOR', 25.0))
    default_wd = float(optim_cfg.get('WEIGHT_DECAY', 0.0))
    param_groups = []
    max_lrs = []
    for lg in logical_groups:
        g_cfg = next((g for g in groups_cfg if g.get('name') == lg['name']), {})
        g_max_lr = float(g_cfg.get('max_lr', optim_cfg.LR))
        g_wd = float(g_cfg.get('weight_decay', default_wd))

        # Separate decay and no_decay
        decay, no_decay = [], []
        for p in lg['params']:
            n = name_by_param.get(p, '')
            if p.ndim == 1 or n.endswith('.bias'):
                no_decay.append(p)
            else:
                decay.append(p)

        init_lr = g_max_lr / div_factor
        if decay:
            param_groups.append({'params': decay, 'lr': init_lr, 'weight_decay': g_wd, 'group_name': f"{lg['name']}_decay"})
        if no_decay:
            param_groups.append({'params': no_decay, 'lr': init_lr, 'weight_decay': 0.0, 'group_name': f"{lg['name']}_nodecay"})
        max_lrs.append(g_max_lr)

    return param_groups, max_lrs


class _StepCompat:
    """Adapter to keep .step(it, epoch) signature used in OpenPCDet."""
    def __init__(self, sched):
        self._sched = sched

    def step(self, step, epoch=None):
        self._sched.step()

    def state_dict(self):
        return self._sched.state_dict()

    def load_state_dict(self, state_dict):
        return self._sched.load_state_dict(state_dict)


def build_optimizer(model, optim_cfg):
    opt_name = optim_cfg.OPTIMIZER.lower()

    if opt_name in ['adam', 'sgd']:
        if opt_name == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=optim_cfg.LR, weight_decay=optim_cfg.WEIGHT_DECAY)
        else:
            optimizer = optim.SGD(
                model.parameters(), lr=optim_cfg.LR, weight_decay=optim_cfg.WEIGHT_DECAY,
                momentum=optim_cfg.MOMENTUM
            )
        return optimizer

    # Legacy fastai-based options kept for backward-compatibility
    if opt_name in ['adam_onecycle', 'adam_cosineanneal']:
        def children(m: nn.Module):
            return list(m.children())

        def num_children(m: nn.Module) -> int:
            return len(children(m))

        flatten_model = lambda m: sum(map(flatten_model, m.children()), []) if num_children(m) else [m]
        get_layer_groups = lambda m: [nn.Sequential(*flatten_model(m))]
        betas = optim_cfg.get('BETAS', (0.9, 0.99))
        betas = tuple(betas)
        optimizer_func = partial(optim.Adam, betas=betas)
        optimizer = OptimWrapper.create(
            optimizer_func, 3e-3, get_layer_groups(model), wd=optim_cfg.WEIGHT_DECAY, true_wd=True, bn_wd=True
        )
        return optimizer

    # New: plain AdamW with config-driven param groups
    if opt_name in ['adamw', 'adamw_onecycle']:
        param_groups, _ = _build_param_groups_adamw(model, optim_cfg)
        betas = optim_cfg.get('BETAS', (0.9, 0.999))
        optimizer = optim.AdamW(param_groups, lr=optim_cfg.LR, betas=tuple(betas))
        # Attach finetune metadata for logging
        try:
            optimizer._freeze_modules = list(optim_cfg.get('FREEZE_MODULES', []))
            optimizer._bn_eval_during_train = bool(optim_cfg.get('BN_EVAL_DURING_TRAIN', False))
            optimizer._bn_freeze_affine = bool(optim_cfg.get('BN_FREEZE_AFFINE', False))
            optimizer._div_factor = float(optim_cfg.get('DIV_FACTOR', 25.0))
        except Exception:
            pass
        return optimizer

    raise NotImplementedError


def get_finetune_groups_summary(optimizer, optim_cfg):
    """Produce a summary dict of resolved param groups for logging to wandb.
    Groups are identified by optimizer.param_groups[*]['group_name'] which encodes '<logical>_decay|_nodecay'.
    """
    # Map from logical group name -> configured max_lr
    groups_cfg = {g.get('name'): g for g in list(optim_cfg.get('PARAM_GROUPS', []))}
    def lr_for_logical(name: str) -> float:
        g = groups_cfg.get(name)
        return float(g.get('max_lr', optim_cfg.LR)) if g is not None else float(optim_cfg.LR)

    # Aggregate counts per logical group
    agg = {}
    for pg in getattr(optimizer, 'param_groups', []):
        gname = pg.get('group_name')
        if not gname:
            # Single-group fallback
            total = sum(p.numel() for p in pg.get('params', []))
            return {
                'finetune': {
                    'groups': [{
                        'name': 'all',
                        'params_count': total,
                        'decay_count': total,
                        'nodecay_count': 0,
                        'max_lr': float(optim_cfg.LR),
                        'weight_decay': float(optim_cfg.WEIGHT_DECAY),
                        'init_lr': float(optim_cfg.LR) / float(optim_cfg.get('DIV_FACTOR', 25.0))
                    }],
                    'freeze_modules': list(optim_cfg.get('FREEZE_MODULES', [])),
                    'bn_eval_during_train': bool(optim_cfg.get('BN_EVAL_DURING_TRAIN', False)),
                    'bn_freeze_affine': bool(optim_cfg.get('BN_FREEZE_AFFINE', False)),
                }
            }
        logical, kind = gname.rsplit('_', 1)
        entry = agg.setdefault(logical, {
            'name': logical,
            'params_count': 0,
            'decay_count': 0,
            'nodecay_count': 0,
            'max_lr': lr_for_logical(logical),
            'weight_decay': None,
            'init_lr': lr_for_logical(logical) / float(optim_cfg.get('DIV_FACTOR', 25.0)),
        })
        cnt = sum(p.numel() for p in pg.get('params', []))
        entry['params_count'] += cnt
        if kind == 'decay':
            entry['decay_count'] += cnt
            entry['weight_decay'] = float(pg.get('weight_decay', 0.0))
        else:
            entry['nodecay_count'] += cnt

    return {
        'finetune': {
            'groups': list(agg.values()),
            'freeze_modules': list(optim_cfg.get('FREEZE_MODULES', [])),
            'bn_eval_during_train': bool(optim_cfg.get('BN_EVAL_DURING_TRAIN', False)),
            'bn_freeze_affine': bool(optim_cfg.get('BN_FREEZE_AFFINE', False)),
        }
    }


def build_scheduler(optimizer, total_iters_each_epoch, total_epochs, last_epoch, optim_cfg):
    decay_steps = [x * total_iters_each_epoch for x in optim_cfg.DECAY_STEP_LIST]
    def lr_lbmd(cur_epoch):
        cur_decay = 1
        for decay_step in decay_steps:
            if cur_epoch >= decay_step:
                cur_decay = cur_decay * optim_cfg.LR_DECAY
        return max(cur_decay, optim_cfg.LR_CLIP / optim_cfg.LR)

    lr_warmup_scheduler = None
    total_steps = total_iters_each_epoch * total_epochs

    opt_name = optim_cfg.OPTIMIZER.lower()
    if opt_name == 'adam_onecycle':
        lr_scheduler = OneCycle(
            optimizer, total_steps, optim_cfg.LR, list(optim_cfg.MOMS), optim_cfg.DIV_FACTOR, optim_cfg.PCT_START
        )
        return lr_scheduler, lr_warmup_scheduler
    elif opt_name == 'adam_cosineanneal':
        lr_scheduler = CosineAnnealing(
            optimizer, total_steps, total_epochs, optim_cfg.LR, list(optim_cfg.MOMS), optim_cfg.PCT_START, optim_cfg.WARMUP_ITER
        )
        return lr_scheduler, lr_warmup_scheduler
    elif opt_name == 'adamw_onecycle':
        from torch.optim.lr_scheduler import OneCycleLR
        # Construct max_lr list aligned with optimizer.param_groups order (decay/nodecay per logical group)
        groups_cfg = {g.get('name'): g for g in list(optim_cfg.get('PARAM_GROUPS', []))}
        def lr_for_logical(name: str) -> float:
            g = groups_cfg.get(name)
            if g is None:
                return float(optim_cfg.LR)
            return float(g.get('max_lr', optim_cfg.LR))

        max_lr_list = []
        for pg in optimizer.param_groups:
            gname = pg.get('group_name', None)
            if gname is None:
                # Fallback: single group setup
                max_lr_list.append(float(optim_cfg.LR))
                continue
            # group_name format: '<logical>_decay' or '<logical>_nodecay'
            logical = gname.rsplit('_', 1)[0]
            max_lr_list.append(lr_for_logical(logical))

        scheduler = OneCycleLR(
            optimizer,
            max_lr=max_lr_list,
            total_steps=total_steps,
            pct_start=float(optim_cfg.get('PCT_START', 0.4)),
            div_factor=float(optim_cfg.get('DIV_FACTOR', 25.0)),
            final_div_factor=float(optim_cfg.get('FINAL_DIV_FACTOR', 1e4)),
            anneal_strategy='cos',
            cycle_momentum=True
        )
        return _StepCompat(scheduler), lr_warmup_scheduler
    else:
        lr_scheduler = _StepCompat(lr_sched.LambdaLR(optimizer, lr_lbmd, last_epoch=last_epoch))

        if optim_cfg.LR_WARMUP:
            lr_warmup_scheduler = _StepCompat(CosineWarmupLR(
                optimizer, T_max=optim_cfg.WARMUP_EPOCH * len(total_iters_each_epoch),
                eta_min=optim_cfg.LR / optim_cfg.DIV_FACTOR
            ))

        return lr_scheduler, lr_warmup_scheduler
