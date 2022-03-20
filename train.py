import argparse
from pathlib import Path
import os
import shutil
import random

from datetime import datetime
import numpy as np
from tqdm import tqdm
import pdb
st = pdb.set_trace

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
import torchvision

from utils import utils
from utils import utils_html
from utils.utils_train import get_dataset, get_fixed_language_model, \
    get_text_feature_extractor, clip_encode_image, save_model, \
    prepare_lr_scheduler, dummy_lr_scheduler_step, get_optimizer, \
    get_vae_model, get_tokenizer
from utils.utils_train import visualize_train as visualize


# helpers

def exists(val):
    return val is not None


def get_trainable_params(model):
    return [params for params in model.parameters() if params.requires_grad]


def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value


def group_weight(model):
    group_decay, group_no_decay = [], []
    for params in model.named_parameters():
        if 'transformer' in params[0]:
            if 'bias' in params[0] or 'norm' in params[0]:
                group_no_decay.append(params[1])
                continue
        group_decay.append(params[1])

    assert len(list(
        model.parameters())) == len(group_decay) + len(group_no_decay)
    groups = [
        dict(params=group_decay),
        dict(params=group_no_decay, weight_decay=.0)
    ]
    return groups


def sample_data(loader, sampler=None):
    epoch = -1
    while True:
        epoch += 1
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            yield batch


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def model_to_gpu(model, gpu, is_train):
    if is_train:
        if gpu is not None:
            model.cuda(gpu)
            model = DDP(model, device_ids=[gpu], find_unused_parameters=True)
        else:
            model.cuda()
            model = DDP(model, find_unused_parameters=True)
    else:
        model.cuda()
        model = nn.DataParallel(model)

    return model


def cleanup():
    dist.destroy_process_group()


# reconstitute vae and dalle params

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0]  #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(
        token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9)


def reduce_loss(loss):  # TODO
    return loss


def main():
    # argument parsing

    from utils.utils_args import get_args_train
    args, parser = get_args_train()

    args.multiprocessing_distributed = True  # TODO: always use multiprocessing_distributed
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.world_batch_size = args.batch_size
    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker,
                 nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args))
    else:
        main_worker(
            args.gpu_ids,
            ngpus_per_node,
            args,
        )


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    torch.backends.cudnn.benchmark = True

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.multiprocessing_distributed and args.gpu != 0:

        def print_pass(*args):
            pass

        __builtins__['print'] = print_pass

    if args.distributed:
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + args.gpu

        utils.seed_everything(args.seed + args.rank)  # TODO

        dist.init_process_group(backend=args.dist_backend,
                                init_method=args.dist_url,
                                world_size=args.world_size,
                                rank=args.rank)
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(
                (args.workers + ngpus_per_node - 1) / ngpus_per_node)

    else:
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    assert Path(args.image_text_folder).exists(
    ), f'The path {args.image_text_folder} was not found.'

    def is_root_worker():
        return (not args.multiprocessing_distributed
                or (args.multiprocessing_distributed
                    and args.rank % ngpus_per_node == 0))

    # constants

    args.debug = False
    args.lr_decay = not args.no_lr_decay
    args.truncate_captions = True
    args.num_visuals *= args.visual

    if args.ar:
        args.debug = False
        args.mask_predict_steps = [0]
        args.mask_predict_steps1 = 0
        args.num_visuals = max(1, args.num_visuals)
    if args.dm:
        args.debug = False

    # Mask-Predict hyperparameters
    mp_config = {
        'T1_n': args.mp_T1n,
        'T2_n': args.mp_T2n,
        'T3_n': args.mp_T3n,
        'N1_n': args.mp_N1n,
        'N2_n': args.mp_N2n,
        'N3_n': args.mp_N3n,
        'N4_n': args.mp_N4n,
        'T1_t': args.mp_T1t,
        'T2_t': args.mp_T2t,
        'T3_t': args.mp_T3t,
        'N1_t': args.mp_N1t,
        'N2_t': args.mp_N2t,
        'N3_t': args.mp_N3t,
        'N4_t': args.mp_N4t,
        'T': args.mp_T,
        'B': args.mp_B,
    }
    args.mp_config = mp_config

    GRAD_CLIP_NORM = args.clip_grad_norm
    ATTN_TYPES = tuple(args.attn_types.split(','))
    MSM_STRATEGY_PROB = np.array(
        list(map(float, args.msm_strategy_prob.split(','))))
    MSM_STRATEGY_PROB /= MSM_STRATEGY_PROB.sum()
    MSM_BERNOULLI_PROB = list(map(float, args.msm_bernoulli_prob.split(',')))
    RELVID_BERNOULLI_PROB = list(
        map(float, args.relvid_bernoulli_prob.split(',')))
    VID_STRATEGY_PROB = np.array(
        list(map(float, args.vid_strategy_prob.split(','))))
    VID_STRATEGY_PROB /= VID_STRATEGY_PROB.sum()

    # logging

    LOG_DIR = Path(args.log_root) / args.name
    LOG_FILE_NAME = LOG_DIR / 'log.txt'
    args.log_dir = LOG_DIR
    args.log_sample_dir = LOG_DIR / 'samples'
    CKPT_DIR = LOG_DIR / 'weights'

    if is_root_worker():
        os.makedirs(LOG_DIR, exist_ok=True)
        os.makedirs(LOG_DIR / 'samples', exist_ok=True)
        os.makedirs(LOG_DIR / 'weights', exist_ok=True)
        utils.print_args(None, args)
        if args.ar:
            shutil.copyfile('dalle_pytorch/dalle_artv.py',
                            LOG_DIR / 'dalle_artv.py.txt')
        elif args.dm:
            shutil.copyfile('dalle_pytorch/dalle_absorb.py',
                            LOG_DIR / 'dalle_absorb.py.txt')
        else:
            shutil.copyfile('dalle_pytorch/dalle_bert.py',
                            LOG_DIR / 'dalle_bert.py.txt')

    USE_HTML = args.use_html
    LOG_WEB_DIR = LOG_DIR / 'web'
    webpage = None
    if USE_HTML and is_root_worker():
        webpage = utils_html.initialize_webpage(LOG_WEB_DIR, 'DALLE: ' + args.name, False)

    # tokenizer

    if args.fixed_language_model is not None:
        tokenizer2, language_model, text_feature_dim, encode_text = get_fixed_language_model(
            args)
        language_model = model_to_gpu(language_model, args.gpu, True)
        tokenizer = None  # TODO: avoid tokenization and get raw text
    else:
        text_feature_dim = 0
        tokenizer = get_tokenizer(args)

    # model path

    args.use_cvae = args.cvae_path is not None
    DALLE_PATH = Path(args.dalle_path) if exists(args.dalle_path) else None

    model_weights, optim_weights = None, None
    START_ITER = args.start_iter or 0

    # get vae model

    vae, vae_params = get_vae_model(
        args.which_vae,
        vae_path=args.vae_path,
        image_size=args.image_size,
        args=args,
    )

    cvae = None
    if args.cvae_path is not None:
        cvae, cvae_params = get_vae_model(
            args.which_vae,
            vae_path=args.cvae_path,
            image_size=args.image_size,
            args=args,
        )

    dalle_params = dict(
        num_text_tokens=tokenizer.vocab_size if tokenizer else 0,
        text_seq_len=args.text_seq_len,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        dim_head=args.dim_head,
        reversible=args.reversible,
        loss_img_weight=args.loss_img_weight,
        attn_types=ATTN_TYPES,
        text_feature_dim=text_feature_dim,
        fixed_language_model=args.fixed_language_model,
        text_emb_bottleneck=args.text_emb_bottleneck,
        pretrained_transformer=args.pretrained_transformer,
        num_targets=args.num_targets,
        num_visuals=args.num_visuals,
        use_separate_visual_emb=args.use_separate_visual_emb,
        insert_sep=args.insert_sep,
        clip_text_emb=args.clip_text_emb,
    )

    if DALLE_PATH is not None:  # if not resume and dalle_path is given, load weights
        assert exists(DALLE_PATH), 'DALLE model file does not exist'
        ckpt = torch.load(str(DALLE_PATH))
        model_weights = ckpt['weights']

    IMAGE_SIZE = args.image_size or vae.image_size
    args.image_size = vae.image_size = IMAGE_SIZE
    if cvae is not None:
        cvae.image_size = IMAGE_SIZE

    # initialize DALL-E / BERT and optimizer

    if args.ar:
        from dalle_pytorch.dalle_artv import DALLE
        dalle = DALLE(vae=vae, cvae=cvae, **dalle_params)
    elif args.dm:
        from dalle_pytorch.dalle_absorb import AbsorbingDiffusion
        dalle_params['num_timesteps'] = args.dm_timesteps
        dalle_params['use_time_embedding'] = args.dm_use_time_embedding
        dalle_params['use_time_token'] = args.dm_use_time_token
        dalle = AbsorbingDiffusion(vae=vae, cvae=cvae, **dalle_params)
    else:
        from dalle_pytorch.dalle_bert import BERT
        dalle = BERT(vae=vae, cvae=cvae, **dalle_params)
    if args.fp16:
        dalle = dalle.half()

    opt = get_optimizer(args, get_trainable_params(dalle))

    if model_weights is not None:
        dalle.load_state_dict(model_weights)
        del ckpt
    if optim_weights is not None:
        opt.load_state_dict(optim_weights)

    dalle = model_to_gpu(dalle, args.gpu, True)
    dalle_module = dalle.module
    # dist.barrier()

    scheduler, scheduler_step = None, dummy_lr_scheduler_step
    if args.lr_decay:
        scheduler, scheduler_step = prepare_lr_scheduler(args, opt)

    # create dataset and dataloader

    args.is_shuffle = True

    ds = get_dataset(args, tokenizer)
    assert len(ds) > 0, 'dataset is empty'
    if args.limit_train_batches < 1:
        indices = torch.randperm(len(ds))[:int(args.limit_train_batches *
                                               len(ds))]
        ds = torch.utils.data.Subset(ds, indices)
    if is_root_worker():
        print(f'{len(ds)} image-text pairs found for training')

    data_sampler = torch.utils.data.distributed.DistributedSampler(
        ds,
        # num_replicas=args.world_size,
        # rank=args.rank,
    )

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=(data_sampler is None),
        drop_last=True,
        sampler=data_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    distr_dalle, distr_opt, distr_dl, distr_scheduler = dalle, opt, dl, scheduler

    # clipper

    clipper = None
    if args.clip_text_emb is not None:
        clipper = get_clipper(args)
        clipper.cuda()
        requires_grad(clipper, False)

    # training

    loss_mi_clip = torch.tensor(0.0).cuda()
    moved_to_device = False

    if is_root_worker():
        with open(LOG_FILE_NAME, 'a+') as f:
            f.write(
                f"Name: {getattr(args, 'name', 'NA')} Time: {datetime.now()}\n{'-'*50}\n"
            )

    distr_dl_iter = sample_data(distr_dl, data_sampler)

    pbar = range(args.iters)  # TODO
    if is_root_worker():
        pbar = tqdm(pbar,
                    initial=START_ITER,
                    dynamic_ncols=True,
                    smoothing=0.01)

    for idx in pbar:
        i = idx + START_ITER
        which_iter = f"{i:07d}"

        if i > args.iters:
            print('done!')
            break

        if args.negvc:
            text, frames, visuals, visuals_neg, text_neg = next(distr_dl_iter)
            visuals_neg, text_neg = map(lambda t: t.cuda(),
                                        (visuals_neg, text_neg))
        else:
            text, frames, visuals = next(
                distr_dl_iter)  # frames [B, T, C, H, W]
        if args.visual and len(visuals.shape) == 4:
            assert args.num_visuals == 1
            visuals = visuals.unsqueeze(1)

        if args.fp16:
            frames = frames.half()
        frames, visuals = map(lambda t: t.cuda(), (frames, visuals))

        if args.fixed_language_model is not None:
            text_description = text
            with torch.no_grad():
                encoded_input = tokenizer2(
                    text_description,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=args.text_seq_len,
                )
                encoded_input = {
                    'input_ids': encoded_input['input_ids'].cuda(),
                    'attention_mask': encoded_input['attention_mask'].cuda(),
                }
                model_output = language_model(**encoded_input)
                text = mean_pooling(model_output,
                                    encoded_input['attention_mask'])
        else:
            text = text.cuda()
            text_description = None

        target = frames[:, :args.num_targets, ...]

        # Train dalle

        loss_msm, loss_rel, _, loss_vid, fake_sample, text_feat_before, text_feat_after = distr_dalle(
            text,
            visual=visuals if
            (args.visual and
             (args.fullvc or random.random() >= args.dropout_vc)) else None,
            target=target,
            erase_visual=args.rand_visual,
            return_loss=True,
            return_fake=False,
            rel=args.beta_rel > 0,
            vid=args.beta_vid > 0,
            msm_strategy_prob=MSM_STRATEGY_PROB,
            msm_bernoulli_prob=MSM_BERNOULLI_PROB,
            relvid_bernoulli_prob=RELVID_BERNOULLI_PROB,
            vid_strategy_prob=VID_STRATEGY_PROB,
            rel_no_fully_masked=args.rel_no_fully_masked,
            negvc=args.negvc,
            visual_neg=visuals_neg if (args.negvc and args.visual) else None,
            text_neg=text_neg if args.negvc else None,
            visual_pos=None,
            pc_prob=args.pc_prob,
            vc_mode=args.vc_mode,
            face_mode=None,
            visual_aug_mode=args.visual_aug_mode,
            time_schedule=args.dm_time_schedule,
            mask_schedule=args.dm_mask_schedule,
            loss_type=args.dm_loss_type,
        )
        loss = (args.beta_msm * loss_msm + args.beta_rel * loss_rel +
                args.beta_vid * loss_vid)

        distr_opt.zero_grad()
        loss.backward()
        clip_grad_norm_(distr_dalle.parameters(), GRAD_CLIP_NORM)
        distr_opt.step()

        avg_loss = reduce_loss(loss)

        if is_root_worker():
            pbar.set_description((f"loss {avg_loss.item():.4f} "))

        if i % args.log_every == 0 and is_root_worker():
            with open(LOG_FILE_NAME, 'a+') as f:
                f.write((f"iter {i:07d}; "
                         f"MSM {reduce_loss(loss_msm).item():.4f}; "
                         f"REL {reduce_loss(loss_rel).item():.4f}; "
                         f"VID {reduce_loss(loss_vid).item():.4f}; "
                         f"lr {distr_opt.param_groups[0]['lr']}"
                         f"\n"))

        if args.save_every_n_steps > 0 and i % args.save_every_n_steps == 0 and is_root_worker(
        ):
            save_model(
                CKPT_DIR / which_iter,
                params={
                    'iter': i,
                    'hparams': dalle_params,
                    'vae_params': vae_params,
                },
                states={
                    'weights': dalle_module.state_dict(),
                    'optimizer': opt.state_dict(),
                },
            )

        # =================== visualization ======================
        if i % args.sample_every == 0 and is_root_worker():
            visualize(
                args,
                dalle_module,
                tokenizer,
                {
                    'description': text_description,
                    'text': text,
                    'target': frames,
                    'visual': visuals,
                },
                which_iter,
                webpage,
            )
        # ========================================================

        if args.lr_decay and (i + 1) % args.lr_scheduler_every == 0:
            scheduler_step(avg_loss)

    # finish

    if is_root_worker():
        save_model(
            CKPT_DIR / 'last',
            params={
                'iter': i,
                'hparams': dalle_params,
                'vae_params': vae_params,
            },
            states={
                'weights': dalle_module.state_dict(),
                'optimizer': opt.state_dict(),
            },
        )

    cleanup()


if __name__ == "__main__":
    main()
