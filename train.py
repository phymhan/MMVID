import argparse
from pathlib import Path
import os
import shutil
import random

from datetime import datetime
import numpy as np
import natsort
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

import utils
import utils_html

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


def get_vae_model(which_vae,
                  vae_params=None,
                  vae_path=None,
                  image_size=None,
                  args=None):
    # if vae_params is given (not None), RESUMING from custom DiscreteVAE(**vae_params)
    # weight loading is handled in dalle model
    if args.dalle_path:
        vae_path = None
    if which_vae == 'vqgan1024':
        from dalle_pytorch.vae import VQGanVAE1024
        vae = VQGanVAE1024(vae_path=vae_path, image_size=image_size)
        vae_params = None
    elif which_vae == 'openai':
        from dalle_pytorch.vae import OpenAIDiscreteVAE
        vae = OpenAIDiscreteVAE()
        vae.enc.blocks.output.conv.use_float16 = True
        vae_params = None
    elif which_vae == 'custom':
        from dalle_pytorch.vae import DiscreteVAE
        if exists(vae_path) and Path(vae_path).exists():
            loaded_obj = torch.load(str(vae_path))
            vae_params, vae_weights = loaded_obj['hparams'], loaded_obj[
                'weights']
            vae = DiscreteVAE(**vae_params)
            vae.load_state_dict(vae_weights)
        elif exists(vae_params):
            vae = DiscreteVAE(**vae_params)
        else:
            raise RuntimeError(
                "At least one of vae_path and vae_params should exist.")
    else:
        raise NotImplementedError
    return vae, vae_params


def get_dalle(args, vae, dalle_params):
    dalle = BERT(vae=vae, **dalle_params)

    if args.fp16:
        dalle = dalle.half()

    return dalle


def get_optimizer(args, params):
    if args.optimizer == 'adam':
        from torch.optim import Adam
        opt = Adam(params,
                   lr=args.learning_rate,
                   weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        from torch.optim import AdamW
        # vqgan uses betas (0.9, 0.95)
        opt = AdamW(params,
                    lr=args.learning_rate,
                    betas=(0.9, 0.95),
                    weight_decay=args.weight_decay)
    else:
        raise NotImplementedError
    return opt


def get_tokenizer(args):
    if args.which_tokenizer == 'yttm':
        from dalle_pytorch.tokenizer import YttmTokenizer
        tokenizer = YttmTokenizer(args.bpe_path)
    elif args.which_tokenizer == 'hug':
        from dalle_pytorch.tokenizer import HugTokenizer
        tokenizer = HugTokenizer(args.bpe_path)
    elif args.which_tokenizer == 'simple':
        from dalle_pytorch.tokenizer import SimpleTokenizer
        tokenizer = SimpleTokenizer()
    elif args.which_tokenizer == 'chinese':
        from dalle_pytorch.tokenizer import ChineseTokenizer
        tokenizer = ChineseTokenizer()
    else:
        raise NotImplementedError
    return tokenizer


def get_dataset(args, tokenizer):
    args.truncate_captions = True
    if args.dataset_keys is not None:
        assert Path(args.dataset_keys).exists()
        with open(args.dataset_keys, 'r') as f:
            keys = [k.rstrip() for k in f.readlines()]
    else:
        keys = None
    if args.dataset == 'video_text':
        from dalle_pytorch.loader import TextVideoDataset
        ds = TextVideoDataset(
            args.image_text_folder,
            text_len=args.text_seq_len,
            image_size=args.image_size,
            resize_ratio=args.resize_ratio,
            truncate_captions=args.truncate_captions,
            tokenizer=tokenizer,
            shuffle=args.is_shuffle,
            mode='video',
            deterministic=args.deterministic,
            frame_num=args.frame_num,
            frame_step=args.frame_step,
            return_vc=True,
            video_only=args.video_only,
            cache=args.dataset_cache,
            return_neg=args.negvc,
            drop_sentence=args.drop_sentence,
            keys=keys,
        )
    elif args.dataset == 'mp4_text':
        from dalle_pytorch.loader import TextMP4Dataset
        ds = TextMP4Dataset(
            args.image_text_folder,
            text_len=args.text_seq_len,
            image_size=args.image_size,
            resize_ratio=args.resize_ratio,
            truncate_captions=args.truncate_captions,
            tokenizer=tokenizer,
            shuffle=args.is_shuffle,
            mode='video',
            deterministic=args.deterministic,
            frame_num=args.frame_num,
            frame_step=args.frame_step,
            return_vc=True,
            cache=args.dataset_cache,
            video_only=args.video_only,
            keys=keys,
        )
    elif args.dataset == 'imagestack_text':
        from dalle_pytorch.loader import TextImageStackDataset
        ds = TextImageStackDataset(
            args.image_text_folder,
            text_len=args.text_seq_len,
            image_size=args.image_size,
            resize_ratio=args.resize_ratio,
            truncate_captions=args.truncate_captions,
            tokenizer=tokenizer,
            shuffle=args.is_shuffle,
            mode='video',
            deterministic=args.deterministic,
            frame_num=args.frame_num,
            frame_step=args.frame_step,
            return_vc=True,
            cache=args.dataset_cache,
        )
    elif args.dataset == 'shape_attr':
        from dalle_pytorch.loader_ext import ShapeAttrDataset
        ds = ShapeAttrDataset(
            args.image_text_folder,
            text_len=args.text_seq_len,
            image_size=args.image_size,
            resize_ratio=args.resize_ratio,
            truncate_captions=args.truncate_captions,
            tokenizer=tokenizer,
            shuffle=args.is_shuffle,
            mode='video',
            deterministic=args.deterministic,
            frame_num=args.frame_num,
            frame_step=args.frame_step,
            return_vc=True,
            cache=args.dataset_cache,
            attr_mode=args.attr_mode,
            keys=keys,
            return_neg=args.negvc,
        )
    elif args.dataset == 'vox':
        from dalle_pytorch.loader_ext import VoxDataset
        ds = VoxDataset(
            args.image_text_folder,
            text_len=args.text_seq_len,
            image_size=args.image_size,
            resize_ratio=args.resize_ratio,
            truncate_captions=args.truncate_captions,
            tokenizer=tokenizer,
            shuffle=args.is_shuffle,
            mode='video',
            deterministic=args.deterministic,
            frame_num=args.frame_num,
            frame_step=args.frame_step,
            return_vc=True,
            video_only=args.video_only,
            cache=args.dataset_cache,
            return_neg=args.negvc,
            attr_mode=args.attr_mode,
        )
    elif args.dataset == 'iper':
        from dalle_pytorch.loader_ext import iPERDataset
        ds = iPERDataset(
            args.image_text_folder,
            text_len=args.text_seq_len,
            image_size=args.image_size,
            resize_ratio=args.resize_ratio,
            truncate_captions=args.truncate_captions,
            tokenizer=tokenizer,
            shuffle=args.is_shuffle,
            mode='video',
            deterministic=args.deterministic,
            frame_num=args.frame_num,
            frame_step=args.frame_step,
            return_vc=True,
            video_only=args.video_only,
            cache=args.dataset_cache,
            return_neg=args.negvc,
            drop_sentence=args.drop_sentence,
            slow=args.slow,
            keys=keys,
        )
    else:
        raise NotImplementedError
    return ds


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0]  #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(
        token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9)


def get_fixed_language_model(args):
    tokenizer2, language_model = None, None
    text_feature_dim, encode_text = 0, None
    if args.fixed_language_model == 'roberta-large':
        from transformers import RobertaTokenizer, RobertaModel
        MODEL = "roberta-large"
        tokenizer2 = RobertaTokenizer.from_pretrained('roberta-large')
        language_model = RobertaModel.from_pretrained('roberta-large').cuda()
        text_feature_dim = 1024

        @torch.no_grad()
        def encode_text(descriptions, device='cuda'):
            encoded_input = tokenizer2(
                descriptions,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=args.text_seq_len,
            )
            encoded_input = {
                'input_ids': encoded_input['input_ids'].to(device),
                'attention_mask': encoded_input['attention_mask'].to(device),
            }
            output = language_model(**encoded_input)
            embeddings = mean_pooling(output, encoded_input['attention_mask'])
            return embeddings

    elif args.fixed_language_model == 'twitter-xlm-roberta-base':
        from transformers import AutoTokenizer, AutoModel, AutoConfig
        MODEL = "cardiffnlp/twitter-xlm-roberta-base"
        tokenizer2 = AutoTokenizer.from_pretrained(MODEL)
        config = AutoConfig.from_pretrained(MODEL)
        language_model = AutoModel.from_pretrained(MODEL).cuda()
        text_feature_dim = 768

        @torch.no_grad()
        def encode_text(descriptions, device='cuda'):
            encoded_input = tokenizer2(
                descriptions,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=args.text_seq_len,
            )
            encoded_input = {
                'input_ids': encoded_input['input_ids'].to(device),
                'attention_mask': encoded_input['attention_mask'].to(device),
            }
            output = language_model(**encoded_input)
            embeddings = mean_pooling(output, encoded_input['attention_mask'])
            return embeddings

    else:
        raise NotImplementedError

    return tokenizer2, language_model, text_feature_dim, encode_text


def get_text_feature_extractor(args):
    text_feature_dim = 0
    text_feature_extractor = None
    if args.pretrained_text_feature == 'roberta':
        from transformers import RobertaTokenizer, RobertaModel
        text_feature_tokenizer = RobertaTokenizer.from_pretrained(
            'roberta-large')
        text_feature_extractor = RobertaModel.from_pretrained(
            'roberta-large').cuda()
        text_feature_dim = 1024

        @torch.no_grad()
        def encode_text(descriptions, device='cuda'):
            batch_size = len(descriptions)
            feature = []
            for b in range(batch_size):
                encoded_input = text_feature_tokenizer(descriptions[b],
                                                       return_tensors='pt')
                encoded_input = {
                    'input_ids': encoded_input['input_ids'].to(device),
                    'attention_mask':
                    encoded_input['attention_mask'].to(device),
                }
                output = text_feature_extractor(**encoded_input)
                feature.append(output.last_hidden_state.squeeze(0))
            return feature

    elif args.pretrained_text_feature == 'openai_clip':
        text_feature_extractor = torch.jit.load(
            "pretrained/ViT-B-32.pt").cuda().eval()
        text_feature_tokenizer = SimpleTokenizer()
        text_feature_dim = 512
        context_length = text_feature_extractor.context_length
        dtype = text_feature_extractor.visual.conv1.weight.dtype

        @torch.no_grad()
        def encode_text(descriptions, device='cuda'):
            text_input = text_feature_tokenizer.tokenize(
                descriptions,
                context_length,
                truncate_text=True,
            ).squeeze(0).cuda()
            x = text_feature_extractor.token_embedding(text_input).type(
                dtype)  # [batch_size, n_ctx, d_model]
            x = x + text_feature_extractor.positional_embedding.type(dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = text_feature_extractor.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = text_feature_extractor.ln_final(x).type(dtype)
            return x.float()

    else:
        encode_text = lambda x: x

    return text_feature_dim, encode_text


def clip_encode_image(model, image):
    device = image.device
    # module = model.module
    input_resolution = model.input_resolution.item()
    if image.shape[2] != input_resolution:
        image = F.interpolate(image, (input_resolution, input_resolution))
    image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device)
    image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device)
    image_input = (image - image_mean[:, None, None]) / image_std[:, None,
                                                                  None]
    image_features = model.encode_image(image_input).float()
    image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features


# lr scheduler


def dummy_lr_scheduler_step(*args):
    pass


def prepare_lr_scheduler(args, optimizer):
    if args.lr_scheduler == 'reducelronplateau':
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2,
            cooldown=5,
            min_lr=1e-6,
            verbose=True,
        )

        def step(*args):
            scheduler.step(*args)

        return None, step

    elif args.lr_scheduler == 'steplr':
        from torch.optim.lr_scheduler import StepLR
        scheduler = StepLR(
            optimizer,
            step_size=args.lr_scheduler_step_size,
            gamma=0.5,
        )

        def step(*args):
            scheduler.step()

        return scheduler, step

    elif args.lr_scheduler == 'cosineannealinglr':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.lr_scheduler_step_size,
            eta_min=1e-6,
        )

        def step(*args):
            scheduler.step()

        return scheduler, step

    elif args.lr_scheduler == 'warmupdecaylr':
        from deepspeed.runtime.lr_schedules import WarmupDecayLR
        scheduler = WarmupDecayLR(
            optimizer,
            args.iters,
            1e-6,
            args.learning_rate,
            args.lr_scheduler_warmup,
        )

        def step(*args):
            scheduler.step()

        return scheduler, step

    elif args.lr_scheduler == 'warmuplr':
        from deepspeed.runtime.lr_schedules import WarmupLR
        scheduler = WarmupLR(
            optimizer,
            1e-6,
            args.learning_rate,
            args.lr_scheduler_warmup,
        )

        def step(*args):
            scheduler.step()

        return scheduler, step

    else:
        raise NotImplementedError


def save_model(save_dir, params={}, states={}, name='dalle.pt'):
    path = save_dir / name  # string specifies which epoch or iter
    save_obj = {
        **params,
        **states,
    }
    os.makedirs(path.parent, exist_ok=True)
    torch.save(save_obj, path)


def reduce_loss(loss):  # TODO
    return loss


@torch.no_grad()
def visualize(args,
              dalle_module,
              tokenizer,
              data_batch,
              which_iter,
              webpage=None):
    text_description, text, frames, visuals = data_batch[
        'description'], data_batch['text'], data_batch['target'], data_batch[
            'visual']
    if isinstance(visuals, (list, tuple)):
        visuals = torch.stack(visuals, dim=1)

    N_SAMPLE = min(args.n_sample, args.batch_size)
    N_PER_SAMPLE = args.n_per_sample
    N_FRAME = args.num_targets
    N_FRAME_ = args.num_targets + args.num_visuals * args.visual
    N_VISUAL = args.num_visuals
    IMAGE_SIZE = args.image_size
    LOG_SAMPLE_DIR = args.log_sample_dir
    which_cvae = 'vae' if args.use_cvae is None else 'cvae'

    if args.dm:
        from functools import partial
        generate_images = partial(
            dalle_module.generate_images,
            sample_mode=args.dm_sample_mode,
            use_mask_predict=args.dm_mask_predict,
        )
    else:
        generate_images = dalle_module.generate_images
    pnag_suffix = '_argmax' if args.pnag_argmax else ''
    pnag_suffix = pnag_suffix + '_dynamic' if args.pnag_dynamic else pnag_suffix
    blank_frame_nvc = torch.ones(N_PER_SAMPLE, N_VISUAL, 3, args.image_size,
                                 args.image_size).cuda()
    blank_frame_1 = torch.ones(1, 3, args.image_size, args.image_size).cuda()

    samples_img = []
    captions_img = []
    if args.use_html:
        samples_web = []
        captions_web = []
        nrow_web = []
    for j in range(N_SAMPLE):
        if args.fixed_language_model is None:
            sample_text = text[j:j + 1]
            token_list = sample_text.masked_select(sample_text != 0).tolist()
            decoded_text = tokenizer.decode(token_list)
            if isinstance(decoded_text, (list, tuple)):
                decoded_text = decoded_text[0]
            text_repeat = text[j:j + 1].repeat(N_PER_SAMPLE, 1)
        else:
            decoded_text = text_description[j]
            text_repeat = text[j:j + 1].repeat(N_PER_SAMPLE, 1)

        # Sample (with visual)
        face_mode = None
        frames_recon = dalle_module.recon_images(frames[j:j + 1, :N_FRAME,
                                                        ...])
        visual = visuals[j:j + 1, ...].repeat(N_PER_SAMPLE, 1, 1, 1,
                                              1) if args.visual else None
        if args.visual:
            visual_real = visuals[j, ...]
            visual_recon = dalle_module.recon_images(visual_real,
                                                     which_vae=which_cvae)
            samples_img.append(
                torch.cat((visual_real, frames[j, :N_FRAME, ...]),
                          0))  # real video sequence
            samples_img.append(torch.cat((visual_recon, frames_recon), 0))
            visual_prompt = visuals[j:j + 1, ...].clone().repeat(
                N_PER_SAMPLE, 1, 1, 1, 1)  # b n c h w
            if args.rand_visual:
                visual_prompt[:, :, :, IMAGE_SIZE // 2:, :] = 1
            if args.vc_mode == 'face_8x8':
                block_size = 16
                visual_prompt_ = torch.ones_like(visual_prompt)
                if random.random() < 0.5:
                    face_mode = 'eyes_nose'
                    visual_prompt_[:, :, :, 2 * block_size:5 * block_size,
                                   1 * block_size:7 *
                                   block_size] = visual_prompt[:, :, :, 2 *
                                                               block_size:5 *
                                                               block_size, 1 *
                                                               block_size:7 *
                                                               block_size]
                else:
                    face_mode = 'mouth'
                    visual_prompt_[:, :, :, 5 * block_size:7 * block_size,
                                   2 * block_size:6 *
                                   block_size] = visual_prompt[:, :, :, 5 *
                                                               block_size:7 *
                                                               block_size, 2 *
                                                               block_size:6 *
                                                               block_size]
                visual_prompt = visual_prompt_
            elif args.vc_mode == 'face2_8x8':
                block_size = 16
                visual_prompt_ = torch.ones_like(visual_prompt)
                visual_prompt_[:, 0, ...] = visual_prompt[:, 0, ...]
                face_mode = 'face2'
                visual_prompt_[:, 1:, :, 2 * block_size:6 * block_size,
                               2 * block_size:6 *
                               block_size] = visual_prompt[:, 1:, :,
                                                           2 * block_size:6 *
                                                           block_size,
                                                           2 * block_size:6 *
                                                           block_size]
                visual_prompt = visual_prompt_
            elif args.vc_mode == 'mask2_8x8':
                block_size = 16
                visual_prompt_ = torch.ones_like(visual_prompt)
                visual_prompt_[:, :, :, 1 * block_size:7 * block_size,
                               1 * block_size:7 *
                               block_size] = visual_prompt[:, :, :,
                                                           1 * block_size:7 *
                                                           block_size,
                                                           1 * block_size:7 *
                                                           block_size]
                visual_prompt = visual_prompt_
                face_mode = 'mask2'
            elif args.vc_mode == 'mask_8x8':
                block_size = 16
                visual_prompt_ = torch.ones_like(visual_prompt)
                visual_prompt_[:, :, :, 1 * block_size:7 * block_size,
                               1 * block_size:7 *
                               block_size] = visual_prompt[:, :, :,
                                                           1 * block_size:7 *
                                                           block_size,
                                                           1 * block_size:7 *
                                                           block_size]
                visual_prompt = visual_prompt_
                face_mode = 'mask'
            elif args.vc_mode == 'shape_4x4':
                block_size = 16
                visual_prompt[:, :, :, 1 * block_size:3 * block_size,
                              1 * block_size:3 * block_size] = 1
                face_mode = 'shape'
        else:
            samples_img.append(frames[j, :N_FRAME, ...])  # real video sequence
            samples_img.append(frames_recon)
        captions_img.append(f'{j+1}. {decoded_text}')
        if args.use_html:
            nrow_web += [0]
            if args.visual:
                samples_web += list(torch.split(visual_real, 1, dim=0))
                samples_web += list(torch.split(visual_recon, 1, dim=0))
                captions_web += [f'vc_{jj+1} [real]' for jj in range(N_VISUAL)]
                captions_web += [
                    f'vc_{jj+1} [recon]' for jj in range(N_VISUAL)
                ]
                nrow_web[-1] += 2 * N_VISUAL
            samples_web.append(frames[j, :N_FRAME, ...])
            samples_web.append(frames_recon)
            captions_web += [decoded_text]
            captions_web += ['sequence [recon]']
            nrow_web[-1] += 2
        for mp_steps in args.mask_predict_steps:
            if mp_steps <= 0:
                mp_steps = args.mp_config['T']
            sample_vc, tmp, _ = generate_images(
                text_repeat,
                visual=visual,
                erase_visual=args.rand_visual,
                argmax=args.pnag_argmax,
                dynamic=args.pnag_dynamic,
                debug=args.debug,
                mask_predict_steps=mp_steps,
                vc_mode=args.vc_mode,
                face_mode=face_mode,
                mp_config=args.mp_config,
            )
            if args.visual:
                samples_img.append(
                    torch.cat((visual_prompt, sample_vc),
                              1).reshape(N_PER_SAMPLE * N_FRAME_,
                                         *frames.shape[2:5]))
            else:
                samples_img.append(
                    sample_vc.reshape(N_PER_SAMPLE * N_FRAME,
                                      *frames.shape[2:5]))
            if args.use_html:
                nrow_web += [0]
                if args.visual:
                    samples_web += list(
                        torch.split(visual_prompt[0, ...], 1, dim=0))
                    captions_web += [
                        f'vc_{jj+1} [prompt]' for jj in range(N_VISUAL)
                    ]
                    nrow_web[-1] += N_VISUAL
                samples_web += list(torch.split(sample_vc, 1, dim=0))
                captions_web += [
                    f'sample {jj+1} [T={mp_steps}]'
                    for jj in range(N_PER_SAMPLE)
                ]
                nrow_web[-1] += N_PER_SAMPLE
            if args.debug:
                os.makedirs(LOG_SAMPLE_DIR / f'{which_iter}_pnag',
                            exist_ok=True)
                tmp.insert(0, frames[j, :N_FRAME, ...])
                tmp = torch.cat(tmp, 0)
                torchvision.utils.save_image(
                    tmp,
                    LOG_SAMPLE_DIR / f'{which_iter}_pnag' /
                    f'{j:02d}{pnag_suffix}_T={mp_steps}.png',
                    nrow=N_FRAME,
                    normalize=True,
                    range=(0, 1))
        mp_steps = args.mask_predict_steps1

        if args.visual:
            j2 = (j + 1) % frames.shape[0]
            visual_prompt = visuals[j2:j2 + 1, ...].clone().repeat(
                N_PER_SAMPLE, 1, 1, 1, 1)  # b n c h w
            if args.rand_visual:
                visual_prompt[:, :, :, IMAGE_SIZE // 2:, :] = 1
            if args.vc_mode == 'face_8x8':
                block_size = 16
                visual_prompt_ = torch.ones_like(visual_prompt)
                if random.random() < 0.5:
                    face_mode = 'eyes_nose'
                    visual_prompt_[:, :, :, 2 * block_size:5 * block_size,
                                   1 * block_size:7 *
                                   block_size] = visual_prompt[:, :, :, 2 *
                                                               block_size:5 *
                                                               block_size, 1 *
                                                               block_size:7 *
                                                               block_size]
                else:
                    face_mode = 'mouth'
                    visual_prompt_[:, :, :, 5 * block_size:7 * block_size,
                                   2 * block_size:6 *
                                   block_size] = visual_prompt[:, :, :, 5 *
                                                               block_size:7 *
                                                               block_size, 2 *
                                                               block_size:6 *
                                                               block_size]
                visual_prompt = visual_prompt_
                visual_cf = visuals[j2:j2 + 1, ...]
                visual_cf = visual_cf.repeat(N_PER_SAMPLE, 1, 1, 1, 1)
            elif args.vc_mode == 'face2_8x8':
                block_size = 16
                visual_prompt_ = torch.ones_like(visual_prompt)
                visual_prompt_[:, 0, ...] = visual_prompt[:, 0, ...]
                face_mode = 'face2'
                visual_prompt1 = visuals[j:j + 1,
                                         ...].repeat(N_PER_SAMPLE, 1, 1, 1, 1)
                visual_prompt_[:, 1:, :, 2 * block_size:6 * block_size,
                               2 * block_size:6 *
                               block_size] = visual_prompt1[:, 1:, :,
                                                            2 * block_size:6 *
                                                            block_size,
                                                            2 * block_size:6 *
                                                            block_size]
                visual_prompt = visual_prompt_
                visual_cf = visuals[j:j + 1, ...].clone()  # !!!
                visual_cf[:, 0, ...] = visuals[j2:j2 + 1, 0, ...]
                visual_cf = visual_cf.repeat(N_PER_SAMPLE, 1, 1, 1, 1)
            elif args.vc_mode == 'mask2_8x8':
                block_size = 16
                visual_prompt_ = torch.ones_like(visual_prompt)
                visual_prompt1 = visuals[j:j + 1,
                                         ...].repeat(N_PER_SAMPLE, 1, 1, 1, 1)
                visual_prompt_[:, 0, :, 1 * block_size:7 * block_size,
                               1 * block_size:7 *
                               block_size] = visual_prompt1[:, 0, :,
                                                            1 * block_size:7 *
                                                            block_size,
                                                            1 * block_size:7 *
                                                            block_size]
                visual_prompt_[:, 1, :, 1 * block_size:7 * block_size,
                               1 * block_size:7 *
                               block_size] = visual_prompt[:, 1, :,
                                                           1 * block_size:7 *
                                                           block_size,
                                                           1 * block_size:7 *
                                                           block_size]
                visual_prompt = visual_prompt_
                visual_cf = visuals[j:j + 1, ...].clone()  # !!!
                visual_cf[:, 1, ...] = visuals[j2:j2 + 1, 1, ...]
                visual_cf = visual_cf.repeat(N_PER_SAMPLE, 1, 1, 1, 1)
                face_mode = 'mask2'
            elif args.vc_mode == 'mask_8x8':
                block_size = 16
                visual_prompt_ = torch.ones_like(visual_prompt)
                visual_prompt_[:, :, :, 1 * block_size:7 * block_size,
                               1 * block_size:7 *
                               block_size] = visual_prompt[:, :, :,
                                                           1 * block_size:7 *
                                                           block_size,
                                                           1 * block_size:7 *
                                                           block_size]
                visual_prompt = visual_prompt_
                visual_cf = visuals[j2:j2 + 1,
                                    ...].repeat(N_PER_SAMPLE, 1, 1, 1, 1)
                face_mode = 'mask'
            elif args.vc_mode == 'shape_4x4':
                block_size = 16
                visual_prompt[:, :, :, 1 * block_size:3 * block_size,
                              1 * block_size:3 * block_size] = 1
                visual_cf = visuals[j2:j2 + 1,
                                    ...].repeat(N_PER_SAMPLE, 1, 1, 1, 1)
                face_mode = 'shape'
            else:
                visual_cf = visuals[j2:j2 + 1,
                                    ...].repeat(N_PER_SAMPLE, 1, 1, 1, 1)
            sample_cf, tmp, _ = generate_images(
                text_repeat,
                visual=visual_cf,
                erase_visual=args.rand_visual,
                argmax=args.pnag_argmax,
                dynamic=args.pnag_dynamic,
                debug=args.debug,
                mask_predict_steps=mp_steps,
                vc_mode=args.vc_mode,
                face_mode=face_mode,
                mp_config=args.mp_config,
            )
            samples_img.append(
                torch.cat((visual_prompt, sample_cf),
                          1).reshape(N_PER_SAMPLE * N_FRAME_,
                                     *frames.shape[2:5]))
            if args.use_html:
                samples_web += list(
                    torch.split(visual_prompt[0, ...], 1, dim=0))
                samples_web += list(torch.split(sample_cf, 1, dim=0))
                captions_web += [
                    f'cf_{jj+1} [prompt]' for jj in range(N_VISUAL)
                ]
                captions_web += [
                    f'sample {jj+1}' for jj in range(N_PER_SAMPLE)
                ]
                nrow_web += [N_VISUAL + N_PER_SAMPLE]
            if args.debug:
                # tmp.insert(0, frames[j,:N_FRAME,...])
                tmp = torch.cat(tmp, 0)
                torchvision.utils.save_image(tmp,
                                             LOG_SAMPLE_DIR /
                                             f'{which_iter}_pnag' /
                                             f'cf_{j:02d}{pnag_suffix}.png',
                                             nrow=N_FRAME,
                                             normalize=True,
                                             range=(0, 1))

        if args.visual and not args.fullvc:
            sample_free, tmp, _ = generate_images(
                text_repeat,
                visual=None,
                erase_visual=args.rand_visual,
                argmax=args.pnag_argmax,
                dynamic=args.pnag_dynamic,
                debug=args.debug,
                mask_predict_steps=mp_steps,
                mp_config=args.mp_config,
            )
            samples_img.append(
                torch.cat((blank_frame_nvc, sample_free),
                          1).reshape(N_PER_SAMPLE * N_FRAME_,
                                     *frames.shape[2:5]))
            if args.use_html:
                samples_web += [blank_frame_1] * N_VISUAL
                samples_web += list(torch.split(sample_free, 1, dim=0))
                captions_web += [f'null [prompt]' for jj in range(N_VISUAL)]
                captions_web += [
                    f'sample {jj+1}' for jj in range(N_PER_SAMPLE)
                ]
                nrow_web += [N_VISUAL + N_PER_SAMPLE]
            if args.debug:
                # tmp.insert(0, frames[j,:N_FRAME,...])
                tmp = torch.cat(tmp, 0)
                torchvision.utils.save_image(tmp,
                                             LOG_SAMPLE_DIR /
                                             f'{which_iter}_pnag' /
                                             f'free_{j:02d}{pnag_suffix}.png',
                                             nrow=N_FRAME,
                                             normalize=True,
                                             range=(0, 1))

    samples_img = torch.cat(samples_img)
    torchvision.utils.save_image(samples_img,
                                 LOG_SAMPLE_DIR / f'{which_iter}.png',
                                 nrow=N_FRAME_,
                                 normalize=True,
                                 range=(0, 1))

    with open(LOG_SAMPLE_DIR / f'{which_iter}.txt', 'w') as f:
        f.write('\n'.join(captions_img))

    if args.use_html:
        webpage.add_header(f'iteration {which_iter}')
        utils_html.save_grid(
            webpage=webpage,
            tensor=samples_web,
            caption=captions_web,
            name=which_iter,
            nrow=nrow_web,
            width=min(IMAGE_SIZE, 256),
        )


def main():
    # argument parsing

    parser = argparse.ArgumentParser()

    parser.add_argument('--vae_path',
                        type=str,
                        help='path to your trained discrete VAE')
    parser.add_argument('--cvae_path',
                        type=str,
                        help='path to your trained discrete VAE')
    parser.add_argument('--dalle_path2', type=str, default=None)
    parser.add_argument('--dalle_path',
                        type=str,
                        default=None,
                        help='path to your partially trained DALL-E')
    parser.add_argument('--which_vae', type=str, default='vqgan1024')
    parser.add_argument('--transformer_path', type=str, default=None)
    parser.add_argument(
        '--image_text_folder',
        type=str,
        required=True,
        help='path to your folder of images and text for learning the DALL-E')
    parser.add_argument('--dataset', type=str, default='video_text')
    parser.add_argument('--dataset_keys', type=str, default=None)
    parser.add_argument('--dataset_cache', type=str, default=None)
    parser.add_argument('--video_only', action='store_true')
    parser.add_argument(
        '--truncate_captions',
        dest='truncate_captions',
        action='store_true',
        help=
        'Captions passed in which exceed the max token length will be truncated if this is set.'
    )
    parser.add_argument('--random_resize_crop_lower_ratio',
                        dest='resize_ratio',
                        type=float,
                        default=0.75,
                        help='Random resized crop lower ratio')
    parser.add_argument('--which_tokenizer',
                        type=str,
                        default='simple',
                        help='(yttm | hug | simple | chinese)')
    parser.add_argument('--taming', dest='taming', action='store_true')
    parser.add_argument('--bpe_path',
                        type=str,
                        help='path to your BPE json file')
    parser.add_argument(
        '--fp16',
        action='store_true',
        help='(experimental) - Enable DeepSpeed 16 bit precision. Reduces VRAM.'
    )
    parser.add_argument(
        '--amp',
        action='store_true',
        help=
        'Apex "O1" automatic mixed precision. More stable than 16 bit precision. Can\'t be used in conjunction with deepspeed zero stages 1-3.'
    )
    parser.add_argument('--name',
                        default='dalle_train_transformer',
                        help='experiment name, if not using wandb')
    parser.add_argument('--visual',
                        action='store_true',
                        help='add visual control?')
    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--rank',
                        type=int,
                        default=0,
                        help='node rank for distributed training')
    parser.add_argument('--gpu_ids',
                        type=int,
                        default=None,
                        help='gpu id to use')
    parser.add_argument('--workers',
                        default=16,
                        type=int,
                        help='# data loading workers')
    parser.add_argument('--world_size',
                        default=1,
                        type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--dist_url',
                        default='tcp://localhost:10001',
                        type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist_backend',
                        default='nccl',
                        type=str,
                        help='distributed backend')
    parser.add_argument(
        '--multiprocessing_distributed',
        action='store_true',
        help=
        'Use multi-processing distributed training to launch N processes per node, which has N GPUs.'
    )

    parser.add_argument('--save_every_n_steps',
                        default=2000,
                        type=int,
                        help='Save a checkpoint every n steps')
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size')
    parser.add_argument(
        '--ga_steps',
        default=1,
        type=int,
        help=
        'Number of steps to accumulate gradients across per each iteration. DeepSpeed only.'
    )
    parser.add_argument('--learning_rate',
                        default=3e-4,
                        type=float,
                        help='Learning rate')
    parser.add_argument('--clip_grad_norm',
                        default=0.5,
                        type=float,
                        help='Clip gradient norm')
    parser.add_argument('--lr_decay', dest='lr_decay', action='store_true')
    parser.add_argument('--freeze_transformer', action='store_true')
    parser.add_argument('--tensorboard', action='store_true')
    parser.add_argument('--use_html', action='store_true')
    parser.add_argument("--log_root",
                        type=str,
                        help="where to save training logs",
                        default='logs')
    parser.add_argument("--log_every",
                        type=int,
                        default=200,
                        help="logging every # iters")
    parser.add_argument("--sample_every",
                        type=int,
                        default=1000,
                        help="sample every # iters")
    parser.add_argument('--n_sample',
                        default=4,
                        type=int,
                        help='Number of samples (text) to visualize')
    parser.add_argument('--n_per_sample',
                        default=1,
                        type=int,
                        help='Number of images per text sample to visualize')
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    parser.add_argument('--iters',
                        default=200000,
                        type=int,
                        help='Number of iterations')
    parser.add_argument('--start_iter',
                        default=None,
                        type=int,
                        help='start iter')
    parser.add_argument('--epochs',
                        default=100,
                        type=int,
                        help='Number of epochs')
    parser.add_argument("--limit_train_batches", type=float, default=1)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument(
        '--keep_n_checkpoints',
        default=None,
        type=int,
        help=
        '(Careful) Deletes old deepspeed checkpoints if there are more than n')
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--lr_scheduler',
                        type=str,
                        default='reducelronplateau')
    parser.add_argument('--clip_ranking', action='store_true')
    parser.add_argument('--clip_path',
                        type=str,
                        default=None,
                        help='path to your pretrained CLIP')
    parser.add_argument('--lr_scheduler_every',
                        default=1,
                        type=int,
                        help='step lr scheduler every n steps')
    parser.add_argument('--lr_scheduler_step_size',
                        default=10000,
                        type=int,
                        help='T_max or step_size')
    parser.add_argument('--lr_scheduler_warmup',
                        default=5000,
                        type=int,
                        help='T_max or step_size')
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--beta_msm', default=1.0, type=float)
    parser.add_argument('--beta_rel', default=0.5, type=float)
    parser.add_argument('--beta_vid', default=0, type=float)
    parser.add_argument('--beta_gan', default=0, type=float)
    parser.add_argument('--beta_clip', default=0, type=float)
    parser.add_argument('--which_fake', type=str, default='mask_predict')
    parser.add_argument('--frame_num', default=8, type=int)
    parser.add_argument('--frame_step', default=8, type=int)
    parser.add_argument('--estimate_real', action='store_true')
    parser.add_argument('--pnag_argmax', action='store_true')
    parser.add_argument('--pnag_dynamic', action='store_true')
    parser.add_argument('--rand_visual', action='store_true')
    # parser.add_argument('--oldrel', action = 'store_true')
    parser.add_argument('--newmask', action='store_true')
    parser.add_argument('--fullvc', action='store_true')
    parser.add_argument('--negvc', action='store_true')
    parser.add_argument('--vc_mode', type=str, default=None)
    parser.add_argument('--attr_mode', type=str, default='object')
    parser.add_argument('--n_accum_step', default=1, type=int)
    parser.add_argument('--dropout_vc',
                        type=float,
                        default=0.1,
                        help='prob of visual control to be zeroed')
    parser.add_argument('--msm_strategy_prob',
                        type=str,
                        default='7,1,1,1',
                        help='comma separated list')
    parser.add_argument('--msm_bernoulli_prob',
                        type=str,
                        default='0.2,0.2',
                        help='comma separated list')
    parser.add_argument('--relvid_bernoulli_prob',
                        type=str,
                        default='0.1,0.9',
                        help='comma separated list')
    parser.add_argument('--vid_strategy_prob',
                        type=str,
                        default='1,1,1,1',
                        help='comma separated list')
    parser.add_argument('--rel_no_fully_masked', action='store_true')
    parser.add_argument('--insert_sep', action='store_true')
    parser.add_argument('--mask_predict_steps',
                        nargs='+',
                        default=[0],
                        type=int)
    parser.add_argument('--mask_predict_steps1', default=0, type=int)
    parser.add_argument('--pc_prob',
                        type=float,
                        default=0,
                        help='prob of preservation control')
    parser.add_argument('--drop_sentence', action='store_true')
    parser.add_argument('--fixed_language_model', type=str, default=None)

    parser.add_argument('--dim', default=512, type=int, help='Model dimension')
    parser.add_argument('--text_seq_len',
                        default=256,
                        type=int,
                        help='Text sequence length')
    parser.add_argument('--depth', default=2, type=int, help='Model depth')
    parser.add_argument('--heads',
                        default=8,
                        type=int,
                        help='Model number of heads')
    parser.add_argument('--dim_head',
                        default=64,
                        type=int,
                        help='Model head dimension')
    parser.add_argument('--reversible', dest='reversible', action='store_true')
    parser.add_argument('--loss_img_weight',
                        default=7,
                        type=int,
                        help='Image loss weight')
    parser.add_argument(
        '--attn_types',
        default='full',
        type=str,
        help=
        'comma separated list of attention types. attention type can be: full or sparse or axial_row or axial_col or conv_like.'
    )
    parser.add_argument('--pretrained_transformer', type=str, default='none')
    parser.add_argument('--image_size',
                        default=None,
                        type=int,
                        help='force to use this size if set to > 0')
    parser.add_argument('--num_targets',
                        default=1,
                        type=int,
                        help='number of frames to generate')
    parser.add_argument('--num_visuals',
                        default=1,
                        type=int,
                        help='number of frames to generate')
    parser.add_argument('--use_separate_visual_emb', action='store_true')
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--text_emb_bottleneck', type=str, default=None)
    parser.add_argument('--clip_text_emb', type=str, default=None)
    parser.add_argument('--visual_aug_mode', type=str, default=None)

    parser.add_argument('--mp_T1n',
                        type=int,
                        default=10,
                        help='L1, number of steps for mask')
    parser.add_argument('--mp_T2n',
                        type=int,
                        default=10,
                        help='L2, number of steps for mask')
    parser.add_argument('--mp_T3n',
                        type=int,
                        default=30,
                        help='L3, number of steps for mask')
    parser.add_argument('--mp_N1n',
                        type=float,
                        default=0.9,
                        help='alpha1 for mask')
    parser.add_argument('--mp_N2n',
                        type=float,
                        default=0.1,
                        help='beta1 for mask')
    parser.add_argument('--mp_N3n',
                        type=float,
                        default=0.125,
                        help='alpha2 for mask')
    parser.add_argument('--mp_N4n',
                        type=float,
                        default=0.0625,
                        help='alpha3 for mask')
    parser.add_argument('--mp_T1t',
                        type=int,
                        default=10,
                        help='L1, number of steps for noise')
    parser.add_argument('--mp_T2t',
                        type=int,
                        default=5,
                        help='L2, number of steps for noise')
    parser.add_argument('--mp_T3t',
                        type=int,
                        default=35,
                        help='L3, number of steps for noise')
    parser.add_argument('--mp_N1t',
                        type=float,
                        default=0.,
                        help='alpha1 for noise')
    parser.add_argument('--mp_N2t',
                        type=float,
                        default=0.,
                        help='beta1 for noise')
    parser.add_argument('--mp_N3t',
                        type=float,
                        default=0.,
                        help='alpha2 for noise')
    parser.add_argument('--mp_N4t',
                        type=float,
                        default=0.,
                        help='alpha3 for noise')
    parser.add_argument('--mp_T',
                        type=int,
                        default=20,
                        help='number of total steps for mask-predict')
    parser.add_argument('--mp_B', type=int, default=1, help='beam search size')

    parser.add_argument('--ar', action='store_true')
    parser.add_argument('--slow', action='store_true')
    parser.add_argument('--dm', action='store_true')
    parser.add_argument('--dm_timesteps', type=int, default=32)
    parser.add_argument('--dm_mask_schedule', default='random', type=str)
    parser.add_argument('--dm_time_schedule', default='gamma_linear', type=str)
    parser.add_argument('--dm_loss_type', default='simple', type=str)
    parser.add_argument('--dm_sample_mode', default='gamma_linear', type=str)
    parser.add_argument('--dm_mask_predict', action='store_true')

    parser.add_argument('--dm_use_time_embedding', action='store_true')
    parser.add_argument('--dm_use_time_token', action='store_true')

    args = parser.parse_args()

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
    args.truncate_captions = True
    args.num_visuals *= args.visual

    if args.ar:
        args.debug = False
        args.mask_predict_steps = [0]
        args.mask_predict_steps1 = 0
        args.num_visuals = max(1, args.num_visuals)
    if args.dm:
        args.debug = False

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

    ITERS = args.iters
    BATCH_SIZE = args.batch_size
    GRAD_CLIP_NORM = args.clip_grad_norm
    LR_DECAY = args.lr_decay

    MODEL_DIM = args.dim
    TEXT_SEQ_LEN = args.text_seq_len
    DEPTH = args.depth
    HEADS = args.heads
    DIM_HEAD = args.dim_head
    REVERSIBLE = args.reversible
    LOSS_IMG_WEIGHT = args.loss_img_weight

    ATTN_TYPES = tuple(args.attn_types.split(','))

    N_FRAME = args.num_targets
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

    RESUME = args.resume
    LOG_DIR = Path(args.log_root) / args.name
    LOG_SAMPLE_DIR = LOG_DIR / 'samples'
    LOG_FILE_NAME = LOG_DIR / 'log.txt'
    args.log_dir = LOG_DIR
    args.log_sample_dir = LOG_SAMPLE_DIR
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
        webpage = utils_html.initialize_webpage(LOG_WEB_DIR,
                                                'DALLE: ' + args.name, RESUME)

    # tokenizer

    # tokenizer = get_tokenizer(args)
    # text_feature_dim, encode_text = get_text_feature_extractor(args)
    # if text_feature_dim > 0:
    #     tokenizer = None  # if use fixed text lang model, set tokenizer to None
    if args.fixed_language_model is not None:
        tokenizer2, language_model, text_feature_dim, encode_text = get_fixed_language_model(
            args)
        language_model = model_to_gpu(language_model, args.gpu,
                                      True)  # TODO: false
        tokenizer = None  # TODO: avoid tokenization and get raw text
    else:
        text_feature_dim = 0
        tokenizer = get_tokenizer(args)

    # model path

    VAE_PATH = args.vae_path
    args.use_cvae = args.cvae_path is not None
    DALLE_PATH = Path(args.dalle_path) if exists(args.dalle_path) else None
    if RESUME and DALLE_PATH is None:
        checkpoints = natsort.natsorted(os.listdir(str(LOG_DIR / 'weights')))
        assert len(checkpoints) > 0, f'Nothing to resume from.'
        DALLE_PATH = LOG_DIR / 'weights' / checkpoints[-1] / 'dalle.pt'

    model_weights, optim_weights = None, None
    START_ITER = args.start_iter or 0
    START_EPOCH = 0
    GLOBAL_STEPS = 0  # global_steps
    NUM_DATA_SEEN = 0
    DL_START_ITER = 0

    # get vae model

    if RESUME:
        assert DALLE_PATH.exists(), 'DALL-E model file does not exist'
        print(f"resuming from {DALLE_PATH}")
        loaded_obj = torch.load(str(DALLE_PATH), map_location='cpu')
        dalle_params, vae_params = loaded_obj['hparams'], loaded_obj[
            'vae_params']
        model_weights, optim_weights = loaded_obj['weights'], loaded_obj[
            'optimizer']
        GLOBAL_STEPS = loaded_obj.get('iter', 0)
        NUM_DATA_SEEN = loaded_obj.get('num_data_seen', 0)
        START_ITER = args.start_iter or GLOBAL_STEPS

        vae, vae_params = get_vae_model(args.which_vae,
                                        vae_params=vae_params,
                                        vae_path=VAE_PATH,
                                        image_size=args.image_size)

        dalle_params = dict(**dalle_params)

    else:
        vae, vae_params = get_vae_model(
            args.which_vae,
            vae_path=VAE_PATH,
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
            text_seq_len=TEXT_SEQ_LEN,
            dim=MODEL_DIM,
            depth=DEPTH,
            heads=HEADS,
            dim_head=DIM_HEAD,
            reversible=REVERSIBLE,
            loss_img_weight=LOSS_IMG_WEIGHT,
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

    ####################################################
    if args.dalle_path2 is not None:
        assert exists(args.dalle_path2), 'DALLE model file does not exist'
        args.dalle_path2 = Path(args.dalle_path2)
        ckpt = torch.load(str(args.dalle_path2))
        weights = {}
        model_weights2 = ckpt['weights']
        for m in model_weights2.keys():
            if not m.startswith('transformer'):
                pass
            else:
                weights[m] = model_weights2[m]
        dalle.load_state_dict(weights, strict=False)
        # del weights
    ####################################################

    dalle = model_to_gpu(dalle, args.gpu, True)  # TODO
    dalle_module = dalle.module  # TODO
    # dist.barrier()

    # if DALLE_PATH is not None:
    #     dalle.load_state_dict(
    #         torch.load(str(DALLE_PATH), map_location={f'cuda:{0}': f'cuda:{args.gpu}'})['weights']
    #     )

    # if '[MASK]' in dalle_module.special_token_lut and '[MASK]' in dalle_module.image_token_lut:
    #     with torch.no_grad():
    #         dalle_module.image_emb.weight[dalle_module.image_token_lut['[MASK]']] = dalle_module.special_emb.weight[dalle_module.special_token_lut['[MASK]']]

    scheduler, scheduler_step = None, dummy_lr_scheduler_step
    if LR_DECAY:
        scheduler, scheduler_step = prepare_lr_scheduler(args, opt)

    # create dataset and dataloader

    args.is_shuffle = True  # TODO

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
        batch_size=BATCH_SIZE,
        shuffle=(data_sampler is None),
        drop_last=True,
        sampler=data_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    distr_dalle, distr_opt, distr_dl, distr_scheduler = dalle, opt, dl, scheduler

    # clipper

    # clipper, clip_tokenizer = get_clipper(args)
    clipper = None
    if args.clip_text_emb is not None:
        clipper = get_clipper(args)
        # clipper = model_to_gpu(clipper, args.gpu, True)
        clipper.cuda()
        requires_grad(clipper, False)

    # training

    loss_mi_clip = torch.tensor(0.0).cuda()
    moved_to_device = False

    if RESUME:
        START_EPOCH = NUM_DATA_SEEN // len(ds)
        DL_START_ITER = (NUM_DATA_SEEN % len(ds)) // BATCH_SIZE

    if is_root_worker():
        with open(LOG_FILE_NAME, 'a+') as f:
            f.write(
                f"Name: {getattr(args, 'name', 'NA')} Time: {datetime.now()}\n{'-'*50}\n"
            )

    distr_dl_iter = sample_data(distr_dl, data_sampler)

    pbar = range(ITERS)  # TODO
    if is_root_worker():
        pbar = tqdm(pbar,
                    initial=START_ITER,
                    dynamic_ncols=True,
                    smoothing=0.01)

    for idx in pbar:
        i = idx + START_ITER
        which_iter = f"{i:07d}"
        GLOBAL_STEPS += 1
        NUM_DATA_SEEN += BATCH_SIZE  # !!!

        if i > ITERS:
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
        # text, frames, visuals = map(lambda t: t.cuda(), (text, frames, visuals))
        frames, visuals = map(lambda t: t.cuda(), (frames, visuals))

        if args.fixed_language_model is not None:
            text_description = text
            # text = encode_text(text_description)
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
        if args.clip_text_emb == 'before':
            frame_idx = random.randint(0, frames.shape[1] - 1)
            image_input = frames[:, frame_idx, ...]
            image_feat = clip_encode_image(clipper, image_input)
            loss_mi_clip = -torch.mean(
                F.cosine_similarity(text_feat_before, image_feat.detach()))
        elif args.clip_text_emb == 'after':
            frame_idx = random.randint(0, frames.shape[1] - 1)
            image_input = frames[:, frame_idx, ...]
            image_feat = clip_encode_image(clipper, image_input)
            loss_mi_clip = -torch.mean(
                F.cosine_similarity(text_feat_after, image_feat.detach()))
        loss = (args.beta_msm * loss_msm + args.beta_rel * loss_rel +
                args.beta_vid * loss_vid + args.beta_clip * loss_mi_clip)

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

        if LR_DECAY and (i + 1) % args.lr_scheduler_every == 0:
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

    cleanup()  # TODO


if __name__ == "__main__":
    main()
