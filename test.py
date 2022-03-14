import argparse
from pathlib import Path
from tensorflow.python.ops.gen_math_ops import _clip_by_value_eager_fallback
import torch
from torch import functional
from torch.cuda import get_gencode_flags
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.utils.data import DataLoader

# from dalle_pytorch import distributed_utils
import utils
import os
import sys
import shutil
from tqdm import tqdm
from glob import glob
import time
import natsort
import warnings
# from itertools import islice
import utils_html
import torchvision
import imageio
import torch.nn.functional as F
import numpy as np
import random
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
from datetime import datetime
import pickle
from einops import rearrange

import pdb
st = pdb.set_trace


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

    assert len(list(model.parameters())) == len(group_decay) + len(group_no_decay)
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
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
    model.cuda()
    model = nn.DataParallel(model)

    return model

def cleanup():
    dist.destroy_process_group()

# reconstitute vae and dalle params

def get_vae_model(which_vae, vae_params=None, vae_path=None, image_size=None):
    # if vae_params is given (not None), RESUMING from custom DiscreteVAE(**vae_params)
    # weight loading is handled in dalle model
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
            vae_params, vae_weights = loaded_obj['hparams'], loaded_obj['weights']
            vae = DiscreteVAE(**vae_params)
            vae.load_state_dict(vae_weights)
        elif exists(vae_params):
            vae = DiscreteVAE(**vae_params)
        else:
            raise RuntimeError("At least one of vae_path and vae_params should exist.")
    else:
        raise NotImplementedError
    return vae, vae_params


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
    if args.dataset_keys is not None and args.dataset_keys != "":
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
            deterministic=True,
            frame_num=args.frame_num,
            frame_step=args.frame_step,
            return_vc=True,
            video_only=args.video_only,
            cache=args.dataset_cache,
            return_neg=args.negvc,
            rep_num=args.rep_num,
            drop_sentence=args.eval_drop_sent,
            keys=keys,
            skip_min_len_check=True,
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
            deterministic=True,
            frame_num=args.frame_num,
            frame_step=args.frame_step,
            return_vc=True,
            video_only=args.video_only,
            cache=args.dataset_cache,
            return_neg=args.negvc,
            attr_mode=args.attr_mode,
            args=args,
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
            deterministic=True,
            frame_num=args.frame_num,
            frame_step=args.frame_step,
            return_vc=True,
            video_only=args.video_only,
            cache=args.dataset_cache,
            return_neg=args.negvc,
            rep_num=args.rep_num,
            drop_sentence=args.eval_drop_sent,
            slow=args.slow,
            slow_mode=args.slow_mode,
            keys=keys,
            skip_min_len_check=True,
        )
    else:
        raise NotImplementedError
    return ds

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

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
        text_feature_tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        text_feature_extractor = RobertaModel.from_pretrained('roberta-large').cuda()
        text_feature_dim = 1024

        @torch.no_grad()
        def encode_text(descriptions, device='cuda'):
            batch_size = len(descriptions)
            feature = []
            for b in range(batch_size):
                encoded_input = text_feature_tokenizer(descriptions[b], return_tensors='pt')
                encoded_input = {
                    'input_ids': encoded_input['input_ids'].to(device),
                    'attention_mask': encoded_input['attention_mask'].to(device),
                }
                output = text_feature_extractor(**encoded_input)
                feature.append(output.last_hidden_state.squeeze(0))
            return feature

    elif args.pretrained_text_feature == 'openai_clip':
        text_feature_extractor = torch.jit.load("pretrained/ViT-B-32.pt").cuda().eval()
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
            x = text_feature_extractor.token_embedding(text_input).type(dtype)  # [batch_size, n_ctx, d_model]
            x = x + text_feature_extractor.positional_embedding.type(dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = text_feature_extractor.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = text_feature_extractor.ln_final(x).type(dtype)
            return x.float()

    else:
        encode_text = lambda x: x

    return text_feature_dim, encode_text

# clip model and helpers

def clip_similarity(model, tokenizer, image, description):
    input_resolution = model.input_resolution.item()
    context_length = model.context_length.item()
    # vocab_size = model.vocab_size.item()

    if image.shape[2] != input_resolution:
        image = F.interpolate(image, (input_resolution, input_resolution))
    image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
    image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()
    image_input = (image - image_mean[:, None, None]) / image_std[:, None, None]
    text_input = tokenizer.tokenize(
        description,
        context_length,
        truncate_text=True,
    ).cuda()
    with torch.no_grad():
        image_features = model.encode_image(image_input).float()
        text_features = model.encode_text(text_input).float()

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    # similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T
    similarity = (text_features.cpu().numpy() * image_features.cpu().numpy()).sum(1)
    return similarity

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
def visualize_long(args, dalle_module, tokenizer, data_batch, which_iter, webpage=None, description=None, tokenizer2=None, language_model=None, **kwargs):
    text_description, text, frames, visuals = data_batch['description'], data_batch['text'], data_batch['target'], data_batch['visual']

    if description is not None:
        bs = text.shape[0]
        description = [description]
        if args.fixed_language_model is not None:
            encoded_input = tokenizer2(
                description,
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
            text = mean_pooling(model_output, encoded_input['attention_mask'])
            text = text.repeat(bs, 1)
        else:
            tokenized_text = tokenizer.tokenize(
                description[0],
                args.text_seq_len,
                truncate_text=True,
            )
            text = tokenized_text.repeat(bs, 1).cuda()
        text_description = description * bs
    if isinstance(visuals, (list, tuple)):
        visuals = torch.stack(visuals, dim = 1)

    args.n_per_sample = 1  # TODO
    N_SAMPLE = min(args.n_sample, args.batch_size)
    N_PER_SAMPLE = args.n_per_sample
    N_FRAME = args.num_targets
    N_FRAME_ = args.num_targets + args.num_visuals
    N_VISUAL = args.num_visuals
    IMAGE_SIZE = args.image_size
    LOG_SAMPLE_DIR = args.log_sample_dir
    which_cvae = 'vae' if args.use_cvae is None else 'cvae'

    args.mask_predict_steps = [0]
    args.mask_predict_steps1 = 0

    generate_images = dalle_module.generate_images
    pnag_suffix = '_argmax' if args.pnag_argmax else ''
    pnag_suffix = pnag_suffix+'_dynamic' if args.pnag_dynamic else pnag_suffix
    blank_frame_nvc = torch.ones(N_PER_SAMPLE, N_VISUAL, 3, args.image_size, args.image_size).cuda()
    blank_frame_1 = torch.ones(1, 3, args.image_size, args.image_size).cuda()

    samples_img = []
    captions_img = []
    if args.use_html:
        samples_web = []
        captions_web = []
        nrow_web = []
    for j in range(N_SAMPLE):
        if args.fixed_language_model is None:
            sample_text = text[j:j+1]
            token_list = sample_text.masked_select(sample_text != 0).tolist()
            decoded_text = tokenizer.decode(token_list)
            if isinstance(decoded_text, (list, tuple)):
                decoded_text = decoded_text[0]
            text_repeat = text[j:j+1].repeat(N_PER_SAMPLE, 1)
        else:
            decoded_text = text_description[j]
            text_repeat = text[j:j+1].repeat(N_PER_SAMPLE, 1)
        
        # Sample (with visual)
        frames_recon = dalle_module.recon_images(frames[j:j+1,:N_FRAME,...])
        visual = visuals[j:j+1,...].repeat(N_PER_SAMPLE, 1, 1, 1, 1) if args.visual else None
        if args.visual:
            visual_real = visuals[j,...]
            visual_recon = dalle_module.recon_images(visual_real, which_vae=which_cvae)
            visual_prompt = visuals[j:j+1,...].repeat(N_PER_SAMPLE, 1, 1, 1, 1)  # b n c h w
            if args.rand_visual:
                visual_prompt[:,:,:,IMAGE_SIZE//2:,:] = 1
        if args.use_html:
            nrow_web += [0]
            if args.visual:
                samples_web += list(torch.split(visual_real, 1, dim = 0))
                samples_web += list(torch.split(visual_recon, 1, dim = 0))
                captions_web += [f'vc_{jj+1} [real]' for jj in range(N_VISUAL)]
                captions_web += [f'vc_{jj+1} [recon]' for jj in range(N_VISUAL)]
                nrow_web[-1] += 2 * N_VISUAL
            samples_web.append(frames[j,:N_FRAME,...])
            samples_web.append(frames_recon)
            captions_web += [decoded_text]
            captions_web += ['sequence [recon]']
            nrow_web[-1] += 2
        for mp_steps in args.mask_predict_steps:
            code_prev = None
            sample_vc = []
            tmp = []
            if args.debug:
                os.makedirs(LOG_SAMPLE_DIR / f'{which_iter}_pnag', exist_ok=True)
            if args.long_mode == 'long':
                code_prev = None
                sample_vc = []
                t_overlap = args.t_overlap
                t_extend = args.t_repeat
                for t in range(t_extend):
                    # sample_code: 1 n
                    sample_vc_, tmp_, code_ = generate_images(
                        text_repeat,
                        visual=visual,
                        erase_visual=args.rand_visual,
                        argmax=args.pnag_argmax,
                        dynamic=args.pnag_dynamic,
                        debug=args.debug,
                        mask_predict_steps=mp_steps,
                        mp_config=args.mp_config,
                        preserve=code_prev,
                        pc_mode=args.pc_mode,
                        t_overlap=0 if t == 0 else t_overlap,
                    )
                    code_prev = code_
                    if t == 0:
                        sample_vc.append(sample_vc_)
                    else:
                        sample_vc.append(sample_vc_[:,t_overlap:,...])
                    tmp += tmp_
                if args.debug:
                    tmp.insert(0, frames[j,:N_FRAME,...])
                    tmp = torch.cat(tmp, 0)
                    torchvision.utils.save_image(
                        tmp,
                        LOG_SAMPLE_DIR / f'{which_iter}_pnag' / f'{j:02d}{pnag_suffix}_T={mp_steps}.png',
                        nrow=N_FRAME,
                        normalize=True,
                        range=(0, 1)
                    )
                sample_vc = torch.cat(sample_vc, 1)
                # print(f"extended length {sample_vc.shape[1]}")
            elif args.long_mode == 'interp':
                code_prev = None
                for t in range(args.t_repeat):
                    sample_vc = []  # only valid at last level
                    code_vc = []  # used every level
                    batch_size = text_repeat.shape[0]
                    device = text_repeat.device
                    tmp = []
                    sample_prev = []
                    for tt in range(2**t):
                        preserve = dalle_module.image_token_lut['[MASK]'] + torch.zeros(batch_size, dalle_module.target_seq_len, device = device).long()
                        if t > 0:
                            preserve[:,:dalle_module.target_seq_len//2] = code_prev[:,dalle_module.target_seq_len//2*tt:dalle_module.target_seq_len//2*(tt+1)]
                        sample_vc_, tmp_, code_ = generate_images(
                            text_repeat,
                            visual=visual,
                            erase_visual=args.rand_visual,
                            argmax=args.pnag_argmax,
                            dynamic=args.pnag_dynamic,
                            debug=args.debug,
                            mask_predict_steps=mp_steps,
                            mp_config=args.mp_config,
                            preserve=preserve if t > 0 else None,
                            pc_mode=args.pc_mode,
                            t_overlap=t,
                            long_mode=args.long_mode,
                        )
                        # code_: (b t) n
                        if t == args.t_repeat - 1:
                            sample_vc.append(sample_vc_)
                        code_vc.append(rearrange(code_, '(b t) n -> b t n', t = dalle_module.num_targets))
                        if args.debug:
                            tmp = torch.cat(tmp_, 0)
                            torchvision.utils.save_image(
                                tmp,
                                LOG_SAMPLE_DIR / f'{which_iter}_pnag' / f'{j:02d}_{t}-{tt}_{pnag_suffix}_T={mp_steps}.png',
                                nrow=N_FRAME,
                                normalize=True,
                                range=(0, 1)
                            )
                    if t == 0:
                        code_prev = rearrange(code_, '(b t) n -> b (t n)', t = dalle_module.num_targets)
                    else:
                        code_prev = rearrange(torch.cat(code_vc, dim = 1), 'b t n -> b (t n)')
                sample_vc = torch.cat(sample_vc, 1)
                # print(f"extended length {sample_vc.shape[1]}")
            elif args.long_mode == 'interp_real':
                real_sample = frames[j:j+1,:N_FRAME,...]
                code_prev = dalle_module.get_image_tokens(real_sample, reshape=True, which_vae=which_cvae)
                curr_len = N_FRAME
                
                for t in range(1, args.t_repeat):
                    sample_vc = []  # only valid at last level
                    code_vc = []  # used every level
                    batch_size = text_repeat.shape[0]
                    device = text_repeat.device
                    tmp = []
                    sample_prev = []

                    if t == 0:
                        last_tt = 0
                    else:
                        last_tt = (curr_len-dalle_module.num_targets//2)//(dalle_module.num_targets//4)
                    if t == 0:
                        curr_len = 0
                    else:
                        curr_len = last_tt * dalle_module.num_targets//2 + dalle_module.num_targets - 1

                    for tt in range(last_tt+1):
                        preserve = dalle_module.image_token_lut['[MASK]'] + torch.zeros(batch_size, dalle_module.target_seq_len, device = device).long()
                        if t > 0:
                            # st()
                            preserve[:,:dalle_module.target_seq_len//2] = code_prev[:,dalle_module.target_seq_len//4*tt:dalle_module.target_seq_len//4*tt+dalle_module.target_seq_len//2]
                        sample_vc_, tmp_, code_ = generate_images(
                            text_repeat,
                            visual=visual,
                            erase_visual=args.rand_visual,
                            argmax=args.pnag_argmax,
                            dynamic=args.pnag_dynamic,
                            debug=args.debug,
                            mask_predict_steps=mp_steps,
                            mp_config=args.mp_config,
                            preserve=preserve if t > 0 else None,
                            pc_mode=args.pc_mode,
                            t_overlap=t,
                            long_mode=args.long_mode,
                        )
                        # code_: (b t) n
                        if args.debug or t == args.t_repeat - 1:
                            if t == 0:
                                sample_vc = [sample_vc_]
                            else:
                                if tt == last_tt:
                                    sample_vc.append(sample_vc_[:,:-1,...])
                                else:
                                    sample_vc.append(sample_vc_[:,:dalle_module.num_targets//2,...])
                        if tt == last_tt:
                            code_vc.append(rearrange(code_, '(b t) n -> b t n', t = dalle_module.num_targets)[:,:-1,:])
                        else:
                            code_vc.append(rearrange(code_, '(b t) n -> b t n', t = dalle_module.num_targets)[:,:dalle_module.num_targets//2,:])
                        if args.debug:
                            tmp = torch.cat(tmp_, 0)
                            torchvision.utils.save_image(
                                tmp,
                                LOG_SAMPLE_DIR / f'{which_iter}_pnag' / f'{j:02d}_{t}-{tt}_{pnag_suffix}_T={mp_steps}.png',
                                nrow=N_FRAME,
                                normalize=True,
                                range=(0, 1)
                            )

                    if t == 0:
                        code_prev = rearrange(code_, '(b t) n -> b (t n)', t = dalle_module.num_targets)
                    else:
                        code_prev = rearrange(torch.cat(code_vc, dim = 1), 'b t n -> b (t n)')

                sample_vc = torch.cat(sample_vc, 1)
                # print(f"extended length {sample_vc.shape[1]}")
            # ==========================================================================================

            if args.save_codebook:
                sam_code, sam_embd = dalle_module.get_codebook_emb(sample_vc, which_vae=which_cvae)
                np.save(LOG_SAMPLE_DIR / f'{which_iter}_{j:02d}{pnag_suffix}_T={mp_steps}_code.npy', sam_code.cpu().numpy())
                np.save(LOG_SAMPLE_DIR / f'{which_iter}_{j:02d}{pnag_suffix}_T={mp_steps}_embed.npy', sam_embd.cpu().numpy())

                sam_down = F.interpolate(rearrange(sample_vc, 'b t c h w -> (b t) c h w'), size=(32,32))
                sam_down = rearrange(sam_down, '(b t) c h w -> b t c (h w)', b = sample_vc.shape[0])
                np.save(LOG_SAMPLE_DIR / f'{which_iter}_{j:02d}{pnag_suffix}_T={mp_steps}_down.npy', sam_down.cpu().numpy())

            if args.use_html:
                nrow_web += [0]
                if args.visual:
                    samples_web += list(torch.split(visual_prompt[0,...], 1, dim = 0))
                    captions_web += [f'vc_{jj+1} [prompt]' for jj in range(N_VISUAL)]
                    nrow_web[-1] += N_VISUAL
                samples_web += list(torch.split(sample_vc, 1, dim = 0))
                captions_web += [f'sample {jj+1} [T={mp_steps}]' for jj in range(N_PER_SAMPLE)]
                nrow_web[-1] += N_PER_SAMPLE
        mp_steps = args.mask_predict_steps1

        if args.visual:
            j2 = (j+1) % frames.shape[0]
            sample_cf, tmp = generate_images(
                text_repeat,
                visual=visuals[j2:j2+1,...].repeat(N_PER_SAMPLE, 1, 1, 1, 1),
                erase_visual=args.rand_visual,
                argmax=args.pnag_argmax,
                dynamic=args.pnag_dynamic,
                debug=args.debug,
                mask_predict_steps=mp_steps,
            )
            visual_prompt = visuals[j2:j2+1,...].repeat(N_PER_SAMPLE, 1, 1, 1, 1)  # b n c h w
            if args.rand_visual:
                visual_prompt[:,:,:,IMAGE_SIZE//2:,:] = 1
            samples_img.append(torch.cat((visual_prompt, sample_cf), 1).reshape(N_PER_SAMPLE*N_FRAME_, *frames.shape[2:5]))
            if args.use_html:
                samples_web += list(torch.split(visual_prompt[0,...], 1, dim = 0))
                samples_web += list(torch.split(sample_cf, 1, dim = 0))
                captions_web += [f'cf_{jj+1} [prompt]' for jj in range(N_VISUAL)]
                captions_web += [f'sample {jj+1}' for jj in range(N_PER_SAMPLE)]
                nrow_web += [N_VISUAL + N_PER_SAMPLE]
            if args.debug:
                # tmp.insert(0, frames[j,:N_FRAME,...])
                tmp = torch.cat(tmp, 0)
                torchvision.utils.save_image(
                    tmp,
                    LOG_SAMPLE_DIR / f'{which_iter}_pnag' / f'cf_{j:02d}{pnag_suffix}.png',
                    nrow=N_FRAME,
                    normalize=True,
                    range=(0, 1)
                )

        if args.visual and not args.fullvc:
            sample_free, tmp = generate_images(
                text_repeat,
                visual=None,
                erase_visual=args.rand_visual,
                argmax=args.pnag_argmax,
                dynamic=args.pnag_dynamic,
                debug=args.debug,
                mask_predict_steps=mp_steps,
                preserve=None,
            )
            samples_img.append(torch.cat((blank_frame_nvc, sample_free), 1).reshape(N_PER_SAMPLE*N_FRAME_, *frames.shape[2:5]))
            if args.use_html:
                samples_web += [blank_frame_1] * N_VISUAL
                samples_web += list(torch.split(sample_free, 1, dim = 0))
                captions_web += [f'null [prompt]' for jj in range(N_VISUAL)]
                captions_web += [f'sample {jj+1}' for jj in range(N_PER_SAMPLE)]
                nrow_web += [N_VISUAL + N_PER_SAMPLE]
            if args.debug:
                tmp = torch.cat(tmp, 0)
                torchvision.utils.save_image(
                    tmp,
                    LOG_SAMPLE_DIR / f'{which_iter}_pnag' / f'free_{j:02d}{pnag_suffix}.png',
                    nrow=N_FRAME,
                    normalize=True,
                    range=(0, 1)
                )

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

@torch.no_grad()
def visualize(args, dalle_module, tokenizer, data_batch, which_iter, webpage=None, description=None, tokenizer2=None, language_model=None, **kwargs):
    text_description, text, frames, visuals = data_batch['description'], data_batch['text'], data_batch['target'], data_batch['visual']
    text_neg, visuals_neg = data_batch['text_neg'], data_batch['visual_neg']

    if description is not None:
        bs = text.shape[0]
        description = [description]
        if args.fixed_language_model is not None:
            encoded_input = tokenizer2(
                description,
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
            text = mean_pooling(model_output, encoded_input['attention_mask'])
            text = text.repeat(bs, 1)
        else:
            tokenized_text = tokenizer.tokenize(
                description[0],
                args.text_seq_len,
                truncate_text=True,
            )
            text = tokenized_text.repeat(bs, 1).cuda()
        text_description = description * bs
    if isinstance(visuals, (list, tuple)):
        visuals = torch.stack(visuals, dim = 1)

    N_SAMPLE = min(args.n_sample, args.batch_size)
    N_PER_SAMPLE = args.n_per_sample
    N_FRAME = args.num_targets
    N_FRAME_ = args.num_targets + args.num_visuals
    N_VISUAL = args.num_visuals
    IMAGE_SIZE = args.image_size
    LOG_SAMPLE_DIR = args.log_sample_dir
    which_cvae = 'vae' if args.use_cvae is None else 'cvae'

    # TODO: this is previously hardcoded
    args.mask_predict_steps = [args.mp_config['T']]
    args.mask_predict_steps1 = args.mp_config['T']

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
    pnag_suffix = pnag_suffix+'_dynamic' if args.pnag_dynamic else pnag_suffix
    blank_frame_nvc = torch.ones(N_PER_SAMPLE, N_VISUAL, 3, args.image_size, args.image_size).cuda()
    blank_frame_1 = torch.ones(1, 3, args.image_size, args.image_size).cuda()

    samples_img = []
    captions_img = []
    if args.use_html:
        samples_web = []
        captions_web = []
        nrow_web = []
    for j in range(N_SAMPLE):
        if args.fixed_language_model is None:
            sample_text = text[j:j+1]
            token_list = sample_text.masked_select(sample_text != 0).tolist()
            decoded_text = tokenizer.decode(token_list)
            if isinstance(decoded_text, (list, tuple)):
                decoded_text = decoded_text[0]
            text_repeat = text[j:j+1].repeat(N_PER_SAMPLE, 1)
        else:
            decoded_text = text_description[j]
            text_repeat = text[j:j+1].repeat(N_PER_SAMPLE, 1)
        
        # Sample (with visual)
        face_mode = None
        frames_recon = dalle_module.recon_images(frames[j:j+1,:N_FRAME,...])
        visual = visuals[j:j+1,...].repeat(N_PER_SAMPLE, 1, 1, 1, 1) if args.visual else None
        if args.visual:
            visual_real = visuals[j,...]
            visual_recon = dalle_module.recon_images(visual_real, which_vae=which_cvae)
            samples_img.append(torch.cat((visual_real, frames[j,:N_FRAME,...]), 0))  # real video sequence
            samples_img.append(torch.cat((visual_recon, frames_recon), 0))
            visual_prompt = visuals[j:j+1,...].clone().repeat(N_PER_SAMPLE, 1, 1, 1, 1)  # b n c h w
            if args.rand_visual:
                visual_prompt[:,:,:,IMAGE_SIZE//2:,:] = 1
            if args.vc_mode == 'face_8x8':
                block_size = 16
                visual_prompt_ = torch.ones_like(visual_prompt)
                if random.random() < 0.5:
                    face_mode = 'eyes_nose'
                    visual_prompt_[:,:,:,2*block_size:5*block_size,1*block_size:7*block_size] = visual_prompt[:,:,:,2*block_size:5*block_size,1*block_size:7*block_size]
                else:
                    face_mode = 'mouth'
                    visual_prompt_[:,:,:,5*block_size:7*block_size,2*block_size:6*block_size] = visual_prompt[:,:,:,5*block_size:7*block_size,2*block_size:6*block_size]
                visual_prompt = visual_prompt_
            elif args.vc_mode == 'face3_8x8':  # for mug evaluation
                block_size = 16
                visual_prompt_ = torch.ones_like(visual_prompt)
                face_mode = 'center'
                visual_prompt_[:,:,:,2*block_size:6*block_size,2*block_size:6*block_size] = visual_prompt[:,:,:,2*block_size:6*block_size,2*block_size:6*block_size]
                visual_prompt = visual_prompt_
            elif args.vc_mode == 'face2_8x8':
                block_size = 16
                visual_prompt_ = torch.ones_like(visual_prompt)
                visual_prompt_[:,0,...] = visual_prompt[:,0,...]
                face_mode = 'face2'
                visual_prompt_[:,1:,:,2*block_size:6*block_size,2*block_size:6*block_size] = visual_prompt[:,1:,:,2*block_size:6*block_size,2*block_size:6*block_size]
                visual_prompt = visual_prompt_
            elif args.vc_mode == 'mask2_8x8':
                block_size = 16
                visual_prompt_ = torch.ones_like(visual_prompt)
                visual_prompt_[:,:,:,1*block_size:7*block_size,1*block_size:7*block_size] = visual_prompt[:,:,:,1*block_size:7*block_size,1*block_size:7*block_size]
                visual_prompt = visual_prompt_
                face_mode = 'mask2'
            elif args.vc_mode == 'mask_8x8':
                block_size = 16
                visual_prompt_ = torch.ones_like(visual_prompt)
                visual_prompt_[:,:,:,1*block_size:7*block_size,1*block_size:7*block_size] = visual_prompt[:,:,:,1*block_size:7*block_size,1*block_size:7*block_size]
                visual_prompt = visual_prompt_
                face_mode = 'mask'
            elif args.vc_mode == 'shape_4x4':
                block_size = 16
                visual_prompt[:,:,:,1*block_size:3*block_size,1*block_size:3*block_size] = 1
                face_mode = 'shape'
        else:
            samples_img.append(frames[j,:N_FRAME,...])  # real video sequence
            samples_img.append(frames_recon)
        captions_img.append(f'{j+1}. {decoded_text}')
        if args.use_html:
            nrow_web += [0]
            if args.visual:
                samples_web += list(torch.split(visual_real, 1, dim = 0))
                samples_web += list(torch.split(visual_recon, 1, dim = 0))
                captions_web += [f'vc_{jj+1} [real]' for jj in range(N_VISUAL)]
                captions_web += [f'vc_{jj+1} [recon]' for jj in range(N_VISUAL)]
                nrow_web[-1] += 2 * N_VISUAL
            samples_web.append(frames[j,:N_FRAME,...])
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
                mp_config=args.mp_config,
                face_mode=face_mode,
            )
            if args.visual:
                samples_img.append(torch.cat((visual_prompt, sample_vc), 1).reshape(N_PER_SAMPLE*N_FRAME_, *frames.shape[2:5]))
            else:
                samples_img.append(sample_vc.reshape(N_PER_SAMPLE*N_FRAME, *frames.shape[2:5]))
            if args.use_html:
                nrow_web += [0]
                if args.visual:
                    samples_web += list(torch.split(visual_prompt[0,...], 1, dim = 0))
                    captions_web += [f'vc_{jj+1} [prompt]' for jj in range(N_VISUAL)]
                    nrow_web[-1] += N_VISUAL
                samples_web += list(torch.split(sample_vc, 1, dim = 0))
                captions_web += [f'sample {jj+1} [T={mp_steps}]' for jj in range(N_PER_SAMPLE)]
                nrow_web[-1] += N_PER_SAMPLE
            if args.debug:
                os.makedirs(LOG_SAMPLE_DIR / f'{which_iter}_pnag', exist_ok=True)
                tmp.insert(0, frames[j,:N_FRAME,...])
                tmp = torch.cat(tmp, 0)
                torchvision.utils.save_image(
                    tmp,
                    LOG_SAMPLE_DIR / f'{which_iter}_pnag' / f'{j:02d}{pnag_suffix}_T={mp_steps}.png',
                    nrow=N_FRAME,
                    normalize=True,
                    range=(0, 1)
                )
        mp_steps = args.mask_predict_steps1

        if args.visual and args.test_mode is None:
            j2 = (j+1) % frames.shape[0]
            visual_prompt = visuals[j2:j2+1,...].clone().repeat(N_PER_SAMPLE, 1, 1, 1, 1)  # b n c h w
            if args.rand_visual:
                visual_prompt[:,:,:,IMAGE_SIZE//2:,:] = 1
            if args.vc_mode == 'face_8x8':
                block_size = 16
                visual_prompt_ = torch.ones_like(visual_prompt)
                if random.random() < 0.5:
                    face_mode = 'eyes_nose'
                    visual_prompt_[:,:,:,2*block_size:5*block_size,1*block_size:7*block_size] = visual_prompt[:,:,:,2*block_size:5*block_size,1*block_size:7*block_size]
                else:
                    face_mode = 'mouth'
                    visual_prompt_[:,:,:,5*block_size:7*block_size,2*block_size:6*block_size] = visual_prompt[:,:,:,5*block_size:7*block_size,2*block_size:6*block_size]
                visual_prompt = visual_prompt_
                visual_cf = visuals[j2:j2+1,...]
                visual_cf = visual_cf.repeat(N_PER_SAMPLE, 1, 1, 1, 1)
            elif args.vc_mode == 'face2_8x8':
                block_size = 16
                visual_prompt_ = torch.ones_like(visual_prompt)
                visual_prompt_[:,0,...] = visual_prompt[:,0,...]
                face_mode = 'face2'
                visual_prompt1 = visuals[j:j+1,...].repeat(N_PER_SAMPLE, 1, 1, 1, 1)
                visual_prompt_[:,1:,:,2*block_size:6*block_size,2*block_size:6*block_size] = visual_prompt1[:,1:,:,2*block_size:6*block_size,2*block_size:6*block_size]
                visual_prompt = visual_prompt_
                visual_cf = visuals[j:j+1,...].clone()  # !!!
                visual_cf[:,0,...] = visuals[j2:j2+1,0,...]
                visual_cf = visual_cf.repeat(N_PER_SAMPLE, 1, 1, 1, 1)
            elif args.vc_mode == 'mask2_8x8':
                block_size = 16
                visual_prompt_ = torch.ones_like(visual_prompt)
                visual_prompt1 = visuals[j:j+1,...].repeat(N_PER_SAMPLE, 1, 1, 1, 1)
                visual_prompt_[:,0,:,1*block_size:7*block_size,1*block_size:7*block_size] = visual_prompt1[:,0,:,1*block_size:7*block_size,1*block_size:7*block_size]
                visual_prompt_[:,1,:,1*block_size:7*block_size,1*block_size:7*block_size] = visual_prompt[:,1,:,1*block_size:7*block_size,1*block_size:7*block_size]
                visual_prompt = visual_prompt_
                visual_cf = visuals[j:j+1,...].clone()  # !!!
                visual_cf[:,1,...] = visuals[j2:j2+1,1,...]
                visual_cf = visual_cf.repeat(N_PER_SAMPLE, 1, 1, 1, 1)
                face_mode = 'mask2'
            elif args.vc_mode == 'mask_8x8':
                block_size = 16
                visual_prompt_ = torch.ones_like(visual_prompt)
                visual_prompt_[:,:,:,1*block_size:7*block_size,1*block_size:7*block_size] = visual_prompt[:,:,:,1*block_size:7*block_size,1*block_size:7*block_size]
                visual_prompt = visual_prompt_
                visual_cf = visuals[j2:j2+1,...].repeat(N_PER_SAMPLE, 1, 1, 1, 1)
                face_mode = 'mask'
            elif args.vc_mode == 'shape_4x4':
                block_size = 16
                visual_prompt[:,:,:,1*block_size:3*block_size,1*block_size:3*block_size] = 1
                visual_cf = visuals[j2:j2+1,...].repeat(N_PER_SAMPLE, 1, 1, 1, 1)
                face_mode = 'shape'
            else:
                visual_cf = visuals[j2:j2+1,...].repeat(N_PER_SAMPLE, 1, 1, 1, 1)
            sample_cf, tmp, _ = generate_images(
                text_repeat,
                visual=visual_cf,
                erase_visual=args.rand_visual,
                argmax=args.pnag_argmax,
                dynamic=args.pnag_dynamic,
                debug=args.debug,
                mask_predict_steps=mp_steps,
                mp_config=args.mp_config,
                face_mode=face_mode,
            )
            samples_img.append(torch.cat((visual_prompt, sample_cf), 1).reshape(N_PER_SAMPLE*N_FRAME_, *frames.shape[2:5]))
            if args.use_html:
                samples_web += list(torch.split(visual_prompt[0,...], 1, dim = 0))
                samples_web += list(torch.split(sample_cf, 1, dim = 0))
                captions_web += [f'cf_{jj+1} [prompt]' for jj in range(N_VISUAL)]
                captions_web += [f'sample {jj+1}' for jj in range(N_PER_SAMPLE)]
                nrow_web += [N_VISUAL + N_PER_SAMPLE]
            if args.debug:
                # tmp.insert(0, frames[j,:N_FRAME,...])
                tmp = torch.cat(tmp, 0)
                torchvision.utils.save_image(
                    tmp,
                    LOG_SAMPLE_DIR / f'{which_iter}_pnag' / f'cf_{j:02d}{pnag_suffix}.png',
                    nrow=N_FRAME,
                    normalize=True,
                    range=(0, 1)
                )

        ## test_mode: shapes
        # =================== for shapes ======================
        if args.visual and args.test_mode == 'shapes':
            # j2 = (j+1) % frames.shape[0]
            # for kk, vc in enumerate(['color', 'shape', 'bg']):
            for kk in range(3):
                visual_prompt = visuals[j:j+1,...].clone()
                visual_prompt[:,kk,...] = visuals_neg[j:j+1,kk,...]
                visual_prompt = visual_prompt.repeat(N_PER_SAMPLE, 1, 1, 1, 1)
                sample_cf, tmp, _ = generate_images(
                    text_repeat,
                    visual=visual_prompt,
                    erase_visual=args.rand_visual,
                    argmax=args.pnag_argmax,
                    dynamic=args.pnag_dynamic,
                    debug=args.debug,
                    mask_predict_steps=mp_steps,
                    mp_config=args.mp_config,
                )
                if args.rand_visual:
                    visual_prompt[:,:,:,IMAGE_SIZE//2:,:] = 1
                samples_img.append(torch.cat((visual_prompt, sample_cf), 1).reshape(N_PER_SAMPLE*N_FRAME_, *frames.shape[2:5]))
                if args.use_html:
                    samples_web += list(torch.split(visual_prompt[0,...], 1, dim = 0))
                    samples_web += list(torch.split(sample_cf, 1, dim = 0))
                    captions_web += [f'cf-{kk+1}_{jj+1} [prompt]' for jj in range(N_VISUAL)]
                    captions_web += [f'sample {jj+1} [T={mp_steps}]' for jj in range(N_PER_SAMPLE)]
                    nrow_web += [N_VISUAL + N_PER_SAMPLE]
        # ========================================================

    samples_img = torch.cat(samples_img)
    torchvision.utils.save_image(
        samples_img,
        LOG_SAMPLE_DIR / f'{which_iter}.png',
        nrow=N_FRAME_,
        normalize=True,
        range=(0, 1)
    )

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
            video_format=args.video_format,
        )

@torch.no_grad()
def extend_video(video, num=2):
    # video (n, t, c, h, w)
    video_ = [video]
    video_flipped = torch.flip(video, [1])
    for n in range(1, num):
        if n % 2 == 0:
            video_.append(video[:,1:,...])
        else:
            video_.append(video_flipped[:,1:,...])
    video_ = torch.cat(video_, dim = 1)
    return video_

@torch.no_grad()
def evaluate(args, dalle_module, tokenizer, tokenizer2, language_model, dl_iter, metrics=['fvd', 'prd']):
    # NOTE: I used conda env py38, with cuda 11.2 (cuda 11.1 + cudnn-11.2-linux-x64-v8.1.0.77)
    LOG_DIR = Path(args.log_root) / args.name
    LOG_WEB_DIR = LOG_DIR / 'web'
    USE_HTML = True
    
    if USE_HTML:
        webpage = utils_html.initialize_webpage(LOG_WEB_DIR, 'DALLE: ' + args.name + ' FVD', reverse=False)
    else:
        webpage = None
    N_FRAME = args.num_targets
    sample_counter = 0
    blank_image = torch.ones(1, 3, 3, 3)
    mp_steps = 0

    import tensorflow.compat.v1 as tf
    from frechet_video_distance import frechet_video_distance as fvd
    import precision_recall_distributions.prd_score as prd

    OUTPUT_DIR = args.log_metric_dir
    VIDEO_LENGTH = 15 if args.num_targets < 16 else 16
    TOTAL_NUM = args.eval_num
    IMAGE_SIZE = args.image_size

    if args.dm:
        from functools import partial
        generate_images = partial(
            dalle_module.generate_images_debug,
            sample_mode=args.dm_sample_mode,
            sample_with_confidence=args.dm_sample_with_confidence,
        )
    else:
        generate_images = dalle_module.generate_images_test

    real_embs = []
    fake_embs = []
    with tf.Graph().as_default():
        vid = tf.placeholder(tf.float32, [args.batch_size,VIDEO_LENGTH,IMAGE_SIZE,IMAGE_SIZE,3],name='vid')
        emb_data1 = tf.placeholder(tf.float32, [None,400],name='emb_data1')
        emb_data2 = tf.placeholder(tf.float32, [None,400],name='emb_data2')

        emb = fvd.create_id3_embedding(fvd.preprocess(vid, (224,224)))
        result = fvd.calculate_fvd(emb_data1, emb_data2)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())

            for _ in tqdm(range(TOTAL_NUM//args.batch_size)):
                text_neg, visuals_neg = None, None
                if args.negvc:
                    text, frames, visuals, visuals_neg, text_neg = next(dl_iter)
                    visuals_neg, text_neg = map(lambda t: t.cuda(), (visuals_neg, text_neg))
                else:
                    text, frames, visuals = next(dl_iter)
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
                        text = mean_pooling(model_output, encoded_input['attention_mask'])
                else:
                    text = text.cuda()
                    text_description = []
                    for j in range(text.shape[0]):
                        sample_text = text[j:j+1]
                        token_list = sample_text.masked_select(sample_text != 0).tolist()
                        decoded_text = tokenizer.decode(token_list)
                        if isinstance(decoded_text, (list, tuple)):
                            decoded_text = decoded_text[0]
                        text_description += [decoded_text]

                if args.vc_mode == 'face_8x8':
                    face_mode = 'eyes_nose' if random.random() < 0.5 else 'mouth'
                elif args.vc_mode == 'face2_8x8':
                    face_mode = 'face2'
                elif args.vc_mode == 'mask_8x8':
                    face_mode = 'mask'
                elif args.vc_mode == 'mask2_8x8':
                    face_mode = 'mask2'
                else:
                    face_mode = None

                sample_vc, _, _ = generate_images(
                    text,
                    visual=visuals if args.visual else None,
                    erase_visual=False,
                    argmax=args.pnag_argmax,
                    dynamic=args.pnag_dynamic,
                    debug=False,
                    mask_predict_steps=mp_steps,
                    vc_mode=args.vc_mode,
                    face_mode=face_mode,
                    mp_config=args.mp_config,
                )
                real_videos = frames.detach().cpu()  # n, t, 3, h, w
                fake_videos = sample_vc.detach().cpu()

                if USE_HTML and sample_counter < args.iters:
                    samples_web = []
                    captions_web = []
                    nrow_web = []
                    # samples_web.append(real_videos[0,:N_FRAME,...])
                    # captions_web += [text_description[0]]
                    # samples_web.append(fake_videos[0,:N_FRAME,...])
                    # captions_web += [f'sample {0}']
                    # nrow_web += [2]
                    samples_web.append(real_videos[0,:N_FRAME,...])
                    captions_web += [f'real {0}']
                    samples_web.append(fake_videos[0,:N_FRAME,...])
                    captions_web += [f'T={mp_steps}']
                    samples_web.append(blank_image)
                    captions_web += [text_description[0]]
                    nrow_web += [3]

                    sample_counter += 1
                    webpage.add_header(f'iteration {sample_counter}')
                    utils_html.save_grid(
                        webpage=webpage,
                        tensor=samples_web,
                        caption=captions_web,
                        name=f'sample_{sample_counter}',
                        nrow=nrow_web,
                        width=min(IMAGE_SIZE, 256),
                    )

                if real_videos.shape[1] < VIDEO_LENGTH:
                    n_frame = real_videos.shape[1]
                    num = int(np.ceil((VIDEO_LENGTH-1) / (n_frame-1)))
                    real_videos = extend_video(real_videos, num)
                    fake_videos = extend_video(fake_videos, num)
                real_videos = real_videos[:,:VIDEO_LENGTH,...]
                fake_videos = fake_videos[:,:VIDEO_LENGTH,...]
                real_videos = (real_videos*255).permute(0, 1, 3, 4, 2).numpy()  # (n, t, h, w, 3)
                fake_videos = (fake_videos*255).permute(0, 1, 3, 4, 2).numpy()  # (n, t, h, w, 3)
                real_emb = sess.run(emb, feed_dict={vid:real_videos})
                fake_emb = sess.run(emb, feed_dict={vid:fake_videos})
                real_embs.append(real_emb)
                fake_embs.append(fake_emb)

            real_embs = np.concatenate(real_embs, axis=0)
            fake_embs = np.concatenate(fake_embs, axis=0)

            np.save(str(OUTPUT_DIR / 'real_embs.npy'), real_embs)
            np.save(str(OUTPUT_DIR / 'fake_embs.npy'), fake_embs)

            # FVD
            score = sess.run(result, feed_dict={emb_data1:real_embs, emb_data2:fake_embs})

        print(f"FVD is: {score}")
        with open(OUTPUT_DIR / 'fvd_score.txt', 'w') as f:
            f.write(f"{score}")

        # PRD
        prd_data = prd.compute_prd_from_embedding(real_embs, fake_embs)
        with open(OUTPUT_DIR / 'prd_data.pkl', 'wb') as f:
            pickle.dump(
                prd_data,
                f
            )
        f_beta, f_beta_inv = prd.prd_to_max_f_beta_pair(prd_data[0], prd_data[1])
        print(f"f_beta: {f_beta}, f_beta_inv: {f_beta_inv}")
        with open(OUTPUT_DIR / 'prd_score.txt', 'w') as f:
            f.write(f"{f_beta}, {f_beta_inv}")
        print(f"FVD is: {score}")

@torch.no_grad()
def evaluate_intra(args, dalle_module, tokenizer, tokenizer2, language_model, dl_iter, metrics=['fvd', 'prd']):
    mp_steps = args.mask_predict_steps1

    import tensorflow.compat.v1 as tf
    from frechet_video_distance import frechet_video_distance as fvd
    import precision_recall_distributions.prd_score as prd

    OUTPUT_DIR = args.log_metric_dir
    VIDEO_LENGTH = 15 if args.num_targets < 16 else 16
    TOTAL_NUM = args.eval_num
    IMAGE_SIZE = args.image_size

    from dalle_pytorch.tokenizer import SimpleTokenizer
    clipper = torch.jit.load("pretrained/ViT-B-32.pt").cuda().eval()
    clip_tokenizer = SimpleTokenizer()

    # cat1 = [ 1, 24, 32,  3, 19, 11, 36, 39, 12,  8, 18, 22, 34, 33, 29, 30]
    if args.attr_mode == 'cat2':  # TODO
        args.cat1 = [20, 39, 4, 15, 13]

    generate_images = dalle_module.generate_images

    real_embs = {c: [] for c in args.cat1}
    fake_embs = {c: [] for c in args.cat1}
    clip_scores = {c: [] for c in args.cat1}

    with tf.Graph().as_default():
        vid = tf.placeholder(tf.float32, [args.batch_size,VIDEO_LENGTH,IMAGE_SIZE,IMAGE_SIZE,3],name='vid')
        emb_data1 = tf.placeholder(tf.float32, [None,400],name='emb_data1')
        emb_data2 = tf.placeholder(tf.float32, [None,400],name='emb_data2')

        emb = fvd.create_id3_embedding(fvd.preprocess(vid, (224,224)))
        result = fvd.calculate_fvd(emb_data1, emb_data2)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        visuals = None

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())

            for _ in tqdm(range(TOTAL_NUM//args.batch_size)):
                frames_, text_ = next(dl_iter)
                frames_ = frames_.cuda()

                for cind, ccc in enumerate(args.cat1):
                    frames = frames_[:,cind,...]
                    if args.fixed_language_model is not None:
                        text = text_[cind]
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
                            text = mean_pooling(model_output, encoded_input['attention_mask'])
                    else:
                        text = text_[:,cind,...]
                        text = text.cuda()
                        text_description = []
                        for j in range(text.shape[0]):
                            sample_text = text[j:j+1]
                            token_list = sample_text.masked_select(sample_text != 0).tolist()
                            decoded_text = tokenizer.decode(token_list)
                            if isinstance(decoded_text, (list, tuple)):
                                decoded_text = decoded_text[0]
                            text_description += [decoded_text]

                    if args.vc_mode == 'face_8x8':
                        face_mode = 'eyes_nose' if random.random() < 0.5 else 'mouth'
                    elif args.vc_mode == 'face2_8x8':
                        face_mode = 'face2'
                    elif args.vc_mode == 'mask_8x8':
                        face_mode = 'mask'
                    elif args.vc_mode == 'mask2_8x8':
                        face_mode = 'mask2'
                    else:
                        face_mode = None

                    sample_vc, _, _ = generate_images(
                        text,
                        visual=visuals if args.visual else None,
                        erase_visual=False,
                        argmax=args.pnag_argmax,
                        dynamic=args.pnag_dynamic,
                        debug=False,
                        mask_predict_steps=mp_steps,
                        vc_mode=args.vc_mode,
                        face_mode=face_mode,
                        mp_config=args.mp_config,
                    )
                    real_videos = frames.detach().cpu()  # n, t, 3, h, w
                    fake_videos = sample_vc.detach().cpu()

                    # clip score of first frame
                    clip_score = clip_similarity(clipper, clip_tokenizer, sample_vc[:,0,...], text_description)
                    clip_scores[ccc].append(clip_score)

                    if real_videos.shape[1] < VIDEO_LENGTH:
                        n_frame = real_videos.shape[1]
                        num = int(np.ceil((VIDEO_LENGTH-1) / (n_frame-1)))
                        real_videos = extend_video(real_videos, num)
                        fake_videos = extend_video(fake_videos, num)
                    real_videos = real_videos[:,:VIDEO_LENGTH,...]
                    fake_videos = fake_videos[:,:VIDEO_LENGTH,...]
                    real_videos = (real_videos*255).permute(0, 1, 3, 4, 2).numpy()  # (n, t, h, w, 3)
                    fake_videos = (fake_videos*255).permute(0, 1, 3, 4, 2).numpy()  # (n, t, h, w, 3)
                    # real_video_list.append(real_videos)
                    # fake_video_list.append(fake_videos)
                    real_emb = sess.run(emb, feed_dict={vid:real_videos})
                    fake_emb = sess.run(emb, feed_dict={vid:fake_videos})
                    real_embs[ccc].append(real_emb)
                    fake_embs[ccc].append(fake_emb)
            
            with open(OUTPUT_DIR / 'embs.pkl', 'wb') as f:
                pickle.dump({'real': real_embs, 'fake': fake_embs}, f)

            scores = []
            f_betas = []
            f_betas_inv = []
            real_embs_all = []
            fake_embs_all = []
            clip_avgs = []
            for cind, ccc in enumerate(args.cat1):
                real_embs1 = np.concatenate(real_embs[ccc], axis=0)
                fake_embs1 = np.concatenate(fake_embs[ccc], axis=0)
                real_embs_all += real_embs[ccc]
                fake_embs_all += fake_embs[ccc]

                # FID
                score = sess.run(result, feed_dict={emb_data1:real_embs1, emb_data2:fake_embs1})
                scores.append(score)

                clip_avgs.append(np.array(clip_scores[ccc]).mean())
            
            print(f"per attr CLIP score: {clip_avgs}")
            with open(OUTPUT_DIR / 'clip_intra.txt', 'w') as f:
                f.write(f"{clip_avgs}")

            print(f"mean Intra-FVD is: {np.array(scores).mean()}")
            with open(OUTPUT_DIR / 'intra_fvd_score.txt', 'w') as f:
                f.write(f"{scores}")

            real_embs_all = np.concatenate(real_embs_all, axis=0)
            fake_embs_all = np.concatenate(fake_embs_all, axis=0)
            score = sess.run(result, feed_dict={emb_data1:real_embs_all, emb_data2:fake_embs_all})
            print(f"uncond FVD is: {score}")
            with open(OUTPUT_DIR / 'fvd_score.txt', 'w') as f:
                f.write(f"{score}")

            # PRD
            prd_data = prd.compute_prd_from_embedding(real_embs_all, fake_embs_all)
            f_beta, f_beta_inv = prd.prd_to_max_f_beta_pair(prd_data[0], prd_data[1])
            print(f"f_beta: {f_beta}, f_beta_inv: {f_beta_inv}")
            with open(OUTPUT_DIR / 'prd_score.txt', 'w') as f:
                f.write(f"{f_beta}, {f_beta_inv}")

            print(f"mean Intra-FVD is: {np.array(scores).mean()}")
            with open(OUTPUT_DIR / 'intra_fvd_score.txt', 'w') as f:
                f.write(f"{scores}")

def normalize(x, eps=1e-10):
    return x * torch.rsqrt(torch.sum(x**2, dim=1, keepdim=True) + eps)

@torch.no_grad()
def evaluate_clip(args, dalle_module, tokenizer, tokenizer2, language_model, dl_iter, upper_bound=False):
    # text_description, text, frames, visuals = data_batch['description'], data_batch['text'], data_batch['target'], data_batch['visual']
    # text_neg, visuals_neg = data_batch['text_neg'], data_batch['visual_neg']
    mp_steps = args.mask_predict_steps1

    # import pcfg

    assert not args.visual

    OUTPUT_DIR = args.log_metric_dir
    VIDEO_LENGTH = 15
    TOTAL_NUM = args.eval_num

    from dalle_pytorch.tokenizer import SimpleTokenizer
    clipper = torch.jit.load("pretrained/ViT-B-32.pt").cuda().eval()
    clip_tokenizer = SimpleTokenizer()

    # cnt = 0

    batch_size = 1  # must be 1
    results = []

    for _ in tqdm(range(TOTAL_NUM//batch_size)):
        if args.negvc:
            text, frames, visuals, visuals_neg, text_neg = next(dl_iter)
            visuals_neg, text_neg = map(lambda t: t.cuda(), (visuals_neg, text_neg))
        else:
            text, frames, visuals = next(dl_iter)
        # frames, visuals = map(lambda t: t.cuda(), (frames, visuals))

        j = 0
        # visuals = visuals[j:j+1,...]

        if args.fixed_language_model is not None:
            text_description = [text[j]]
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
                text = mean_pooling(model_output, encoded_input['attention_mask'])
        else:
            text = text[j:j+1,...].cuda()
            sample_text = text
            token_list = sample_text.masked_select(sample_text != 0).tolist()
            decoded_text = tokenizer.decode(token_list)
            if isinstance(decoded_text, (list, tuple)):
                decoded_text = decoded_text[0]
            text_description = [decoded_text]

        if upper_bound:
            images = frames[j].cuda()
        else:
            if args.vc_mode == 'face_8x8':
                face_mode = 'eyes_nose' if random.random() < 0.5 else 'mouth'
            elif args.vc_mode == 'face2_8x8':
                face_mode = 'face2'
            elif args.vc_mode == 'mask_8x8':
                face_mode = 'mask'
            elif args.vc_mode == 'mask2_8x8':
                face_mode = 'mask2'
            else:
                face_mode = None
            sample_vc, _, _ = dalle_module.generate_images_debug(
                text,
                visual=None,
                erase_visual=False,
                argmax=args.pnag_argmax,
                dynamic=args.pnag_dynamic,
                debug=False,
                mask_predict_steps=mp_steps,
                vc_mode=args.vc_mode,
                face_mode=face_mode,
                mp_config=args.mp_config,
            )
            images = sample_vc.squeeze(0)

        scores = clip_similarity(clipper, clip_tokenizer, images, text_description)
        results.append((text_description[0], scores))
        # cnt += batch_size

    suffix = '_real' if upper_bound else ''
    with open(OUTPUT_DIR / f'clip{suffix}_data.pkl', 'wb') as f:
        pickle.dump(
            results,
            f
        )

    scores = np.array([res[1] for res in results])
    scores = scores.max(axis=1)
    print(f"CLIP score: {scores.mean()}")
    with open(OUTPUT_DIR / f'clip{suffix}_score.txt', 'w') as f:
        f.write(f"{scores.mean()}, {scores.std()}")

def main():
    # argument parsing

    parser = argparse.ArgumentParser()

    parser.add_argument('--vae_path', type=str, help='path to your trained discrete VAE')
    parser.add_argument('--cvae_path', type=str, help='path to your trained discrete VAE')
    parser.add_argument('--dalle_path2', type=str, default=None)
    parser.add_argument('--dalle_path', type=str, default=None, help='path to your partially trained DALL-E')
    parser.add_argument('--which_vae', type=str, default='vqgan1024')
    parser.add_argument('--transformer_path', type=str, default=None)
    parser.add_argument('--image_text_folder', type=str, required=True, help='path to your folder of images and text for learning the DALL-E')
    parser.add_argument('--dataset', type=str, default='video_text')
    parser.add_argument('--dataset_keys', type=str, default=None)
    parser.add_argument('--dataset_cache', type=str, default=None)
    parser.add_argument('--video_only', action = 'store_true')
    parser.add_argument('--pretrained_text_feature', type=str, default=None)
    parser.add_argument('--truncate_captions', dest='truncate_captions', action='store_true', help='Captions passed in which exceed the max token length will be truncated if this is set.')
    parser.add_argument('--random_resize_crop_lower_ratio', dest='resize_ratio', type=float, default=1, help='Random resized crop lower ratio')
    parser.add_argument('--which_tokenizer', type=str, default='simple', help='(yttm | hug | simple | chinese)')
    parser.add_argument('--taming', dest='taming', action='store_true')
    parser.add_argument('--bpe_path', type=str, help='path to your BPE json file')
    parser.add_argument('--fp16', action='store_true', help='(experimental) - Enable DeepSpeed 16 bit precision. Reduces VRAM.')
    parser.add_argument('--amp', action='store_true', help='Apex "O1" automatic mixed precision. More stable than 16 bit precision. Can\'t be used in conjunction with deepspeed zero stages 1-3.')
    parser.add_argument('--name', default='dalle_train_transformer', help='experiment name, if not using wandb')
    parser.add_argument('--name_suffix', default='', type=str)
    parser.add_argument('--visual', action='store_true', help='add visual control?')
    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--rank', type=int, default=0, help='node rank for distributed training')
    parser.add_argument('--gpu_ids', type=int, default=None, help='gpu id to use')
    parser.add_argument('--workers', default=16, type=int, help='# data loading workers')
    parser.add_argument('--world_size', default=1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--dist_url', default='tcp://localhost:10001', type=str, help='url used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--multiprocessing_distributed', action='store_true', help='Use multi-processing distributed training to launch N processes per node, which has N GPUs.')

    parser.add_argument('--save_every_n_steps', default = 2000, type = int, help = 'Save a checkpoint every n steps')
    parser.add_argument('--batch_size', default = 4, type = int, help = 'Batch size')
    parser.add_argument('--ga_steps', default = 1, type = int, help = 'Number of steps to accumulate gradients across per each iteration. DeepSpeed only.')
    parser.add_argument('--learning_rate', default = 3e-4, type = float, help = 'Learning rate')
    parser.add_argument('--clip_grad_norm', default = 0.5, type = float, help = 'Clip gradient norm')
    parser.add_argument('--lr_decay', dest = 'lr_decay', action = 'store_true')
    parser.add_argument('--freeze_transformer', action = 'store_true')
    parser.add_argument('--tensorboard', action = 'store_true')
    parser.add_argument('--use_html', action = 'store_true')
    parser.add_argument("--log_root", type=str, help="where to save training logs", default='logs')
    parser.add_argument("--log_every", type=int, default=200, help="logging every # iters")
    parser.add_argument("--sample_every", type=int, default=1000, help="sample every # iters")
    parser.add_argument('--n_sample', default = 4, type = int, help = 'Number of samples (text) to visualize')
    parser.add_argument('--n_per_sample', default = 1, type = int, help = 'Number of images per text sample to visualize')
    parser.add_argument('--seed', default = 42, type = int, help = 'Random seed')
    parser.add_argument('--iters', default = 20, type = int, help = 'Number of iterations')
    parser.add_argument('--epochs', default = 100, type = int, help = 'Number of epochs')
    parser.add_argument("--limit_train_batches", type=float, default=1)
    parser.add_argument('--resume', action = 'store_true')
    parser.add_argument('--keep_n_checkpoints', default = None, type = int, help = '(Careful) Deletes old deepspeed checkpoints if there are more than n')
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--lr_scheduler', type=str, default='reducelronplateau')
    parser.add_argument('--clip_ranking', action = 'store_true')
    parser.add_argument('--clip_path', type=str, default=None, help='path to your pretrained CLIP')
    parser.add_argument('--lr_scheduler_every', default = 1, type = int, help = 'step lr scheduler every n steps')
    parser.add_argument('--lr_scheduler_step_size', default = 10000, type = int, help = 'T_max or step_size')
    parser.add_argument('--lr_scheduler_warmup', default = 5000, type = int, help = 'T_max or step_size')
    parser.add_argument('--weight_decay', type = float, default = 0)
    parser.add_argument('--deterministic', action = 'store_true')
    parser.add_argument('--beta_msm', default = 1.0, type = float)
    parser.add_argument('--beta_rel', default = 0.5, type = float)
    parser.add_argument('--beta_vid', default = 0, type = float)
    parser.add_argument('--beta_gan', default = 0, type = float)
    parser.add_argument('--which_fake', type = str, default = 'mask_predict')
    parser.add_argument('--frame_num', default = 8, type = int)
    parser.add_argument('--frame_step', default = 8, type = int)
    parser.add_argument('--estimate_real', action = 'store_true')
    parser.add_argument('--pnag_argmax', action = 'store_true')
    parser.add_argument('--pnag_dynamic', action = 'store_true')
    parser.add_argument('--rand_visual', action = 'store_true')
    parser.add_argument('--fullvc', action = 'store_true')
    parser.add_argument('--negvc', action = 'store_true')
    parser.add_argument('--attr_mode', type=str, default='object')
    parser.add_argument('--n_accum_step', default = 1, type = int)
    parser.add_argument('--dropout_vc', type = float, default = 0.1, help = 'prob of visual control to be zeroed')
    parser.add_argument('--msm_strategy_prob', type = str, default = '7,1,1,1', help = 'comma separated list')
    parser.add_argument('--msm_bernoulli_prob', type = str, default = '0.2,0.2', help = 'comma separated list')
    parser.add_argument('--relvid_bernoulli_prob', type = str, default = '0.1,0.9', help = 'comma separated list')
    parser.add_argument('--vid_strategy_prob', type = str, default = '1,1,1,1', help = 'comma separated list')
    parser.add_argument('--rel_no_fully_masked', action = 'store_true')
    parser.add_argument('--insert_sep', action = 'store_true')
    parser.add_argument('--mask_predict_steps', nargs = '+', default = [0], type = int)
    parser.add_argument('--mask_predict_steps1', default = 0, type = int)
    parser.add_argument('--mask_predict_steps_train', default = 5, type = int)
    parser.add_argument('--use_pc', action = 'store_true')
    parser.add_argument('--pc_prob', type = float, default = 0, help = 'prob of preservation control')

    parser.add_argument('--dim', default = 512, type = int, help = 'Model dimension')
    parser.add_argument('--text_seq_len', default = 256, type = int, help = 'Text sequence length')
    parser.add_argument('--depth', default = 2, type = int, help = 'Model depth')
    parser.add_argument('--heads', default = 8, type = int, help = 'Model number of heads')
    parser.add_argument('--dim_head', default = 64, type = int, help = 'Model head dimension')
    parser.add_argument('--reversible', dest = 'reversible', action='store_true')
    parser.add_argument('--loss_img_weight', default = 7, type = int, help = 'Image loss weight')
    parser.add_argument('--attn_types', default = 'full', type = str, help = 'comma separated list of attention types. attention type can be: full or sparse or axial_row or axial_col or conv_like.')
    parser.add_argument('--pretrained_transformer', type=str, default='none')
    parser.add_argument('--image_size', default = None, type = int, help = 'force to use this size if set to > 0')
    parser.add_argument('--num_targets', default = 1, type = int, help = 'number of frames to generate')
    parser.add_argument('--num_visuals', default = 1, type = int, help = 'number of frames to generate')
    parser.add_argument('--use_separate_visual_emb', action='store_true')
    parser.add_argument('--text_emb_bottleneck', type = str, default = None)
    parser.add_argument('--fixed_language_model', type=str, default=None)
    parser.add_argument('--drop_sentence', action = 'store_true')
    parser.add_argument('--clip_text_emb', type = str, default = None)
    parser.add_argument('--beta_clip', default = 0, type = float)

    parser.add_argument('--test_mode', type = str, default = None)
    parser.add_argument('--eval_mode', type = str, default = None)
    parser.add_argument('--eval_metric', type = str, nargs = '+', default = ['fvd_prd'])
    parser.add_argument('--eval_num', type = int, default = 2048)
    parser.add_argument('--vc_mode', type = str, default = None)
    parser.add_argument('--pc_mode', type = str, default = None)

    parser.add_argument('--description', type = str, default = None)
    parser.add_argument('--no_debug', action='store_true')
    parser.add_argument('--num_workers', default = 16, type = int)
    parser.add_argument('--video_format', type = str, default = 'gif')

    parser.add_argument('--ar', action='store_true')

    parser.add_argument('--mp_T1n', type = int, default = 10, help = 'L1, number of steps for mask')
    parser.add_argument('--mp_T2n', type = int, default = 10, help = 'L2, number of steps for mask')
    parser.add_argument('--mp_T3n', type = int, default = 30, help = 'L3, number of steps for mask')
    parser.add_argument('--mp_N1n', type = float, default = 0.9, help = 'alpha1 for mask')
    parser.add_argument('--mp_N2n', type = float, default = 0.1, help = 'beta1 for mask')
    parser.add_argument('--mp_N3n', type = float, default = 0.125, help = 'alpha2 for mask')
    parser.add_argument('--mp_N4n', type = float, default = 0.0625, help = 'alpha3 for mask')
    parser.add_argument('--mp_T1t', type = int, default = 10, help = 'L1, number of steps for noise')
    parser.add_argument('--mp_T2t', type = int, default = 5, help = 'L2, number of steps for noise')
    parser.add_argument('--mp_T3t', type = int, default = 35, help = 'L3, number of steps for noise')
    # parser.add_argument('--mp_N1t', type = float, default = 0.4)
    # parser.add_argument('--mp_N2t', type = float, default = 0.02)
    # parser.add_argument('--mp_N3t', type = float, default = 0.01)
    parser.add_argument('--mp_N1t', type = float, default = 0., help = 'alpha1 for noise')
    parser.add_argument('--mp_N2t', type = float, default = 0., help = 'beta1 for noise')
    parser.add_argument('--mp_N3t', type = float, default = 0., help = 'alpha2 for noise')
    parser.add_argument('--mp_N4t', type = float, default = 0., help = 'alpha3 for noise')
    parser.add_argument('--mp_T', type = int, default = 30, help = 'number of total steps for mask-predict')
    parser.add_argument('--mp_B', type = int, default = 1, help = 'beam search size')

    parser.add_argument('--eval_image_folder', action='store_true')
    parser.add_argument('--visual_aug_mode', type = str, default = None)

    parser.add_argument('--rep_num', default = 1, type = int)
    parser.add_argument('--t_overlap', default = 1, type = int)
    parser.add_argument('--t_repeat', default = 10, type = int)
    parser.add_argument('--use_cvae', action='store_true')
    parser.add_argument('--eval_drop_sent', action='store_true')
    parser.add_argument('--save_codebook', action='store_true')
    parser.add_argument('--long_mode', type = str, default = 'long')
    parser.add_argument('--slow', action='store_true')
    parser.add_argument('--slow_mode', type=str, default=None)
    parser.add_argument('--drop_sentence_mode', type=str, default=None)

    parser.add_argument('--dm', action='store_true')
    parser.add_argument('--dm_timesteps', type=int, default=32)
    parser.add_argument('--dm_mask_schedule', default='random', type=str)
    parser.add_argument('--dm_time_schedule', default='uniform', type=str)
    parser.add_argument('--dm_loss_type', default='simple', type=str)
    parser.add_argument('--dm_sample_mode', default='gamma_cosine', type=str)
    parser.add_argument('--dm_mask_predict', action='store_true')
    parser.add_argument('--dm_sample_with_confidence', action='store_true')
    parser.add_argument('--dm_use_time_embedding', action='store_true')
    parser.add_argument('--dm_use_time_token', action='store_true')

    args = parser.parse_args()

    args.multiprocessing_distributed = True  # TODO: always use multiprocessing_distributed
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.world_batch_size = args.batch_size
    ngpus_per_node = torch.cuda.device_count()
    main_worker(
        args.gpu_ids,
        ngpus_per_node,
        args,
    )

@torch.no_grad()
def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    torch.backends.cudnn.benchmark = True

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    assert Path(args.image_text_folder).exists(), f'The path {args.image_text_folder} was not found.'

    def is_root_worker():
        return True

    # constants

    if args.no_debug:
        args.debug = False
    args.truncate_captions = True
    args.num_visuals *= args.visual
    args.sample_every = 1

    if args.ar:
        args.debug = False
        args.mask_predict_steps = [0]
        args.mask_predict_steps1 = 0
        args.num_visuals = max(1, args.num_visuals)
    if args.dm:
        args.debug = False

    if True:
        args.deterministic = True

    if args.eval_mode == 'eval':
        args.batch_size = 16  # make samples reproducible

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

    MODEL_DIM = args.dim
    TEXT_SEQ_LEN = args.text_seq_len
    DEPTH = args.depth
    HEADS = args.heads
    DIM_HEAD = args.dim_head
    REVERSIBLE = args.reversible
    LOSS_IMG_WEIGHT = args.loss_img_weight

    ATTN_TYPES = tuple(args.attn_types.split(','))

    MSM_STRATEGY_PROB = np.array(list(map(float, args.msm_strategy_prob.split(','))))
    MSM_STRATEGY_PROB /= MSM_STRATEGY_PROB.sum()
    VID_STRATEGY_PROB = np.array(list(map(float, args.vid_strategy_prob.split(','))))
    VID_STRATEGY_PROB /= VID_STRATEGY_PROB.sum()

    # logging

    DALLE_PATH = Path(args.dalle_path) if exists(args.dalle_path) else None
    if args.eval_image_folder:
        pass
    else:
        if args.dalle_path is None:
            checkpoints = natsort.natsorted(os.listdir(str(Path(args.log_root) / args.name / 'weights')))
            assert len(checkpoints) > 0, f'Nothing to resume from.'
            DALLE_PATH = Path(args.log_root) / args.name / 'weights' / checkpoints[-1] / 'dalle.pt'
    args.dalle_path = DALLE_PATH

    args.name += args.name_suffix  # TODO: remove this

    RESUME = False
    LOG_DIR = Path(args.log_root) / args.name
    LOG_SAMPLE_DIR = LOG_DIR / 'samples'
    args.log_dir = LOG_DIR
    args.log_sample_dir = LOG_SAMPLE_DIR

    assert args.dalle_path 
    which_ckpt = str(DALLE_PATH).split('/')[-2]

    args.log_metric_dir = LOG_DIR / 'metrics' / which_ckpt
    os.makedirs(args.log_metric_dir, exist_ok=True)

    if is_root_worker():
        os.makedirs(LOG_DIR, exist_ok=True)
        os.makedirs(LOG_DIR / 'samples', exist_ok=True)
        os.makedirs(LOG_DIR / 'weights', exist_ok=True)
        utils.print_args(None, args)
        if args.ar:
            shutil.copyfile('dalle_pytorch/dalle_artv.py', LOG_DIR / 'dalle_artv.py.txt')
        elif args.dm:
            shutil.copyfile('dalle_pytorch/dalle_absorb.py', LOG_DIR / 'dalle_absorb.py.txt')
        else:
            shutil.copyfile('dalle_pytorch/dalle_bert.py', LOG_DIR / 'dalle_bert.py.txt')

    USE_HTML = args.use_html
    LOG_WEB_DIR = LOG_DIR / 'web'
    webpage = None
    if USE_HTML and is_root_worker():
        webpage = utils_html.initialize_webpage(LOG_WEB_DIR, 'DALLE: ' + args.name, RESUME)

    # tokenizer

    # tokenizer = get_tokenizer(args)
    # text_feature_dim, encode_text = get_text_feature_extractor(args)
    # if text_feature_dim > 0:
    #     tokenizer = None  # if use fixed text lang model, set tokenizer to None
    if args.fixed_language_model is not None:
        tokenizer2, language_model, text_feature_dim, encode_text = get_fixed_language_model(args)
        # language_model = model_to_gpu(language_model, args.gpu, True)  # TODO: false
        language_model = language_model.cuda()
        tokenizer = None  # TODO: avoid tokenization and get raw text
    else:
        language_model, tokenizer2 = None, None
        text_feature_dim = 0
        tokenizer = get_tokenizer(args)
    
    # model path

    if args.vae_path == '':
        args.vae_path = None
    if args.cvae_path == '':
        args.cvae_path = None

    VAE_PATH = args.vae_path
    args.use_cvae = args.use_cvae or args.cvae_path is not None

    model_weights, optim_weights = None, None
    START_ITER = 0
    GLOBAL_STEPS = 0  # global_steps

    # get vae model

    vae = None
    if not args.eval_image_folder:
        vae, vae_params = get_vae_model(
            args.which_vae,
            vae_path=VAE_PATH,
            image_size=args.image_size
        )

    cvae = None
    if not args.eval_image_folder:
        if args.use_cvae:
            cvae, cvae_params = get_vae_model(
                args.which_vae,
                vae_path=args.cvae_path,
                image_size=args.image_size
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

    # if DALLE_PATH is not None:  # if not resume and dalle_path is given, load weights
    if not args.eval_image_folder:
        assert exists(DALLE_PATH), 'DALLE model file does not exist'
        ckpt = torch.load(str(DALLE_PATH))
        model_weights = ckpt['weights']

        IMAGE_SIZE = args.image_size or vae.image_size
        args.image_size = vae.image_size = IMAGE_SIZE
        if cvae is not None:
            cvae.image_size = IMAGE_SIZE
    else:
        IMAGE_SIZE = args.image_size

    # initialize DALL-E / BERT and optimizer
    dalle = None
    dalle_module = None
    if not args.eval_image_folder:
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

        if model_weights is not None:
            dalle.load_state_dict(model_weights, strict=True)

        dalle = model_to_gpu(dalle, args.gpu, True)  # TODO
        dalle_module = dalle.module  # TODO

    args.is_shuffle = True  # TODO

    ds = get_dataset(args, tokenizer)
    assert len(ds) > 0, 'dataset is empty'
    if is_root_worker():
        print(f'{len(ds)} image-text pairs found for training')

    data_sampler = None

    dl = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,#(data_sampler is None),
        drop_last=True,
        sampler=data_sampler,
        num_workers=0,
        pin_memory=False,
    )
    dl_iter = sample_data(dl, data_sampler)

    if args.eval_mode == 'eval':  # evaluate quantitative metrics
        if 'fvd_prd' in args.eval_metric:
            evaluate(
                args,
                dalle_module,
                tokenizer,
                tokenizer2,
                language_model,
                dl_iter,
            )
        if 'intra_fvd' in args.eval_metric:
            evaluate_intra(
                args,
                dalle_module,
                tokenizer,
                tokenizer2,
                language_model,
                dl_iter,
            )
        if 'clip' in args.eval_metric:
            evaluate_clip(
                args,
                dalle_module,
                tokenizer,
                tokenizer2,
                language_model,
                dl_iter,
            )
        if 'clip_real' in args.eval_metric:
            evaluate_clip(
                args,
                dalle_module,
                tokenizer,
                tokenizer2,
                language_model,
                dl_iter,
                upper_bound=True,
            )
        exit(0)
    
    pbar = range(ITERS)  # TODO
    if is_root_worker():
        pbar = tqdm(pbar, initial=START_ITER, dynamic_ncols=True, smoothing=0.01)

    for idx in pbar:
        i = idx + START_ITER
        which_iter = f"{i:07d}"
        GLOBAL_STEPS += 1

        if i > ITERS:
            print('done!')
            break

        text_neg, visuals_neg = None, None
        if args.negvc:
            text, frames, visuals, visuals_neg, text_neg = next(dl_iter)
            visuals_neg, text_neg = map(lambda t: t.cuda(), (visuals_neg, text_neg))
        else:
            text, frames, visuals = next(dl_iter)  # frames [B, T, C, H, W]
        if args.visual and len(visuals.shape) == 4:
            assert args.num_visuals == 1
            visuals = visuals.unsqueeze(1)

        if args.fp16:
            frames = frames.half()

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
                text = mean_pooling(model_output, encoded_input['attention_mask'])
        else:
            text = text.cuda()
            text_description = None

        # =================== visualization ======================
        if args.eval_mode == 'long':
            visualize_long(
                args,
                dalle_module,
                tokenizer,
                {
                    'description': text_description,
                    'text': text,
                    'text_neg': text_neg,
                    'target': frames,
                    'visual': visuals,
                    'visual_neg': visuals_neg,
                },
                which_iter,
                webpage,
                args.description,
                tokenizer2,
                language_model,
            )
        else:
            visualize(
                args,
                dalle_module,
                tokenizer,
                {
                    'description': text_description,
                    'text': text,
                    'text_neg': text_neg,
                    'target': frames,
                    'visual': visuals,
                    'visual_neg': visuals_neg,
                },
                which_iter,
                webpage,
                args.description,
                tokenizer2,
                language_model,
            )
        # ========================================================

if __name__ == "__main__":
    main()
