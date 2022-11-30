from math import log2, sqrt
from numpy.core.numeric import _convolve_dispatcher
import torch
from torch import nn, einsum
import torch.nn.functional as F
import torchvision.transforms as T

from axial_positional_embedding import AxialPositionalEmbedding
from einops import rearrange

from mmvid_pytorch import distributed_utils
from mmvid_pytorch.vae import OpenAIDiscreteVAE
from mmvid_pytorch.vae import VQGanVAE1024, VQGanVAE
from mmvid_pytorch.transformer import Transformer, DivideMax
from mmvid_pytorch.modules import AxialPositionalEmbeddingList
import random
import numpy as np
from image_pool import ImagePool
import sys
import math
import torch.distributions as dists
import scipy.special
import pdb
st = pdb.set_trace

FAKE_POOL_SIZE = 64

# helpers

def disp(x):
    print(x[0].reshape(8, 8, 8)[0])

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def always(val):
    def inner(*args, **kwargs):
        return val
    return inner

def is_empty(t):
    return t.nelement() == 0

def masked_mean(t, mask, dim = 1):
    t = t.masked_fill(~mask[:, :, None], 0.)
    return t.sum(dim = 1) / mask.sum(dim = 1)[..., None]

def set_requires_grad(model, value):
    if model is not None:
        for param in model.parameters():
            param.requires_grad = value

def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

# sampling helpers

def top_k(logits, thres = 0.5):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

# Augmentation helpers

from itertools import permutations
PERM_LIST = None

def randperm(n, ordered=False):
    global PERM_LIST
    # ordered: include ordered permutation?
    if ordered:
        return torch.randperm(n)
    else:
        if n < 6:
            if PERM_LIST is None:
                PERM_LIST = list(permutations(range(n)))[1:]
            return random.choice(PERM_LIST)
        perm_ord = torch.tensor(range(n))
        while True:
            perm = torch.randperm(n)
            if (perm != perm_ord).any():
                return perm

def swap(tensor, dim=0):
    if tensor.shape[dim] % 2 == 0:
        tensor_swapped = torch.cat(torch.chunk(tensor, 2, dim=dim)[::-1], dim=dim)
    else:
        idx_perm = randperm(tensor.shape[dim], False)
        if dim == 0:
            tensor_swapped = tensor[idx_perm, ...]
        elif dim == 1:
            tensor_swapped = tensor[:,idx_perm, ...]
        else:
            raise RuntimeError
    return tensor_swapped

def swap_one_frame_along_batch(tokens, t=1, shuffle=False):
    tokens_shuffled = tokens.detach().clone()
    b, n, c = tokens.shape
    tokens_shuffled = tokens_shuffled.reshape(b, t, n // t, c)
    idx = np.random.randint(0, t, b)
    if shuffle:
        perm_idx = randperm(t)
        frames_shuffled = tokens_shuffled[range(b),idx,...][perm_idx,...]
    else:
        frames_shuffled = swap(tokens_shuffled[range(b),idx,...], 0)
    tokens_shuffled[range(b),idx,...] = frames_shuffled
    tokens_shuffled = tokens_shuffled.reshape(b, n, c)
    return tokens_shuffled

def warp_video_with_color(video):
    # video (n, t, 3, h, w)
    out = []
    for n in range(video.shape[0]):
        x = video[n]  # x (c, h, w)
        c_shift = torch.rand(1) - 0.5
        c_shift = c_shift.to(x.device)
        m = torch.zeros_like(x)
        num = random.randint(0, 3)
        if num == 0:
            m.data += c_shift
        elif num == 1:
            m[:,0].data += c_shift
        elif num == 2:
            m[:,1].data += c_shift
        else:
            m[:,2].data += c_shift
        out.append(torch.clamp(x + m, 0, 1))
    return torch.stack(out)

def warp_with_color(x):
    # x (c, h, w)
    c_shift = torch.rand(1) - 0.5
    c_shift = c_shift.to(x.device)
    m = torch.zeros_like(x)
    num = random.randint(0, 3)
    if num == 0:
        m.data += c_shift
    elif num == 1:
        m[0].data += c_shift
    elif num == 2:
        m[1].data += c_shift
    else:
        m[2].data += c_shift
    out = torch.clamp(x + m, 0, 1)
    return out.unsqueeze(0)  # out (1, 3, h, w)

def warp_with_affine(x, angle=180, trans=0.1, scale=0.05):
    angle = np.pi * angle / 180.

    pa = torch.FloatTensor(4)
    th = torch.FloatTensor(2, 3)

    pa[0].uniform_(-angle, angle)
    pa[1].uniform_(-trans, trans)
    pa[2].uniform_(-trans, trans)
    pa[3].uniform_(1. - scale, 1. + scale)

    th[0][0] = pa[3] * torch.cos(pa[0])
    th[0][1] = pa[3] * torch.sin(-pa[0])
    th[0][2] = pa[1]
    th[1][0] = pa[3] * torch.sin(pa[0])
    th[1][1] = pa[3] * torch.cos(pa[0])
    th[1][2] = pa[2]

    x = x.unsqueeze(0)
    th = th.unsqueeze(0)
    grid = F.affine_grid(th, x.size()).to(x.device)
    out = F.grid_sample(x, grid, padding_mode="reflection")
    return out  # out (1, 3, h, w)

def warp(x, vid_strategy_prob=[0.25, 0.25, 0.25, 0.25]):
    # x (b, t, c, h, w)
    b, t, c, h, w = x.shape
    out = []
    for i in range(b):
        strategy = np.random.choice(range(4), p=vid_strategy_prob)
        if strategy == 0:
            # swap frame from another seq
            i_ = np.random.choice(list(set(range(b))-{i}))
            y = x[i].detach().clone()
            j1 = random.randint(0, t-1)
            j2 = random.randint(0, t-1)
            y[j1,...] = x[i_,j2,...]
            out.append(y)
        elif strategy == 1:
            # shuffle frames
            perm_idx = randperm(t)
            y = x[i,perm_idx,...].detach().clone()
            out.append(y)
        elif strategy == 2:
            # color
            j1 = random.randint(0, t-1)
            y = x[i].detach().clone()
            y[j1,...] = warp_with_color(y[j1]).squeeze(0)
            out.append(y)
        elif strategy == 3:
            # affine
            j1 = random.randint(0, t-1)
            y = x[i].detach().clone()
            y[j1,...] = warp_with_affine(y[j1], 30, 0.1, 0.1).squeeze(0)
            out.append(y)
        else:
            raise NotImplementedError
    out = torch.stack(out, 0)
    return out

# discrete vae class

class ResBlock(nn.Module):
    def __init__(self, chan):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(chan, chan, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(chan, chan, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(chan, chan, 1)
        )

    def forward(self, x):
        return self.net(x) + x

# helper

# def define_transformer():
#     return None

# main DALL-E class

class AbsorbingDiffusion(nn.Module):
    def __init__(
        self,
        *,
        dim,
        vae,
        cvae = None,
        num_text_tokens = 10000,
        text_seq_len = 256,
        depth,
        heads = 8,
        dim_head = 64,
        reversible = False,
        attn_dropout = 0.,
        ff_dropout = 0,
        sparse_attn = False,
        attn_types = None,
        loss_img_weight = 7,
        stable = False,
        text_feature_dim = 0,
        fixed_language_model = None,
        pretrained_transformer = 'none',
        max_time_len = 16,
        num_visuals = 1,
        num_targets = 1,
        use_separate_visual_emb = False,
        insert_sep = False,
        text_emb_bottleneck = False,
        clip_text_emb = None,
        num_timesteps = 256,
        num_timesteps_train = 256,
        num_timesteps_inference = 32,
        use_time_embedding = False,
        use_time_token = False,
    ):
        super().__init__()
        # assert isinstance(vae, (DiscreteVAE, OpenAIDiscreteVAE, VQGanVAE1024)), 'vae must be an instance of DiscreteVAE'
        """
        Special Tokens:
        [REL]  if text-video are relevant
        [FDL]  fedility
        [VID]  if video is continuous (shuffle frames)
        [SUM]  indicating summary
        [MASK] masking
        [EOT]  end of text
        [EOV]  end of visual
        [SEP]  separation (reserved)
        """
        image_size = vae.image_size
        num_image_tokens = vae.num_tokens
        image_fmap_size = (vae.image_size // (2 ** vae.num_layers))
        image_seq_len = image_fmap_size ** 2
        self.dim = dim
        self.num_timesteps = num_timesteps
        self.use_time_embedding = use_time_embedding
        self.use_time_token = use_time_token
        # time embedding from 0 to T
        self.time_emb = nn.Embedding(num_timesteps+1, dim) if use_time_embedding else None

        # assert num_visuals <= 1
        self.num_visuals = num_visuals
        self.num_targets = num_targets

        # self.fake_sample_pool = ImagePool(FAKE_POOL_SIZE, 1)
        self.random_erasing = T.RandomErasing(p=1, scale=(0.2, 0.8), ratio=(0.5, 2), value=0)

        if fixed_language_model is None:
            num_text_tokens = num_text_tokens + text_seq_len  # reserve unique padding tokens for each position (text seq len)
            self.text_emb = nn.Embedding(num_text_tokens, dim)
            self.text_pos_emb = nn.Embedding(text_seq_len, dim)
            self.text_feature_mapping = lambda x: x
        else:
            assert text_feature_dim > 0
            text_seq_len = 1
            num_text_tokens = 1
            self.text_emb, self.text_pos_emb = None, None
            if text_emb_bottleneck is not None:
                nf = int(text_emb_bottleneck)
                self.text_feature_mapping = nn.Sequential(
                    nn.LayerNorm(text_feature_dim),
                    nn.Linear(text_feature_dim, nf),
                    nn.LayerNorm(nf),
                    nn.Linear(nf, dim),
                    nn.LayerNorm(dim),
                )
            else:
                self.text_feature_mapping = nn.Linear(text_feature_dim, dim)
                # self.text_feature_mapping = nn.Sequential(
                #     nn.LayerNorm(text_feature_dim),
                #     nn.Linear(text_feature_dim, dim),
                # )

        self.image_emb = nn.Embedding(num_image_tokens + 2, dim)  # TODO: for masking+separate visual
        
        # if use_time_embedding:
        #     self.target_pos_emb = AxialPositionalEmbedding(dim, axial_shape = (num_timesteps, num_targets, image_fmap_size, image_fmap_size))
        # else:
        self.target_pos_emb = AxialPositionalEmbedding(dim, axial_shape = (num_targets, image_fmap_size, image_fmap_size))

        if cvae is not None:
            use_separate_visual_emb = True
        if num_visuals > 0:
            if use_separate_visual_emb:
                self.visual_emb = nn.Embedding(num_image_tokens + 2, dim)  # TODO: for masking+separate visual
            else:
                self.visual_emb = None
            # if self.num_visuals > 1:
            #     self.visual_pos_emb = AxialPositionalEmbedding(dim, axial_shape = (num_visuals, image_fmap_size, image_fmap_size))
            # else:
            #     self.visual_pos_emb = AxialPositionalEmbedding(dim, axial_shape = (image_fmap_size, image_fmap_size))
            self.visual_pos_emb = AxialPositionalEmbeddingList(dim, num_visuals, axial_shape = (image_fmap_size, image_fmap_size))

        self.image_token_lut = {
            '[MASK]': num_image_tokens,
            '[SEP]' : num_image_tokens + 1,
        }
        self.mask_id = self.image_token_lut['[MASK]']

        self.num_text_tokens = num_text_tokens # for offsetting logits index and calculating cross entropy loss
        self.num_image_tokens = num_image_tokens
        self.text_seq_len = text_seq_len
        self.image_seq_len = image_seq_len
        self.image_fmap_size = image_fmap_size
        self.image_size = image_size
        self.visual_seq_len = num_visuals * image_seq_len + (num_visuals * insert_sep)
        self.target_seq_len = num_targets * image_seq_len
        self.insert_sep = insert_sep

        self.special_token_lut = {
            '[REL]':  0,
            '[FDL]':  1,
            '[VID]':  2,
            '[SUM]':  3,
            '[MASK]': 4,
        }
        self.num_special_tokens = len(self.special_token_lut)
        self.before_control_tok = [0]  # rel
        self.after_control_tok = [1, 2]  # vid, fdl
        self.before_control_seq_len = len(self.before_control_tok)
        self.after_control_seq_len = len(self.after_control_tok)
        self.special_emb = nn.Embedding(self.num_special_tokens, dim)
        self.special_pos_emb = nn.Embedding(self.num_special_tokens, dim)
        self.rel_tok_index = 0
        self.fdl_tok_index = self.before_control_seq_len + self.text_seq_len + self.visual_seq_len + 0
        self.vid_tok_index = self.before_control_seq_len + self.text_seq_len + self.visual_seq_len + 1
        self.txt_tok_index = self.before_control_seq_len + 0
        # TODO: FDL is reused for TIME
        self.time_tok_index = self.fdl_tok_index

        seq_len = self.before_control_seq_len + self.text_seq_len + self.visual_seq_len + self.after_control_seq_len + self.target_seq_len
        # control_seq_len = self.before_control_seq_len + self.text_seq_len + self.visual_seq_len + self.after_control_seq_len
        self.total_seq_len = seq_len

        self.vae = vae
        self.cvae = cvae
        set_requires_grad(self.vae, False) # freeze VAE from being trained
        set_requires_grad(self.cvae, False) # freeze VAE from being trained

        # self.pretrained_text_feature = pretrained_text_feature
        self.fixed_language_model = fixed_language_model
        self.pretrained_transformer = pretrained_transformer
        mask_prev_index = [self.fdl_tok_index, self.vid_tok_index]
        assert pretrained_transformer != 'default'
        if pretrained_transformer.startswith('vqgan'):
            from mmvid_pytorch.transformers.vqgan_model import VQGanTransformer
            self.transformer = VQGanTransformer(
                pretrained_transformer,
                seq_len,
                causal=True,
                mask_type='mask_prev',
                mask_kwargs={'index': mask_prev_index},
            )
        elif pretrained_transformer == 'mingpt':
            from mmvid_pytorch.transformers.vqgan_model import minGPTTransformer
            self.transformer = minGPTTransformer(
                pretrained_transformer,
                seq_len,
                causal=True,
                mask_type='mask_prev',
                mask_kwargs={'index': mask_prev_index},
                n_head=heads,
                n_layers=depth,
                n_emb=dim,
            )
        elif pretrained_transformer.startswith('openai_clip'):
            from mmvid_pytorch.transformers.clip_model import OpenAICLIPTransformer
            self.transformer = OpenAICLIPTransformer(
                seq_len,
                pretrained_transformer,
                causal=True,
                mask_type='mask_prev',
                mask_kwargs={'index': mask_prev_index},
            )
        elif pretrained_transformer in ['none', 'default']:  # train from scratch
            self.transformer = Transformer(
                dim = dim,
                causal = False,
                seq_len = seq_len,
                depth = depth,
                heads = heads,
                dim_head = dim_head,
                reversible = reversible,
                attn_dropout = attn_dropout,
                ff_dropout = ff_dropout,
                attn_types = attn_types,
                image_fmap_size = image_fmap_size,
                sparse_attn = sparse_attn,
                stable = stable
            )

        self.stable = stable

        if stable:
            self.norm_by_max = DivideMax(dim = -1)

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, self.num_image_tokens),
        )
        self.to_logits_rel = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1),
        )
        if use_time_token:  # TODO: reuse FDL for TIME token
            self.to_logits_fdl = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, self.num_timesteps+1),  # t in [1, ..., T]
            )
        else:
            self.to_logits_fdl = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, 1),
            )
        self.to_logits_vid = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1),
        )
        self.clip_text_emb = clip_text_emb
        if clip_text_emb is not None:
            self.to_logits_txt = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, 512),
            )

        self.current_step = 0
        self.loss_img_weight = loss_img_weight
        self.visual_eraser = T.RandomErasing(p=0.95, scale=(0.55, 0.85), ratio=(0.5, 2), value=self.num_image_tokens)  # erase visual

        self.elbo_weight = None
        self.register_buffer('Lt_count', torch.zeros(self.num_timesteps+1))
        self.register_buffer('Lt_history', torch.zeros(self.num_timesteps+1))
        self.register_buffer('mce_history', torch.zeros(self.num_timesteps+1))

    @torch.no_grad()
    @eval_decorator
    def generate_images(
        self,
        text,
        *,
        clip = None,
        visual = None,
        mask = None,
        filter_thres = 0.5,
        temperature = 1.,
        img = None,
        num_init_img_tokens = None,
        t_cond = None,
        argmax = False,
        dynamic = True,
        debug = False,
        erase_visual = False,
        mask_predict_steps = 10,
        vc_mode = None,
        face_mode = None,
        use_mask_predict = False,
        sample_mode = 'gamma_linear',
        mp_config = None,
    ):
        vae, text_seq_len, image_seq_len, num_text_tokens = self.vae, self.text_seq_len, self.image_seq_len, self.num_text_tokens

        control_emb = self(
            text,
            visual=visual,
            erase_visual=erase_visual,
            erase_visual_half=True,  # TODO: always erase half during generation if erase visual
            vc_mode=vc_mode,
            face_mode=face_mode,
            return_loss = False,
        )
        # if sample_mode.startswith('gamma_'):
        img_seq, pnag_samples, _ = self.sample_denoising(
            control_emb,
            argmax=argmax,
            dynamic=dynamic,
            debug=debug,
            steps=mask_predict_steps,
            gamma=sample_mode[6:],
        )
        img_seq = rearrange(img_seq, 'b (t n) -> (b t) n', n = self.image_seq_len)
        images = vae.decode(img_seq)
        images = rearrange(images, '(b t) c h w -> b t c h w', t = self.num_targets)

        return images, pnag_samples
    
    @torch.no_grad()
    @eval_decorator
    def generate_images_debug(
        self,
        text,
        *,
        clip = None,
        visual = None,
        mask = None,
        filter_thres = 0.5,
        temperature = 1.,
        img = None,
        num_init_img_tokens = None,
        t_cond = None,
        argmax = False,
        dynamic = True,
        debug = False,
        erase_visual = False,
        mask_predict_steps = 10,
        vc_mode = None,
        face_mode = None,
        use_mask_predict = False,
        sample_mode = 'gamma_linear',
        mp_config = None,
        sample_with_confidence = True,
    ):
        vae, text_seq_len, image_seq_len, num_text_tokens = self.vae, self.text_seq_len, self.image_seq_len, self.num_text_tokens

        control_emb = self(
            text,
            visual=visual,
            erase_visual=erase_visual,
            erase_visual_half=True,  # TODO: always erase half during generation if erase visual
            vc_mode=vc_mode,
            face_mode=face_mode,
            return_loss = False,
        )
        if sample_mode.startswith('gamma_'):
            img_seq, pnag_samples, _ = self.sample_denoising2(
                control_emb,
                argmax=argmax,
                dynamic=dynamic,
                debug=debug,
                steps=mp_config['T'],
                gamma=sample_mode[6:],
                use_confidence=sample_with_confidence,
            )
        elif sample_mode == 'unleash' or sample_mode == 'diffusion':
            img_seq, pnag_samples, _ = self.sample_denoising0(
                control_emb,
                argmax=argmax,
                dynamic=dynamic,
                debug=debug,
                steps=mp_config['T'],
                gamma=sample_mode,
                # use_confidence=sample_with_confidence,
            )
        img_seq = rearrange(img_seq, 'b (t n) -> (b t) n', n = self.image_seq_len)
        images = vae.decode(img_seq)
        images = rearrange(images, '(b t) c h w -> b t c h w', t = self.num_targets)

        return images, pnag_samples, None

    def decode_masks(self, mask):
        mask = rearrange(mask, 'b (t h w) -> (b t) 1 h w', h = self.image_fmap_size, w = self.image_fmap_size)
        patch_size = self.image_size // self.image_fmap_size
        mask_ = torch.repeat_interleave(torch.repeat_interleave(mask, patch_size, 2), patch_size, 3)
        mask = F.pad(mask_, (0, 0, 0, 0, 0, 2))  # red
        return mask
    
    def transformer_forward(self, tokens):
        # tokens are embeddings
        out = self.transformer(tokens)
        if self.stable:  # TODO: should we keep this?
            out = self.norm_by_max(out)
        return out

    def decode_images(self, img_seq):
        img_seq = rearrange(img_seq, 'b (t n) -> (b t) n', n = self.image_seq_len)
        images = self.vae.decode(img_seq)
        # images = rearrange(images, '(b t) c h w -> b t c h w', t = self.num_targets)
        return images
    
    def decode_masks(self, mask):
        mask = rearrange(mask, 'b (t h w) -> (b t) 1 h w', h = self.image_fmap_size, w = self.image_fmap_size)
        patch_size = self.image_size // self.image_fmap_size
        mask_ = torch.repeat_interleave(torch.repeat_interleave(mask, patch_size, 2), patch_size, 3)
        mask = F.pad(mask_, (0, 0, 0, 0, 0, 2))  # red
        return mask
    
    def transformer_forward(self, tokens):
        # tokens are embeddings
        out = self.transformer(tokens)
        if self.stable:  # TODO: should we keep this?
            out = self.norm_by_max(out)
        return out

    @torch.no_grad()
    def sample_denoising(self, control_emb, argmax=False, dynamic=True, debug=False, steps=10, preserve=None, t_overlap=0, gamma='linear'):
        def predict_from_logits(logits, argmax=False):
            if argmax:
                # return logits.argmax(dim=-1)
                probabilities = F.softmax(logits, dim=-1)
                probabilities, predictions = torch.max(probabilities, dim=-1)
            else:
                # return dists.Categorical(logits=logits).sample()
                probabilities = F.softmax(logits, dim=-1)
                x_dist = dists.Categorical(probs=probabilities)
                predictions = x_dist.sample().long()
                probabilities = torch.gather(probabilities, -1, predictions.unsqueeze(-1)).squeeze(-1)
            return predictions, probabilities

        # TODO: hardcode just in case
        temp = 1.0
        use_confidence = True
        b, device = control_emb.shape[0], control_emb.device
        control_seq_len = control_emb.shape[1]
        
        T = steps
        N = self.target_seq_len
        ts = np.arange(0, T + 1)
        if gamma == 'linear':
            ns = 1 - ts / T
        elif gamma == 'cosine':
            ns = np.cos(ts / T * np.pi / 2)
        else:
            raise NotImplementedError
        # ns from 1 to 0, t_equiv from T to 0
        t_equiv = np.round(ns * self.num_timesteps)
        # print('t_equiv:', t_equiv)
        ns = np.round(ns * N)

        x_t = torch.ones((b, self.target_seq_len), device=device).long() * self.mask_id
        unmasked = torch.zeros_like(x_t, device=device).bool()
        one_probabilities = torch.ones((b, N), device=device) + 0.01

        fully_masked_emb = self.image_emb(x_t)
        target_pos_emb = self.target_pos_emb(fully_masked_emb)  # TODO
        image_samples = []

        for t in range(T):
            # predict x_0_hat from x_t
            x_t_emb = self.image_emb(x_t)
            t_emb = self.time_emb(torch.full((b,), t_equiv[t], device=device, dtype=torch.long).view(-1, 1)) if self.use_time_embedding else 0
            x_0_out = self.transformer_forward(torch.cat((control_emb, x_t_emb + target_pos_emb + t_emb), dim = 1))
            x_0_logits = self.to_logits(x_0_out[:,control_seq_len:,:])
            x_0_hat, x_0_prob = predict_from_logits(x_0_logits / temp)

            if use_confidence:
                confidences = torch.where(unmasked, one_probabilities, x_0_prob)
            else:
                random_probabilities = torch.rand(x_0_prob.shape, device=device)
                confidences = torch.where(unmasked, one_probabilities, random_probabilities)
            if t == T-1:
                thresholds = 0
            else:
                thresholds = torch.topk(confidences, int(N - ns[t+1]), -1)[0][..., [-1]]

            # where to unmask
            changes = confidences >= thresholds  # changes (newly unmasked) are with high confidence
            # don't unmask somewhere already unmasked
            changes = torch.bitwise_xor(changes, torch.bitwise_and(changes, unmasked))

            # update mask with changes
            unmasked = torch.bitwise_or(unmasked, changes)
            x_t[changes] = x_0_hat[changes]
        
        assert (x_t == self.mask_id).sum() == 0  # no mask_id in the sequence

        return x_t, image_samples, None
    
    @torch.no_grad()
    def sample_denoising2(self, control_emb, argmax=False, dynamic=True, debug=False, steps=10, preserve=None, t_overlap=0, gamma='linear', use_confidence=True):
        def predict_from_logits(logits, argmax=False):
            if argmax:
                # return logits.argmax(dim=-1)
                probabilities = F.softmax(logits, dim=-1)
                probabilities, predictions = torch.max(probabilities, dim=-1)
            else:
                # return dists.Categorical(logits=logits).sample()
                probabilities = F.softmax(logits, dim=-1)
                x_dist = dists.Categorical(probs=probabilities)
                predictions = x_dist.sample().long()
                probabilities = torch.gather(probabilities, -1, predictions.unsqueeze(-1)).squeeze(-1)
            return predictions, probabilities

        # TODO: hardcode just in case
        temp = 1
        # use_confidence = True
        # print(f'num_timesteps: {self.num_timesteps}')
        b, device = control_emb.shape[0], control_emb.device
        control_seq_len = control_emb.shape[1]
        
        T = steps
        N = self.target_seq_len
        ts = np.arange(0, T + 1)
        if gamma == 'linear':
            ns = 1 - ts / T
        elif gamma == 'cosine':
            ns = np.cos(ts / T * np.pi / 2)
        elif gamma == 'square':
            ns = 1 - (ts / T)**2
        elif gamma == 'cubic':
            ns = 1 - (ts / T)**3
        elif gamma == 'sqrt':
            ns = 1 - np.sqrt(ts / T)
        elif gamma == 'sqrtom':
            ns = np.sqrt(1 - ts / T)
        else:
            raise NotImplementedError
        # ns from 1 to 0, t_equiv from T to 0
        t_equiv = np.round(ns * self.num_timesteps)
        ns = np.round(ns * N)

        x_t = torch.ones((b, self.target_seq_len), device=device).long() * self.mask_id
        unmasked = torch.zeros_like(x_t, device=device).bool()
        one_probabilities = torch.ones((b, N), device=device) + 0.01

        fully_masked_emb = self.image_emb(x_t)
        target_pos_emb = self.target_pos_emb(fully_masked_emb)  # TODO
        image_samples = []

        for t in range(T):
            # predict x_0_hat from x_t
            x_t_emb = self.image_emb(x_t)
            t_emb = self.time_emb(torch.full((b,), t_equiv[t], device=device, dtype=torch.long).view(-1, 1)) if self.use_time_embedding else 0
            x_0_out = self.transformer_forward(torch.cat((control_emb, x_t_emb + target_pos_emb + t_emb), dim = 1))
            x_0_logits = self.to_logits(x_0_out[:,control_seq_len:,:])
            x_0_hat, x_0_prob = predict_from_logits(x_0_logits / temp)

            if use_confidence:
                confidences = torch.where(unmasked, one_probabilities, x_0_prob)
            else:
                random_probabilities = torch.rand(x_0_prob.shape, device=device)
                confidences = torch.where(unmasked, one_probabilities, random_probabilities)
            if t == T-1:
                thresholds = 0
            else:
                thresholds = torch.topk(confidences, int(N - ns[t+1]), -1)[0][..., [-1]]

            # where to unmask
            changes = confidences >= thresholds  # changes (newly unmasked) are with high confidence
            # don't unmask somewhere already unmasked
            changes = torch.bitwise_xor(changes, torch.bitwise_and(changes, unmasked))

            # update mask with changes
            unmasked = torch.bitwise_or(unmasked, changes)
            x_t[changes] = x_0_hat[changes]
        
        assert (x_t == self.mask_id).sum() == 0  # no mask_id in the sequence

        return x_t, image_samples, None
    
    @torch.no_grad()
    def sample_denoising0(self, control_emb, argmax=False, dynamic=True, debug=False, steps=10, preserve=None, t_overlap=0, gamma='linear'):
        # TODO: hardcode just in case
        temp = 0.9
        # print(f'num_timesteps: {self.num_timesteps}')
        b, device = control_emb.shape[0], control_emb.device
        control_seq_len = control_emb.shape[1]
        x_t = torch.ones((b, self.target_seq_len), device=device).long() * self.mask_id
        unmasked = torch.zeros_like(x_t, device=device).bool()
        ts = np.arange(1, steps + 1)
        fully_masked_emb = self.image_emb(x_t)
        target_pos_emb = self.target_pos_emb(fully_masked_emb)
        image_samples = []
        for t in reversed(ts):
            # print(f'Sample timestep {t:4d}', end='\r')
            t = torch.full((b,), t, device=device, dtype=torch.long)
            # where to unmask
            changes = torch.rand(x_t.shape, device=device) < 1 / t.float().unsqueeze(-1)
            # don't unmask somewhere already unmasked
            changes = torch.bitwise_xor(changes, torch.bitwise_and(changes, unmasked))
            # update mask with changes
            unmasked = torch.bitwise_or(unmasked, changes)
            x_t_emb = self.image_emb(x_t)
            x_0_out = self.transformer_forward(torch.cat((control_emb, x_t_emb + target_pos_emb), dim = 1))
            x_0_logits = self.to_logits(x_0_out[:,control_seq_len:,:])
            # scale by temperature
            x_0_logits = x_0_logits / temp
            x_0_dist = dists.Categorical(logits=x_0_logits)
            x_0_hat = x_0_dist.sample().long()
            x_t[changes] = x_0_hat[changes]
        return x_t, image_samples, None

    @torch.no_grad()
    def denoising_mask_predict(self, control_emb, argmax=False, dynamic=True, debug=False, steps=10, preserve=None, t_overlap=0):
        def sample_multinomial(logits, temperature=1.):
            # logits = logits + temperature * sample_gumbel(logits)
            probs = F.softmax(logits, dim = 2)
            tok = torch.multinomial(rearrange(probs, 'b n c -> (b n) c'), 1)
            tok = rearrange(tok, '(b n) 1 -> b n 1', b=probs.shape[0])
            Y = torch.gather(probs, 2, tok)
            Y, tok = Y.squeeze(2), tok.squeeze(2)
            return Y, tok

        def sample_gumbel(logit, eps=1e-20):
            U = torch.rand_like(logit)
            return -torch.log(-torch.log(U + eps) + eps)
        
        # TODO: hardcode just in case
        temp = 0.9
        sample_steps = self.num_timesteps
        # print(f'num_timesteps: {self.num_timesteps}')
        b, device = control_emb.shape[0], control_emb.device
        control_seq_len = control_emb.shape[1]
        x_t = torch.ones((b, self.target_seq_len), device=device).long() * self.mask_id
        unmasked = torch.zeros_like(x_t, device=device).bool()
        sample_steps = list(range(1, sample_steps+1))

        fully_masked_emb = self.image_emb(x_t)
        target_pos_emb = self.target_pos_emb(fully_masked_emb)  # TODO
        image_samples = []

        for t in reversed(sample_steps):
            # print(f'Sample timestep {t:4d}', end='\r')
            t = torch.full((b,), t, device=device, dtype=torch.long)

            # where to unmask
            changes = torch.rand(x_t.shape, device=device) < 1/t.float().unsqueeze(-1)
            # don't unmask somewhere already unmasked
            changes = torch.bitwise_xor(changes, torch.bitwise_and(changes, unmasked))
            # update mask with changes
            unmasked = torch.bitwise_or(unmasked, changes)

            # x_0_logits = self._denoise_fn(x_t, t=t)
            x_t_emb = self.image_emb(x_t)
            x_0_out = self.transformer_forward(torch.cat((control_emb, x_t_emb + target_pos_emb), dim = 1))
            x_0_logits = self.to_logits(x_0_out[:,control_seq_len:,:])

            # scale by temperature
            x_0_logits = x_0_logits / temp
            x_0_dist = dists.Categorical(logits=x_0_logits)
            x_0_hat = x_0_dist.sample().long()
            x_t[changes] = x_0_hat[changes]

        # Mask-Predict
        N = self.target_seq_len
        B = 3
        # Tmax = 25
        # n = list(N * np.linspace(0.8, 0.2, 10)) + list(max(int(N*0.1),1)*np.ones(10)) + list(max(int(N*0.05),1)*np.ones(5))
        Tmax = 20
        n = list(max(int(N*0.1),1)*np.ones(10)) + list(max(int(N*0.05),1)*np.ones(10))
        n = list(map(int, n))

        sample_toks = []
        target_pos_emb = target_pos_emb[0:1,...]
        mask_emb = self.image_emb.weight[self.image_token_lut['[MASK]']]

        for i in range(control_emb.shape[0]):
            control_emb_ = control_emb[i:i+1,...]
            tok_in = x_t[i:i+1,...]            
            emb_in = self.image_emb(tok_in)
            tokens = torch.cat((control_emb_, emb_in + target_pos_emb), dim = 1)
            out = self.transformer_forward(tokens)[:,control_seq_len:,:]
            logits = self.to_logits(out)  # b n c
            Y, I_new = sample_multinomial(logits, 0)
            I_tok = I_new

            if debug:
                # print('PNAG:')
                image_samples.append(self.decode_images(I_tok))

            Smax = 0
            tmax = 0
            Imax = None
            for t in range(1, Tmax):
                # Mask: sample B seqs [I_in] (masked sequences) according to Y
                emb_in = []
                masks1 = []
                for j in range(B):
                    mask1_idx = torch.multinomial(Y, self.target_seq_len-n[t-1], replacement=False)
                    mask1 = torch.zeros(1, self.target_seq_len, device=device).scatter_(1, mask1_idx, 1)
                    mask1 = mask1 == 1
                    masks1.append(mask1)
                    emb_out = self.image_emb(I_tok)
                    emb_masked = torch.where(mask1.unsqueeze(2), emb_out, mask_emb)
                    emb_in.append(emb_masked)

                # Predict: predict I_out and select b with highest score; update Y
                S = torch.zeros(B)
                S_rel = torch.zeros(B)
                S_vid = torch.zeros(B)
                YB, tokB = [], []
                for j in range(B):
                    tokens = torch.cat((control_emb_, emb_in[j] + target_pos_emb), dim = 1)
                    out = self.transformer_forward(tokens)
                    logits = self.to_logits(out[:,control_seq_len:,:])  # b n c
                    Y_new, I_new = sample_multinomial(logits, 0)  # b n
                    mask1_j = masks1[j]
                    Y = torch.where(mask1_j, Y, Y_new)
                    I_tok = torch.where(mask1_j, I_tok, I_new)
                    S_rel[j] = torch.sigmoid(self.to_logits_rel(out[:,self.rel_tok_index,:]))
                    S_vid[j] = torch.sigmoid(self.to_logits_vid(out[:,self.vid_tok_index,:]))
                    S[j] = S_rel[j]*0.5 + S_vid[j]*0.5
                    YB.append(Y)
                    tokB.append(I_tok)
                jmax = S.argmax()
                Y, I_tok = YB[jmax], tokB[jmax]

                if debug:
                    # print(f'-> t = {t}, rel = {S_rel}, vid = {S_vid}, avg = {S_rel*0.5+S_vid*0.5}')
                    mask_img = self.decode_masks((~masks1[jmax]).float())
                    masked_img = image_samples[-1]
                    masked_img = torch.clamp(masked_img*0.7 + mask_img*0.4, 0, 1)
                    image_samples.append(masked_img)
                    image_samples.append(self.decode_images(I_tok))
                if dynamic:
                    if S[jmax] > Smax:
                        tmax = t
                        Smax = S[jmax]
                        Imax = I_tok
                    if t - tmax >= 5:  # dynamic termination
                        # print(f'early stopping at {tmax}!!!')
                        break
                else:
                    Imax = I_tok
            sample_toks.append(Imax)
        sample_toks = torch.cat(sample_toks, 0)

        return sample_toks, image_samples, None

    def get_image_tokens(self, image, reshape=True, insert_sep=False, which_vae='vae'):
        if which_vae == 'cvae' and self.cvae is not None:
            vae = self.cvae
        else:
            vae = self.vae
        if isinstance(image, list):
            assert len(image[0].shape) == 4, 'image should be list of 4d image tensors'
            image = torch.stack(image, dim=1)
        if len(image.shape) == 4:
            image = image.unsqueeze(1)
        is_raw_image = len(image.shape) == 5  # [B, T, C, H, W]
        if is_raw_image:
            b, t, c, h, w = image.shape
            image_size = vae.image_size
            assert (c, h, w) == (3, image_size, image_size), f'invalid image of dimensions {image.shape} passed in during training'
            image = rearrange(image, 'b t c h w -> (b t) c h w')
            image = vae.get_codebook_indices(image)  # ((b t) n)
            if reshape:
                if insert_sep:
                    image = rearrange(image, '(b t) n -> b t n', t = t)
                    image = torch.cat((image, torch.empty(b, t, 1, device=image.device).long().fill_(self.image_token_lut['[SEP]'])), dim = 2)
                    image = rearrange(image, 'b t n -> b (t n)')
                else:
                    image = rearrange(image, '(b t) n -> b (t n)', t = t)
        return image

    @torch.no_grad()
    def recon_images(self, images, which_vae='vae'):
        if which_vae == 'cvae' and self.cvae is not None:
            vae = self.cvae
        else:
            vae = self.vae
        img_seq = self.get_image_tokens(images, reshape=False, which_vae=which_vae)
        images = vae.decode(img_seq)
        # images = rearrange(images, '(b t) c h w -> b t c h w', t = self.num_targets)
        return images
    
    @torch.no_grad()
    def get_codebook_emb(self, images, which_vae='vae'):
        b, t, c, h, w = images.shape
        if which_vae == 'cvae' and self.cvae is not None:
            vae = self.cvae
        else:
            vae = self.vae
        img_seq = self.get_image_tokens(images, reshape=False, which_vae=which_vae)
        img_code = rearrange(img_seq, '(b t) n -> b t n', t = t)
        img_embd = self.image_emb(img_code)
        return img_code, img_embd
    
    @torch.no_grad()
    def update_memory_bank(self, sample):
        batch_size_t = sample.shape[0]
        ptr = int(self.q_ptr)
        self.q[ptr:ptr + batch_size_t, :] = sample
        ptr = (ptr + batch_size_t) % self.q_len
        self.q_ptr[0] = ptr
    
    def random_erase_codebook(self, image, eraser, erase_half=False):
        image = rearrange(image, 'b (t h w) -> b t h w', h = self.image_fmap_size, w = self.image_fmap_size)
        if erase_half:
            image_ = image
            image_[:,:,self.image_fmap_size//2:,:] = self.image_token_lut['[MASK]']
        else:
            image_ = torch.stack([eraser(c) for c in image], dim = 0)
        image = rearrange(image_, 'b t h w -> b (t h w)', h = self.image_fmap_size, w = self.image_fmap_size)
        return image
    
    def erase_codebook_face(self, image, vc_mode, face_mode=None):
        image = rearrange(image, 'b (t h w) -> b t h w', h = self.image_fmap_size, w = self.image_fmap_size)
        if vc_mode == 'face_8x8':
            image_ = self.image_token_lut['[MASK]'] + torch.zeros_like(image).long()
            if face_mode is None:
                face_mode = 'eyes_nose' if random.random() < 0.5 else 'mouth'
            if face_mode == 'eyes_nose':  # eyes and nose
                image_[:,:,2:5,1:7] = image[:,:,2:5,1:7]
            else:  # mouth
                image_[:,:,5:7,2:6] = image[:,:,5:7,2:6]
            image = image_
        elif vc_mode == 'face2_8x8':
            image_ = self.image_token_lut['[MASK]'] + torch.zeros_like(image).long()
            image_[:,0,...] = image[:,0,...]
            image_[:,1:,2:6,2:6] = image[:,1:,2:6,2:6]
            image = image_
        elif vc_mode == 'face3_8x8':
            image_ = self.image_token_lut['[MASK]'] + torch.zeros_like(image).long()
            image_[:,0,...] = image[:,0,...]
            image_[:,:,2:6,2:6] = image[:,:,2:6,2:6]
            image = image_
        elif vc_mode == 'mask_8x8' or vc_mode == 'mask2_8x8':
            if face_mode is None:
                which_strategy = np.random.choice([1, 2, 3], p=[0.5, 0.25, 0.25])
            else:
                which_strategy = 3
            if which_strategy == 1:
                image_ = image
            elif which_strategy == 2:
                image_ = self.image_token_lut['[MASK]'] + torch.zeros_like(image).long()
                image_[:,:,2:6,2:6] = image[:,:,2:6,2:6]
            elif which_strategy == 3:
                image_ = self.image_token_lut['[MASK]'] + torch.zeros_like(image).long()
                image_[:,:,1:7,1:7] = image[:,:,1:7,1:7]
            image = image_
        elif vc_mode == 'shape_4x4':
            image[:,:,1:3,1:3] = self.image_token_lut['[MASK]']
        else:
            raise NotImplementedError
        image = rearrange(image, 'b t h w -> b (t h w)', h = self.image_fmap_size, w = self.image_fmap_size)
        return image

    def get_special_token(self, tok_list, batch_size=1, device='cuda'):
        tok = torch.tensor(tok_list, dtype=torch.long, device=device)
        return tok.repeat(batch_size, 1)
    
    def swap_one_frame_along_batch(self, tokens, t=1):
        tokens_shuffled = tokens.detach().clone()
        # if len(tokens.shape) == 4:
        #     b, t, l, c = tokens.shape
        # else:
        b, n, c = tokens.shape
        tokens_shuffled = tokens_shuffled.reshape(b, t, n // t, -1)
        idx = np.random.randint(0, t, b)
        # perm_idx = randperm(b)
        # frames_shuffled = tokens_shuffled[range(b),idx,...][perm_idx,...]
        frames_shuffled = torch.cat(torch.chunk(tokens_shuffled[range(b),idx,...], 2, dim=0)[::-1], dim = 0)
        tokens_shuffled[range(b),idx,...] = frames_shuffled
        tokens_shuffled = tokens_shuffled.reshape(b, n, c)
        return tokens_shuffled

    def sample_time(self, b, device, method='uniform'):
        # num_timesteps = self.target_seq_len  # TODO: T is always D during training
        num_time_steps = self.num_timesteps

        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self.sample_time(b, device, method='uniform')

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = Lt_sqrt / Lt_sqrt.sum()
            t = torch.multinomial(pt_all, num_samples=b, replacement=True)
            pt = pt_all.gather(dim=0, index=t)
            return t, pt

        elif method == 'uniform':
            t = torch.randint(1, self.num_timesteps+1, (b,), device=device).long()
            pt = torch.ones_like(t).float() / self.num_timesteps
            return t, pt
        
        # ========== MaskGIT style training ==========
        elif method == 'cosine':
            r = torch.rand(b, device=device)  # [0, 1]
            gamma_r = torch.cos(r * np.pi / 2)  # [0, 1]
            t = torch.round(gamma_r * (self.num_timesteps - 1) + 1).long()
            pr = torch.ones_like(t).float() / self.num_timesteps  # TODO: pt is not used for MaskGIT
            return t, pr
        
        elif method == 'linear':
            r = torch.rand(b, device=device)  # [0, 1]
            gamma_r = 1 - r  # [0, 1]
            t = torch.round(gamma_r * (self.num_timesteps - 1) + 1).long()
            pr = torch.ones_like(t).float() / self.num_timesteps  # TODO: pt is not used for MaskGIT
            return t, pr
        
        elif method == 'square':
            r = torch.rand(b, device=device)  # [0, 1]
            gamma_r = 1 - torch.pow(r, 2)  # [0, 1]
            t = torch.round(gamma_r * (self.num_timesteps - 1) + 1).long()
            pr = torch.ones_like(t).float() / self.num_timesteps  # TODO: pt is not used for MaskGIT
            return t, pr
        
        elif method == 'cubic':
            r = torch.rand(b, device=device)  # [0, 1]
            gamma_r = 1 - torch.pow(r, 3)  # [0, 1]
            t = torch.round(gamma_r * (self.num_timesteps - 1) + 1).long()
            pr = torch.ones_like(t).float() / self.num_timesteps  # TODO: pt is not used for MaskGIT
            return t, pr
        
        elif method == 'square_root':
            r = torch.rand(b, device=device)  # [0, 1]
            gamma_r = 1 - torch.sqrt(r)  # [0, 1]
            t = torch.round(gamma_r * (self.num_timesteps - 1) + 1).long()
            pr = torch.ones_like(t).float() / self.num_timesteps  # TODO: pt is not used for MaskGIT
            return t, pr
        
        elif method == 'square_root_one_minus':
            r = torch.rand(b, device=device)  # [0, 1]
            gamma_r = torch.sqrt(1 - r)  # [0, 1]
            t = torch.round(gamma_r * (self.num_timesteps - 1) + 1).long()
            pr = torch.ones_like(t).float() / self.num_timesteps  # TODO: pt is not used for MaskGIT
            return t, pr
        
        elif method.startswith('range_'):
            lb = float(method[6:].split('_')[0])
            ub = float(method[6:].split('_')[1])
            r = torch.rand(b, device=device)  # [0, 1]
            gamma_r = lb + (ub - lb) * r
            t = torch.round(gamma_r * (self.num_timesteps - 1) + 1).long()
            pr = torch.ones_like(t).float() / self.num_timesteps  # TODO: pt is not used for MaskGIT
            return t, pr
        
        # ========== MaskGIT style training ==========

        else:
            raise ValueError

    def q_sample(self, x_0, t):
        # samples q(x_t | x_0)
        # randomly set token to mask with probability t/T
        # mask == True are the tokens that are masked
        x_t, x_0_ignore = x_0.clone(), x_0.clone()

        mask = torch.rand_like(x_t.float()) < (t.float().unsqueeze(-1) / self.num_timesteps)
        x_t[mask] = self.mask_id
        x_0_ignore[torch.bitwise_not(mask)] = -1
        return x_t, x_0_ignore, mask
    
    def q_sample_mlm(self, x_0, t):
        # samples q(x_t | x_0)
        # fixed noise schedule, masks exactly int(t/T * latent_size) tokens
        x_t, x_0_ignore = x_0.clone(), x_0.clone()

        mask = torch.zeros_like(x_t).to(torch.bool)

        # TODO: offset so each n_masked_tokens is picked with equal probability
        n_masked_tokens = (t.float() / self.num_timesteps) * x_t.size(1)
        n_masked_tokens = torch.round(n_masked_tokens).to(torch.int64)
        n_masked_tokens[n_masked_tokens == 0] = 1
        ones = torch.ones_like(mask[0]).to(torch.bool).to(x_0.device)

        for idx, n_tokens_to_mask in enumerate(n_masked_tokens):
            index = torch.randperm(x_0.size(1))[:n_tokens_to_mask].to(x_0.device)
            mask[idx].scatter_(dim=0, index=index, src=ones)

        x_t[mask] = self.mask_id
        x_0_ignore[torch.bitwise_not(mask)] = -1
        return x_t, x_0_ignore, mask
    
    def compute_mixture_of_binomial_weight(self, T, t):
        return None

    # ======================= forward ==========================
    def forward(
        self,
        text,
        visual = None,
        target = None,
        mask = None,
        return_loss = False,
        return_fake = False,
        t_cond = None,
        rel = True,
        fdl = True,
        vid = True,
        erase_visual = True,
        erase_visual_half = False,
        oldrel = True,
        newmask = False,
        msm_strategy_prob = [0.7, 0.1, 0.1, 0.1],
        msm_bernoulli_prob = [0.2, 0.5],
        relvid_bernoulli_prob = [0.1, 0.9],
        rel_no_fully_masked = False,
        vid_strategy_prob = [0.25, 0.25, 0.25, 0.25],
        negvc = False,
        posvc = False,
        visual_neg = None,
        text_neg = None,
        visual_pos = None,
        text_pos = None,
        static_image_as_neg = False,
        aug_static_image_as_neg = False,
        pc_prob = 0,
        vc_mode = None,
        face_mode = None,
        visual_aug_mode = None,
        time_schedule = 'uniform',
        mask_schedule = 'random',
        loss_type = 'reweighted_elbo',
    ):
        # visual and target are lists or 5d tensors (B, T, C, H, W)

        device, total_seq_len = text[0].device, self.total_seq_len
        # text_shape = [len(text), self.text_seq_len]
        if self.fixed_language_model is None:
            text_shape = text.shape
        else:
            text_shape = [text.shape[0], 1]  # TODO: use embedding which takes a single token
        batch_size = text_shape[0]
        # half_batch_size = batch_size // 2

        # Prepend [REL] [FDL]

        before_tok = self.get_special_token(self.before_control_tok, batch_size, device)
        before_emb = self.special_emb(before_tok)
        before_emb += self.special_pos_emb(before_tok)
        control_emb = before_emb
        control_seq_len = before_emb.shape[1]
        if negvc:
            control_neg_emb = before_emb
        if posvc:
            control_emb_pos = before_emb

        # make sure padding in text tokens get unique padding token id

        text_feature_before, text_feature_after = None, None
        if self.fixed_language_model is None:
            assert text.shape[-1] == self.text_seq_len, f'the length {text.shape[-1]} of the text tokens you passed in does not have the correct length ({self.text_seq_len})'
            text_range = torch.arange(self.text_seq_len, device = device) + (self.num_text_tokens - self.text_seq_len)
            text = torch.where(text == 0, text_range, text)
            text_emb = self.text_emb(text)
            text_emb += self.text_pos_emb(torch.arange(text_shape[1], device = device))
        else:
            # text is a single embedding
            text_emb = self.text_feature_mapping(text)
            if self.clip_text_emb == 'before':
                text_feature_before = self.to_logits_txt(text_emb)
            text_emb = text_emb.unsqueeze(1)

        control_emb = torch.cat((control_emb, text_emb), dim = 1)
        control_seq_len += text_emb.shape[1]
        if negvc:
            # control_neg_emb = torch.cat((control_neg_emb, text_emb), dim = 1)  # TODO: current text_neg does not guarantee to be neg
            text_neg = torch.where(text_neg == 0, text_range, text_neg)
            text_neg_emb = self.text_emb(text_neg)
            text_neg_emb += self.text_pos_emb(torch.arange(text_shape[1], device = device))
            control_neg_emb = torch.cat((control_neg_emb, text_neg_emb), dim = 1)
        if posvc:
            if text_pos is None:
                text_emb_pos = text_emb
            else:
                text_pos = torch.where(text_pos == 0, text_range, text_pos)
                text_emb_pos = self.text_emb(text_pos)
                text_emb_pos += self.text_pos_emb(torch.arange(text_shape[1], device = device))
            control_emb_pos = torch.cat((control_emb_pos, text_emb_pos), dim = 1)

        visual_emb = None
        if self.num_visuals > 0:
            if exists(visual) and len(visual):
                if visual_aug_mode == 'motion_color' and random.random() < 0.9:
                    visual_ = visual.detach().clone()
                    visual_[:,1:,...] = warp_video_with_color(visual[:,1:,...])
                    visual = visual_
                visual = self.get_image_tokens(visual, insert_sep = self.insert_sep, which_vae = 'cvae')
                if erase_visual:
                    visual = self.random_erase_codebook(visual, self.visual_eraser, erase_visual_half)
                if vc_mode is not None:
                    visual = self.erase_codebook_face(visual, vc_mode, face_mode)
            else:
                visual = torch.empty(batch_size, self.visual_seq_len, device = device).long().fill_(self.image_token_lut['[MASK]'])
            visual_emb = self.visual_emb(visual) if self.visual_emb else self.image_emb(visual)

            visual_pos_emb = self.visual_pos_emb(visual_emb)
            visual_emb += visual_pos_emb
            # tokens = torch.cat((tokens, visual_emb), dim = 1)
            control_emb = torch.cat((control_emb, visual_emb), dim = 1)
            control_seq_len += visual.shape[1]
        
        if negvc and exists(visual_neg) and len(visual_neg):
            visual_neg = self.get_image_tokens(visual_neg, insert_sep = self.insert_sep, which_vae = 'cvae')

            # TODO: hard coded for color+shape+bg
            mask_neg = torch.zeros(batch_size, self.num_visuals, device=device)
            nn = self.num_visuals-1
            for i in range(batch_size):
                ss = np.binary_repr(random.randint(1, 2**nn-1), width=nn)
                for j in range(nn):
                    mask_neg[i,j] = ss[j]=='1'
            mask_neg = torch.repeat_interleave(mask_neg, self.image_seq_len+self.insert_sep, 1)
            visual_neg = torch.where(mask_neg==1, visual_neg, visual)
            visual_neg_emb = self.visual_emb(visual_neg) if self.visual_emb else self.image_emb(visual_neg)
            visual_neg_emb += visual_pos_emb
            control_neg_emb = torch.cat((control_neg_emb, visual_neg_emb), dim = 1)
        if posvc and exists(visual_pos) and len(visual_pos):
            visual_pos = self.get_image_tokens(visual_pos, insert_sep = self.insert_sep, which_vae = 'cvae')
            visual_emb_pos = self.visual_emb(visual_pos) if self.visual_emb else self.image_emb(visual_pos)
            visual_emb_pos += visual_pos_emb
            control_emb_pos = torch.cat((control_emb_pos, visual_emb_pos), dim = 1)

        # Append [VID]
        after_tok = self.get_special_token(self.after_control_tok, batch_size, device)
        after_emb = self.special_emb(after_tok)
        after_emb += self.special_pos_emb(after_tok)
        control_emb = torch.cat((control_emb, after_emb), dim = 1)
        control_seq_len += after_emb.shape[1]
        if negvc:
            control_neg_emb = torch.cat((control_neg_emb, after_emb), dim = 1)
        if posvc:
            control_emb_pos = torch.cat((control_emb_pos, after_emb), dim = 1)

        # TODO: add summary embedding

        if not return_loss:
            return control_emb

        target_emb = None
        target_orig = None
        if exists(target) and len(target):
            target_orig = target.detach().clone()
            target = self.get_image_tokens(target)
            # target_emb = self.image_emb(target)
            # target_pos_emb = self.target_pos_emb(target_emb)

        # assert tokens.shape[1] == total_seq_len

        #============== Masked Language Modeling ==============#
        x_0 = target
        b, device = x_0.size(0), x_0.device

        # choose what time steps to compute loss at
        if time_schedule == 'uniform':
            t, pt = self.sample_time(b, device, 'uniform')
        elif time_schedule.startswith('gamma_'):
            t, pt = self.sample_time(b, device, time_schedule[6:])
        else:
            raise ValueError('Unrecognized time schedule: {}'.format(time_schedule))

        if mask_schedule == 'random':
            x_t, x_0_ignore, mask = self.q_sample(x_0=x_0, t=t)
        elif mask_schedule == 'fixed':
            x_t, x_0_ignore, mask = self.q_sample_mlm(x_0=x_0, t=t)
        
        # sample p(x_0 | x_t), denoise x_t
        target_emb_masked = self.image_emb(x_t)
        time_emb = self.time_emb(t.view(-1, 1)) if self.use_time_embedding else 0
        target_pos_emb = self.target_pos_emb(target_emb_masked) + time_emb  # TODO
        in_mlm = torch.cat((control_emb, target_emb_masked + target_pos_emb), dim = 1)
        out_mlm = self.transformer_forward(in_mlm)
        x_0_hat_logits = self.to_logits(out_mlm[:,control_seq_len:,:]).permute(0, 2, 1)

        cross_entropy_loss = F.cross_entropy(x_0_hat_logits, x_0_ignore, ignore_index=-1, reduction='none').sum(1)

        loss = cross_entropy_loss / (math.log(2) * x_0.shape[1:].numel())
        if loss_type == 'elbo':
            loss = loss / t / pt  # if uniform, 1 / t / pt is in [1, T]
        elif loss_type == 'reweighted_t_inverse':
            loss = loss / t  # if uniform, 1 / t is in [1/T, 1]
        elif loss_type == 'reweighted_t':
            weight = (t + 1) / self.num_timesteps
            loss = weight * loss
        elif loss_type == 'reweighted_elbo':
            weight = torch.clamp(1 - (t / self.num_timesteps), min=1/self.num_timesteps)
            loss = weight * loss
        elif loss_type == 'simple':
            loss = loss * 1
        elif loss_type == 'mixture_of_binomial':
            raise NotImplementedError
        else:
            raise ValueError('Unrecognized loss type: {}'.format(loss_type))

        loss_msm = loss.mean()
        # print(f'absorbing loss: {loss.mean().item():.4f}')

        # Track mean cross entropy loss at each time step history for bar plot
        denom = mask.float().sum(1)
        denom[denom == 0] = 1  # prevent divide by 0 errors.
        mce_loss = cross_entropy_loss / denom
        mce_prev = self.mce_history.gather(dim=0, index=t)
        new_mce_history = (0.1 * mce_loss + 0.9 * mce_prev).detach().to(self.mce_history.dtype)
        self.mce_history.scatter_(dim=0, index=t, src=new_mce_history)

        # Track loss at each time step for importance sampling
        vb_loss = cross_entropy_loss / t
        vb_loss = vb_loss / pt
        vb_loss = vb_loss / (math.log(2) * x_0.shape[1:].numel())
        Lt2 = vb_loss.detach().clone().pow(2)
        Lt2_prev = self.Lt_history.gather(dim=0, index=t)
        new_Lt_history = (0.1 * Lt2 + 0.9 * Lt2_prev).detach().to(self.Lt_history.dtype)
        self.Lt_history.scatter_(dim=0, index=t, src=new_Lt_history)
        self.Lt_count.scatter_add_(dim=0, index=t, src=torch.ones_like(Lt2).to(self.Lt_history.dtype))

        # to be compatible with the original code
        rel_no_fully_masked = False
        not_fully_masked = None
        out = out_mlm
        logits_msm = x_0_hat_logits.permute(0, 2, 1)
        mask1 = torch.bitwise_not(mask)  # mask1 is the mask for the non-masked tokens
        #=====================================================#

        target_prob = torch.zeros(batch_size, self.target_seq_len, self.num_image_tokens, device=device).scatter(2, target.unsqueeze(2), 1)
        if return_fake:
            sample_prob = F.gumbel_softmax(logits_msm, tau=1., hard=True, dim=2)
            sample_prob = torch.where(mask1.unsqueeze(2), target_prob, sample_prob)
            sample_prob = rearrange(sample_prob, 'b (t n) c -> (b t) n c', n = self.image_seq_len)
            fake_sample = self.vae.decode_train(sample_prob)
            fake_sample = rearrange(fake_sample, '(b t) c h w -> b t c h w', t = self.num_targets)

        # Relevance Estimation Task

        if rel:
            # assert text_shape[0] >= 2 and text_shape[0] % 2 == 0  # for REL swapping
            control_swapped, frame_swapped = False, False
            if negvc:
                tokens_neg_rel = torch.cat((control_neg_emb, target_emb_masked + target_pos_emb), dim = 1)
                out_neg_rel = self.transformer_forward(tokens_neg_rel)
                logits_pos_rel = self.to_logits_rel(out[:,self.rel_tok_index,:]).squeeze()
                logits_neg_rel = self.to_logits_rel(out_neg_rel[:,self.rel_tok_index,:]).squeeze()
            else:
                control_emb_swap = swap(control_emb, 0)
                tokens_neg_rel = torch.cat((control_emb_swap, target_emb_masked + target_pos_emb), dim = 1)
                out_neg_rel = self.transformer_forward(tokens_neg_rel)
                logits_pos_rel = self.to_logits_rel(out[:,self.rel_tok_index,:]).squeeze()
                logits_neg_rel = self.to_logits_rel(out_neg_rel[:,self.rel_tok_index,:]).squeeze()
            weight_pos = 0.5 if posvc else 1
            if rel_no_fully_masked:
                loss_rel_pos = F.binary_cross_entropy_with_logits(logits_pos_rel, torch.ones(batch_size, device=device), reduction='none') * weight_pos
                loss_rel_neg = F.binary_cross_entropy_with_logits(logits_neg_rel, torch.zeros(batch_size, device=device), reduction='none')
                loss_rel = (loss_rel_pos * not_fully_masked + loss_rel_neg * not_fully_masked).sum() / max(1., not_fully_masked.sum())
            else:
                loss_rel = (F.binary_cross_entropy_with_logits(logits_pos_rel, torch.ones(batch_size, device=device)) * weight_pos
                    + F.binary_cross_entropy_with_logits(logits_neg_rel, torch.zeros(batch_size, device=device)))
            if posvc:
                tokens_rel_pos = torch.cat((control_emb_pos, target_emb_masked + target_pos_emb), dim = 1)
                out_rel_pos = self.transformer_forward(tokens_rel_pos)
                logits_rel_pos = self.to_logits_rel(out_rel_pos[:,self.rel_tok_index,:]).squeeze()
                if rel_no_fully_masked:
                    loss_rel_pos = F.binary_cross_entropy_with_logits(logits_rel_pos, torch.ones(batch_size, device=device), reduction='none')
                    loss_rel_pos = loss_rel_pos.sum() / max(1., not_fully_masked.sum())
                else:
                    loss_rel_pos = F.binary_cross_entropy_with_logits(logits_rel_pos, torch.ones(batch_size, device=device))
                loss_rel += loss_rel_pos * weight_pos
        else:
            loss_rel = torch.tensor(0.0, device=device)

        # Fidelity Estimation Task

        if fdl and self.use_time_token:  # TODO: FDL is reused for time token
            logit_time = self.to_logits_fdl(out[:,self.time_tok_index,:])
            label_time = t.clone()
            loss_fdl = F.cross_entropy(logit_time, label_time)
        else:
            loss_fdl = torch.tensor(0.0, device=device)

        # Continuity Estimation Task
        if vid and self.num_targets > 1:
            weight_pos = 1
            weight_neg = 0.5 / (static_image_as_neg + aug_static_image_as_neg) if static_image_as_neg else 1
            # get warped frames
            target_warp = warp(target_orig, vid_strategy_prob)
            target_warp = self.get_image_tokens(target_warp)
            target_warp_masked = torch.where(mask1, target_warp, self.image_token_lut['[MASK]'])
            target_emb_warp_masked = self.image_emb(target_warp_masked)
            # target_emb_warp_masked = swap_one_frame_along_batch(target_emb_masked, self.num_targets, shufflevid)
            tokens_neg_vid = torch.cat((control_emb, target_emb_warp_masked + target_pos_emb), dim = 1)
            out_neg_vid = self.transformer_forward(tokens_neg_vid)
            out_pos = out
            logits_pos_vid = self.to_logits_vid(out_pos[:,self.vid_tok_index,:])
            logits_neg_vid = self.to_logits_vid(out_neg_vid[:,self.vid_tok_index,:])
            if rel_no_fully_masked:
                loss_vid = (F.binary_cross_entropy_with_logits(logits_pos_vid, torch.ones(batch_size, 1, device=device), reduction='none').sum()/max(1., not_fully_masked.sum()) * weight_pos
                    + F.binary_cross_entropy_with_logits(logits_neg_vid, torch.zeros(batch_size, 1, device=device), reduction='none').sum()/max(1., not_fully_masked.sum()) * weight_neg)
            else:
                loss_vid = (F.binary_cross_entropy_with_logits(logits_pos_vid, torch.ones(batch_size, 1, device=device)) * weight_pos
                    + F.binary_cross_entropy_with_logits(logits_neg_vid, torch.zeros(batch_size, 1, device=device)) * weight_neg)
            if static_image_as_neg:
                frame_idx = random.randint(0, self.num_targets-1)
                image = target[:,frame_idx*self.image_seq_len:(frame_idx+1)*self.image_seq_len]
                image_static = image.repeat(1, self.num_targets)
                image_static_masked = torch.where(mask1, image_static, self.image_token_lut['[MASK]'])
                image_static_emb_masked = self.image_emb(image_static_masked)
                tokens_neg_vid_static = torch.cat((control_emb, image_static_emb_masked + target_pos_emb), dim = 1)
                out_neg_vid_static = self.transformer_forward(tokens_neg_vid_static)
                logits_neg_vid_static = self.to_logits_vid(out_neg_vid_static[:,self.vid_tok_index,:])
                if rel_no_fully_masked:
                    loss_vid += F.binary_cross_entropy_with_logits(logits_neg_vid_static, torch.zeros(batch_size, 1, device=device), reduction='none').sum()/max(1., not_fully_masked.sum()) * weight_neg
                else:
                    loss_vid += F.binary_cross_entropy_with_logits(logits_neg_vid_static, torch.zeros(batch_size, 1, device=device)) * weight_neg
            if aug_static_image_as_neg:
                frame_idx = random.randint(0, self.num_targets-1)
                image_static_orig = target_orig[:,frame_idx,...].unsqueeze(1).repeat(1, self.num_targets, 1, 1, 1)
                image_static_warp = warp(image_static_orig, vid_strategy_prob)
                image_static_warp = self.get_image_tokens(image_static_warp)
                image_static_warp_masked = torch.where(mask1, image_static_warp, self.image_token_lut['[MASK]'])
                image_static_emb_warp_masked1 = self.image_emb(image_static_warp_masked)
                tokens_neg_vid = torch.cat((control_emb, image_static_emb_warp_masked1 + target_pos_emb), dim = 1)
                out_neg_vid_static = self.transformer_forward(tokens_neg_vid)
                logits_neg_vid_static = self.to_logits_vid(out_neg_vid_static[:,self.vid_tok_index,:])
                if rel_no_fully_masked:
                    loss_vid += F.binary_cross_entropy_with_logits(logits_neg_vid_static, torch.zeros(batch_size, 1, device=device), reduction='none').sum()/max(1., not_fully_masked.sum()) * weight_neg
                else:
                    loss_vid += F.binary_cross_entropy_with_logits(logits_neg_vid_static, torch.zeros(batch_size, 1, device=device)) * weight_neg
        else:
            loss_vid = torch.tensor(0.0, device=device)

        self.current_step += 1

        if return_fake:
            return loss_msm, loss_rel, loss_fdl, loss_vid, fake_sample, text_feature_before, text_feature_after
        return loss_msm, loss_rel, loss_fdl, loss_vid, None, text_feature_before, text_feature_after
