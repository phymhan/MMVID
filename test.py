import os
import shutil
from pathlib import Path
from tqdm import tqdm
import natsort

import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.nn as nn

from utils import utils
from utils import utils_html
from utils.utils_train import get_dataset, get_fixed_language_model, \
    get_vae_model, get_tokenizer
from utils.utils_train import visualize_test as visualize
from utils.utils_train import visualize_long
from utils.utils_eval import evaluate, evaluate_clip, evaluate_intra


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


def main():
    # argument parsing

    from utils.utils_args import process_args
    args = process_args()

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

    assert Path(args.image_text_folder).exists(
    ), f'The path {args.image_text_folder} was not found.'

    def is_root_worker():
        return True

    # constants
    # if args.no_debug:
    #     args.debug = False
    # args.truncate_captions = True
    # args.num_visuals *= args.visual
    # args.sample_every = 1

    args.deterministic = True  # NOTE: make everything deterministic

    if args.eval_mode == 'eval':
        args.batch_size = 16  # make samples reproducible

    # logging
    dalle_path = Path(args.dalle_path) if exists(args.dalle_path) else None
    if args.eval_image_folder:
        pass
    else:
        if args.dalle_path is None:
            checkpoints = natsort.natsorted(
                os.listdir(str(Path(args.log_root) / args.name / 'weights')))
            assert len(checkpoints) > 0, f'Nothing to resume from.'
            dalle_path = Path(
                args.log_root
            ) / args.name / 'weights' / checkpoints[-1] / 'dalle.pt'
    args.dalle_path = dalle_path

    args.name += args.name_suffix  # TODO: remove this

    resume = False
    log_dir = Path(args.log_root) / args.name
    log_sample_dir = log_dir / 'samples'
    args.log_dir = log_dir
    args.log_sample_dir = log_sample_dir

    assert args.dalle_path
    which_ckpt = str(dalle_path).split('/')[-2]

    args.log_metric_dir = log_dir / 'metrics' / which_ckpt
    os.makedirs(args.log_metric_dir, exist_ok=True)

    if is_root_worker():
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(log_dir / 'samples', exist_ok=True)
        os.makedirs(log_dir / 'weights', exist_ok=True)
        utils.print_args(None, args)
        if args.ar:
            shutil.copyfile('dalle_pytorch/dalle_artv.py',
                            log_dir / 'dalle_artv.py.txt')
        elif args.dm:
            shutil.copyfile('dalle_pytorch/dalle_absorb.py',
                            log_dir / 'dalle_absorb.py.txt')
        else:
            shutil.copyfile('dalle_pytorch/dalle_bert.py',
                            log_dir / 'dalle_bert.py.txt')

    webpage = None
    if args.use_html and is_root_worker():
        webpage = utils_html.initialize_webpage(log_dir / 'web',
                                                'DALLE: ' + args.name, resume)

    # tokenizer

    if args.fixed_language_model is not None:
        tokenizer2, language_model, text_feature_dim, encode_text = get_fixed_language_model(
            args)
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

    args.use_cvae = args.use_cvae or args.cvae_path is not None

    model_weights = None
    start_iter = 0

    # get vae model
    vae = None
    if not args.eval_image_folder:
        vae, _ = get_vae_model(args.which_vae,
                               vae_path=args.vae_path,
                               image_size=args.image_size)

    cvae = None
    if not args.eval_image_folder:
        if args.use_cvae:
            cvae, _ = get_vae_model(args.which_vae,
                                    vae_path=args.cvae_path,
                                    image_size=args.image_size)

    dalle_params = dict(
        num_text_tokens=tokenizer.vocab_size if tokenizer else 0,
        text_seq_len=args.text_seq_len,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        dim_head=args.dim_head,
        reversible=args.reversible,
        loss_img_weight=args.loss_img_weight,
        attn_types=args.attn_types,
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
        assert exists(dalle_path), 'DALLE model file does not exist'
        ckpt = torch.load(str(dalle_path))
        model_weights = ckpt['weights']

        image_size = args.image_size or vae.image_size
        args.image_size = vae.image_size = image_size
        if cvae is not None:
            cvae.image_size = image_size
    else:
        image_size = args.image_size

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
        batch_size=args.batch_size,
        shuffle=False,  #(data_sampler is None),
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

    pbar = range(args.iters)  # TODO
    if is_root_worker():
        pbar = tqdm(pbar,
                    initial=start_iter,
                    dynamic_ncols=True,
                    smoothing=0.01)

    for idx in pbar:
        i = idx + start_iter
        which_iter = f"{i:07d}"

        if i > args.iters:
            print('done!')
            break

        text_neg, visuals_neg = None, None
        if args.negvc:
            text, frames, visuals, visuals_neg, text_neg = next(dl_iter)
            visuals_neg, text_neg = map(lambda t: t.cuda(),
                                        (visuals_neg, text_neg))
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
                text = mean_pooling(model_output,
                                    encoded_input['attention_mask'])
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
