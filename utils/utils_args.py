import argparse
import numpy as np


def get_args_train():
    parser = argparse.ArgumentParser()

    parser.add_argument('--vae_path',
                        type=str,
                        help='path to your trained discrete VAE')
    parser.add_argument('--cvae_path',
                        type=str,
                        help='path to your trained discrete VAE')
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
                        default=1,
                        help='Random resized crop lower ratio')
    parser.add_argument('--which_tokenizer',
                        type=str,
                        default='simple',
                        help='(yttm | hug | simple | chinese)')
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
                        default=5000,
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
                        default=1e-4,
                        type=float,
                        help='Learning rate')
    parser.add_argument('--clip_grad_norm',
                        default=1.0,
                        type=float,
                        help='Clip gradient norm')
    # parser.add_argument('--lr_decay', dest='lr_decay', action='store_true')
    parser.add_argument('--no_lr_decay', action='store_true')
    parser.add_argument('--freeze_transformer', action='store_true')
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
                        default=5000,
                        help="sample every # iters")
    parser.add_argument('--n_sample',
                        default=4,
                        type=int,
                        help='Number of samples (text) to visualize')
    parser.add_argument('--n_per_sample',
                        default=4,
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
    # parser.add_argument('--epochs',
    #                     default=100,
    #                     type=int,
    #                     help='Number of epochs')
    parser.add_argument("--limit_train_batches", type=float, default=1)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument(
        '--keep_n_checkpoints',
        default=None,
        type=int,
        help=
        '(Careful) Deletes old deepspeed checkpoints if there are more than n')
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--lr_scheduler', type=str, default='warmuplr')
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
    parser.add_argument('--beta_msm', default=7.0, type=float)
    parser.add_argument('--beta_rel', default=0.5, type=float)
    parser.add_argument('--beta_vid', default=0.5, type=float)
    parser.add_argument('--beta_gan', default=0, type=float)
    parser.add_argument('--which_fake', type=str, default='mask_predict')
    parser.add_argument('--frame_num', default=8, type=int)
    parser.add_argument('--frame_step', default=4, type=int)
    parser.add_argument('--estimate_real', action='store_true')
    parser.add_argument('--pnag_argmax', action='store_true')
    parser.add_argument('--pnag_dynamic', action='store_true')
    parser.add_argument('--rand_visual', action='store_true')
    parser.add_argument('--fullvc', action='store_true')
    parser.add_argument('--negvc', action='store_true')
    parser.add_argument('--vc_mode', type=str, default=None)
    parser.add_argument('--attr_mode', type=str, default='object')
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

    parser.add_argument('--dim', default=768, type=int, help='Model dimension')
    parser.add_argument('--text_seq_len',
                        default=50,
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
    parser.add_argument('--pretrained_transformer',
                        type=str,
                        default='openai_clip_visual')
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

    return args, parser


def get_args_test():
    parser = argparse.ArgumentParser()

    parser.add_argument('--vae_path',
                        type=str,
                        help='path to your trained discrete VAE')
    parser.add_argument('--cvae_path',
                        type=str,
                        help='path to your trained discrete VAE')
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
    parser.add_argument('--pretrained_text_feature', type=str, default=None)
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
                        default=1,
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
    parser.add_argument('--name_suffix', default='', type=str)
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
                        default=5000,
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
    parser.add_argument('--freeze_transformer', action='store_true')
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
                        default=20,
                        type=int,
                        help='Number of iterations')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument(
        '--keep_n_checkpoints',
        default=None,
        type=int,
        help=
        '(Careful) Deletes old deepspeed checkpoints if there are more than n')
    parser.add_argument('--clip_ranking', action='store_true')
    parser.add_argument('--clip_path',
                        type=str,
                        default=None,
                        help='path to your pretrained CLIP')
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--which_fake', type=str, default='mask_predict')
    parser.add_argument('--frame_num', default=8, type=int)
    parser.add_argument('--frame_step', default=4, type=int)
    parser.add_argument('--estimate_real', action='store_true')
    parser.add_argument('--pnag_argmax', action='store_true')
    parser.add_argument('--pnag_dynamic', action='store_true')
    parser.add_argument('--rand_visual', action='store_true')
    parser.add_argument('--fullvc', action='store_true')
    parser.add_argument('--negvc', action='store_true')
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
    parser.add_argument('--mask_predict_steps_train', default=5, type=int)
    parser.add_argument('--use_pc', action='store_true')
    parser.add_argument('--pc_prob',
                        type=float,
                        default=0,
                        help='prob of preservation control')

    parser.add_argument('--dim', default=768, type=int, help='Model dimension')
    parser.add_argument('--text_seq_len',
                        default=50,
                        type=int,
                        help='Text sequence length')
    parser.add_argument('--depth', default=24, type=int, help='Model depth')
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
    parser.add_argument('--pretrained_transformer',
                        type=str,
                        default='openai_clip_visual')
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
    parser.add_argument('--text_emb_bottleneck', type=str, default=None)
    parser.add_argument('--fixed_language_model', type=str, default=None)
    parser.add_argument('--drop_sentence', action='store_true')
    parser.add_argument('--clip_text_emb', type=str, default=None)
    parser.add_argument('--beta_clip', default=0, type=float)

    parser.add_argument('--test_mode', type=str, default=None)
    parser.add_argument('--eval_mode', type=str, default=None)
    parser.add_argument('--eval_metric',
                        type=str,
                        nargs='+',
                        default=['fvd_prd'])
    parser.add_argument('--eval_num', type=int, default=2048)
    parser.add_argument('--vc_mode', type=str, default=None)
    parser.add_argument('--pc_mode', type=str, default=None)

    parser.add_argument('--description', type=str, default=None)
    parser.add_argument('--no_debug', action='store_true')
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--video_format', type=str, default='gif')

    parser.add_argument('--ar', action='store_true')

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
                        default=30,
                        help='number of total steps for mask-predict')
    parser.add_argument('--mp_B', type=int, default=1, help='beam search size')

    parser.add_argument('--eval_image_folder', action='store_true')
    parser.add_argument('--visual_aug_mode', type=str, default=None)

    # parser.add_argument('--rep_num', default = 1, type = int)
    parser.add_argument('--t_overlap', default=1, type=int)
    parser.add_argument('--t_repeat', default=10, type=int)
    parser.add_argument('--use_cvae', action='store_true')
    parser.add_argument('--eval_drop_sent', action='store_true')
    parser.add_argument('--save_codebook', action='store_true')
    parser.add_argument('--long_mode', type=str, default='long')
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

    return args, parser


def process_args(args):
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

    if args.msm_strategy_prob is not None:
        msm_strategy_prob = np.array(
            list(map(float, args.msm_strategy_prob.split(','))))
        msm_strategy_prob /= msm_strategy_prob.sum()
        args.msm_strategy_prob = msm_strategy_prob

    if args.vid_strategy_prob is not None:
        vid_strategy_prob = np.array(
            list(map(float, args.vid_strategy_prob.split(','))))
        vid_strategy_prob /= vid_strategy_prob.sum()
        args.vid_strategy_prob = vid_strategy_prob

    args.msm_bernoulli_prob = list(
        map(float, args.msm_bernoulli_prob.split(',')))

    args.relvid_bernoulli_prob = list(
        map(float, args.relvid_bernoulli_prob.split(',')))

    args.attn_types = tuple(args.attn_types.split(','))
    return args