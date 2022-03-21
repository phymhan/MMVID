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


@torch.no_grad()
def extend_video(video, num=2):
    # video (n, t, c, h, w)
    video_ = [video]
    video_flipped = torch.flip(video, [1])
    for n in range(1, num):
        if n % 2 == 0:
            video_.append(video[:, 1:, ...])
        else:
            video_.append(video_flipped[:, 1:, ...])
    video_ = torch.cat(video_, dim=1)
    return video_


@torch.no_grad()
def evaluate(args,
             dalle_module,
             tokenizer,
             tokenizer2,
             language_model,
             dl_iter,
             metrics=['fvd', 'prd']):
    # NOTE: I used conda env py38, with cuda 11.2 (cuda 11.1 + cudnn-11.2-linux-x64-v8.1.0.77)
    LOG_DIR = Path(args.log_root) / args.name
    LOG_WEB_DIR = LOG_DIR / 'web'
    USE_HTML = True

    if USE_HTML:
        webpage = utils_html.initialize_webpage(LOG_WEB_DIR,
                                                'DALLE: ' + args.name + ' FVD',
                                                reverse=False)
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
        vid = tf.placeholder(
            tf.float32,
            [args.batch_size, VIDEO_LENGTH, IMAGE_SIZE, IMAGE_SIZE, 3],
            name='vid')
        emb_data1 = tf.placeholder(tf.float32, [None, 400], name='emb_data1')
        emb_data2 = tf.placeholder(tf.float32, [None, 400], name='emb_data2')

        emb = fvd.create_id3_embedding(fvd.preprocess(vid, (224, 224)))
        result = fvd.calculate_fvd(emb_data1, emb_data2)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())

            for _ in tqdm(range(TOTAL_NUM // args.batch_size)):
                text_neg, visuals_neg = None, None
                if args.negvc:
                    text, frames, visuals, visuals_neg, text_neg = next(
                        dl_iter)
                    visuals_neg, text_neg = map(lambda t: t.cuda(),
                                                (visuals_neg, text_neg))
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
                            'input_ids':
                            encoded_input['input_ids'].cuda(),
                            'attention_mask':
                            encoded_input['attention_mask'].cuda(),
                        }
                        model_output = language_model(**encoded_input)
                        text = mean_pooling(model_output,
                                            encoded_input['attention_mask'])
                else:
                    text = text.cuda()
                    text_description = []
                    for j in range(text.shape[0]):
                        sample_text = text[j:j + 1]
                        token_list = sample_text.masked_select(
                            sample_text != 0).tolist()
                        decoded_text = tokenizer.decode(token_list)
                        if isinstance(decoded_text, (list, tuple)):
                            decoded_text = decoded_text[0]
                        text_description += [decoded_text]

                if args.vc_mode == 'face_8x8':
                    face_mode = 'eyes_nose' if random.random(
                    ) < 0.5 else 'mouth'
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
                    samples_web.append(real_videos[0, :N_FRAME, ...])
                    captions_web += [f'real {0}']
                    samples_web.append(fake_videos[0, :N_FRAME, ...])
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
                    num = int(np.ceil((VIDEO_LENGTH - 1) / (n_frame - 1)))
                    real_videos = extend_video(real_videos, num)
                    fake_videos = extend_video(fake_videos, num)
                real_videos = real_videos[:, :VIDEO_LENGTH, ...]
                fake_videos = fake_videos[:, :VIDEO_LENGTH, ...]
                real_videos = (real_videos * 255).permute(
                    0, 1, 3, 4, 2).numpy()  # (n, t, h, w, 3)
                fake_videos = (fake_videos * 255).permute(
                    0, 1, 3, 4, 2).numpy()  # (n, t, h, w, 3)
                real_emb = sess.run(emb, feed_dict={vid: real_videos})
                fake_emb = sess.run(emb, feed_dict={vid: fake_videos})
                real_embs.append(real_emb)
                fake_embs.append(fake_emb)

            real_embs = np.concatenate(real_embs, axis=0)
            fake_embs = np.concatenate(fake_embs, axis=0)

            np.save(str(OUTPUT_DIR / 'real_embs.npy'), real_embs)
            np.save(str(OUTPUT_DIR / 'fake_embs.npy'), fake_embs)

            # FVD
            score = sess.run(result,
                             feed_dict={
                                 emb_data1: real_embs,
                                 emb_data2: fake_embs
                             })

        print(f"FVD is: {score}")
        with open(OUTPUT_DIR / 'fvd_score.txt', 'w') as f:
            f.write(f"{score}")

        # PRD
        prd_data = prd.compute_prd_from_embedding(real_embs, fake_embs)
        with open(OUTPUT_DIR / 'prd_data.pkl', 'wb') as f:
            pickle.dump(prd_data, f)
        f_beta, f_beta_inv = prd.prd_to_max_f_beta_pair(
            prd_data[0], prd_data[1])
        print(f"f_beta: {f_beta}, f_beta_inv: {f_beta_inv}")
        with open(OUTPUT_DIR / 'prd_score.txt', 'w') as f:
            f.write(f"{f_beta}, {f_beta_inv}")
        print(f"FVD is: {score}")


@torch.no_grad()
def evaluate_intra(args,
                   dalle_module,
                   tokenizer,
                   tokenizer2,
                   language_model,
                   dl_iter,
                   metrics=['fvd', 'prd']):
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
        vid = tf.placeholder(
            tf.float32,
            [args.batch_size, VIDEO_LENGTH, IMAGE_SIZE, IMAGE_SIZE, 3],
            name='vid')
        emb_data1 = tf.placeholder(tf.float32, [None, 400], name='emb_data1')
        emb_data2 = tf.placeholder(tf.float32, [None, 400], name='emb_data2')

        emb = fvd.create_id3_embedding(fvd.preprocess(vid, (224, 224)))
        result = fvd.calculate_fvd(emb_data1, emb_data2)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        visuals = None

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())

            for _ in tqdm(range(TOTAL_NUM // args.batch_size)):
                frames_, text_ = next(dl_iter)
                frames_ = frames_.cuda()

                for cind, ccc in enumerate(args.cat1):
                    frames = frames_[:, cind, ...]
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
                                'input_ids':
                                encoded_input['input_ids'].cuda(),
                                'attention_mask':
                                encoded_input['attention_mask'].cuda(),
                            }
                            model_output = language_model(**encoded_input)
                            text = mean_pooling(
                                model_output, encoded_input['attention_mask'])
                    else:
                        text = text_[:, cind, ...]
                        text = text.cuda()
                        text_description = []
                        for j in range(text.shape[0]):
                            sample_text = text[j:j + 1]
                            token_list = sample_text.masked_select(
                                sample_text != 0).tolist()
                            decoded_text = tokenizer.decode(token_list)
                            if isinstance(decoded_text, (list, tuple)):
                                decoded_text = decoded_text[0]
                            text_description += [decoded_text]

                    if args.vc_mode == 'face_8x8':
                        face_mode = 'eyes_nose' if random.random(
                        ) < 0.5 else 'mouth'
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
                    clip_score = clip_similarity(clipper, clip_tokenizer,
                                                 sample_vc[:, 0, ...],
                                                 text_description)
                    clip_scores[ccc].append(clip_score)

                    if real_videos.shape[1] < VIDEO_LENGTH:
                        n_frame = real_videos.shape[1]
                        num = int(np.ceil((VIDEO_LENGTH - 1) / (n_frame - 1)))
                        real_videos = extend_video(real_videos, num)
                        fake_videos = extend_video(fake_videos, num)
                    real_videos = real_videos[:, :VIDEO_LENGTH, ...]
                    fake_videos = fake_videos[:, :VIDEO_LENGTH, ...]
                    real_videos = (real_videos * 255).permute(
                        0, 1, 3, 4, 2).numpy()  # (n, t, h, w, 3)
                    fake_videos = (fake_videos * 255).permute(
                        0, 1, 3, 4, 2).numpy()  # (n, t, h, w, 3)
                    # real_video_list.append(real_videos)
                    # fake_video_list.append(fake_videos)
                    real_emb = sess.run(emb, feed_dict={vid: real_videos})
                    fake_emb = sess.run(emb, feed_dict={vid: fake_videos})
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
                score = sess.run(result,
                                 feed_dict={
                                     emb_data1: real_embs1,
                                     emb_data2: fake_embs1
                                 })
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
            score = sess.run(result,
                             feed_dict={
                                 emb_data1: real_embs_all,
                                 emb_data2: fake_embs_all
                             })
            print(f"uncond FVD is: {score}")
            with open(OUTPUT_DIR / 'fvd_score.txt', 'w') as f:
                f.write(f"{score}")

            # PRD
            prd_data = prd.compute_prd_from_embedding(real_embs_all,
                                                      fake_embs_all)
            f_beta, f_beta_inv = prd.prd_to_max_f_beta_pair(
                prd_data[0], prd_data[1])
            print(f"f_beta: {f_beta}, f_beta_inv: {f_beta_inv}")
            with open(OUTPUT_DIR / 'prd_score.txt', 'w') as f:
                f.write(f"{f_beta}, {f_beta_inv}")

            print(f"mean Intra-FVD is: {np.array(scores).mean()}")
            with open(OUTPUT_DIR / 'intra_fvd_score.txt', 'w') as f:
                f.write(f"{scores}")


def normalize(x, eps=1e-10):
    return x * torch.rsqrt(torch.sum(x**2, dim=1, keepdim=True) + eps)


@torch.no_grad()
def evaluate_clip(args,
                  dalle_module,
                  tokenizer,
                  tokenizer2,
                  language_model,
                  dl_iter,
                  upper_bound=False):
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

    for _ in tqdm(range(TOTAL_NUM // batch_size)):
        if args.negvc:
            text, frames, visuals, visuals_neg, text_neg = next(dl_iter)
            visuals_neg, text_neg = map(lambda t: t.cuda(),
                                        (visuals_neg, text_neg))
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
                text = mean_pooling(model_output,
                                    encoded_input['attention_mask'])
        else:
            text = text[j:j + 1, ...].cuda()
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

        scores = clip_similarity(clipper, clip_tokenizer, images,
                                 text_description)
        results.append((text_description[0], scores))
        # cnt += batch_size

    suffix = '_real' if upper_bound else ''
    with open(OUTPUT_DIR / f'clip{suffix}_data.pkl', 'wb') as f:
        pickle.dump(results, f)

    scores = np.array([res[1] for res in results])
    scores = scores.max(axis=1)
    print(f"CLIP score: {scores.mean()}")
    with open(OUTPUT_DIR / f'clip{suffix}_score.txt', 'w') as f:
        f.write(f"{scores.mean()}, {scores.std()}")
