# Multi-Modal Vox-Celeb Dataset
[folder](https://drive.google.com/drive/folders/18ebgGGTw0610_SRxiu5M3mdJCZqa-O74?usp=sharing)

[video](https://drive.google.com/file/d/1eG4CkNNqEuLz9LCa2XtesNepa9bsa1TP/view?usp=sharing)

[mask](https://drive.google.com/file/d/1Y36Or0pEnLQwn9uyORu9394_EcNpa3gl/view?usp=sharing)

[draw](https://drive.google.com/file/d/15UiX1KtyPPSagLjPhnEpm0ynG8PpMT8u/view?usp=sharing)

[text](https://drive.google.com/file/d/19e-9w-0-5FHwIXJ1CmHSKHli3jVMKkLu/view?usp=sharing)

[label](https://drive.google.com/file/d/1Eta6BrTTtV9vv1Hw05n3qo1uvH-3lB4t/view?usp=sharing)

[json](https://drive.google.com/file/d/1Q-ZxGfhNLlIC0X1cW2riBFZ6cz_3tcjy/view?usp=sharing)

# MMVID Code
## Multi-Modal Vox-Celeb
### Text-to-Video
Training:
``` 
bash scripts/mmvoxceleb/text_to_video/train.sh
```

Testing:
``` 
bash scripts/mmvoxceleb/text_to_video/test.sh
```

For Quantitative Evaluation (FVD and PRD):
```
bash scripts/mmvoxceleb/text_to_video/evaluation.sh
```

### Text and Mask
Training:
``` python
CUDA_VISIBLE_DEVICES=2,3 python3 train.py --name train_vox_text+mask --image_text_folder data/vox-celeba-alex_v2 --dataset vox --attr_mode mask+text_dropout --visual --vc_mode mask_8x8 --num_visuals 1 --fullvc --batch_size 20 --text_seq_len 50 --dim 768 --pretrained_transformer openai_clip_visual --iters 200000 --learning_rate 1e-4 --random_resize_crop_lower_ratio 1 --clip_grad_norm 1 --lr_decay --lr_scheduler warmuplr --optimizer adam --weight_decay 0.0 --which_tokenizer simple --use_html --log_every 200 --sample_every 5000 --n_sample 4 --n_per_sample 4 --num_targets 8 --frame_num 8 --frame_step 4 --which_vae vqgan1024 --image_size 128 --beta_rel 0.5 --beta_vid 0.5 --beta_msm 7 --log_root logs --lr_scheduler_warmup 5000 --msm_strategy_prob 7,1,1,1 --msm_bernoulli_prob 0.2,0.2 --vid_strategy_prob 1,1,1,1 --dropout_vc 0.4 --dist_url tcp://localhost:10002 --vae_path pretrained_models/vae_vox.ckpt --cvae_path pretrained_models/cvae_vox.ckpt --rel_no_fully_masked --mask_predict_steps 10 20 30 --mask_predict_steps1 20 
```

Testing:
``` python
CUDA_VISIBLE_DEVICES=0 python3 test.py --name test_vox_text+mask --image_text_folder data/vox-celeba-alex_v2 --dataset vox --attr_mode mask+text_dropout --visual --vc_mode mask_8x8 --num_visuals 1 --fullvc --text_seq_len 50 --dim 768 --pretrained_transformer openai_clip_visual --which_tokenizer simple --use_html --num_targets 8 --frame_num 8 --frame_step 4 --which_vae vqgan1024 --image_size 128 --log_root logs --use_cvae --iters 20 --batch_size 16 --n_per_sample 4 --n_sample 1 --no_debug --mp_T 20 --dalle_path vox_bert_text+mask_bs20_200k.pt
```

### Text and Drawing
Training:
``` python
CUDA_VISIBLE_DEVICES=4,5 python3 train.py --name train_vox_text+draw --image_text_folder data/vox-celeba-alex_v2 --dataset vox --attr_mode draw+text_dropout --visual --vc_mode mask_8x8 --num_visuals 1 --fullvc --batch_size 20 --text_seq_len 50 --dim 768 --pretrained_transformer openai_clip_visual --iters 200000 --learning_rate 1e-4 --random_resize_crop_lower_ratio 1 --clip_grad_norm 1 --lr_decay --lr_scheduler warmuplr --optimizer adam --weight_decay 0.0 --which_tokenizer simple --use_html --log_every 200 --sample_every 5000 --n_sample 4 --n_per_sample 4 --num_targets 8 --frame_num 8 --frame_step 4 --which_vae vqgan1024 --image_size 128 --beta_rel 0.5 --beta_vid 0.5 --beta_msm 7 --log_root logs --lr_scheduler_warmup 5000 --msm_strategy_prob 7,1,1,1 --msm_bernoulli_prob 0.2,0.2 --vid_strategy_prob 1,1,1,1 --dropout_vc 0.4 --dist_url tcp://localhost:10003 --vae_path pretrained_models/vae_vox.ckpt --cvae_path pretrained_models/cvae_vox.ckpt --rel_no_fully_masked --mask_predict_steps 10 20 30 --mask_predict_steps1 20 
```

Testing:
``` python
CUDA_VISIBLE_DEVICES=2 python3 test.py --name test_vox_text+draw --image_text_folder data/vox-celeba-alex_v2 --dataset vox --attr_mode draw+text_dropout --visual --vc_mode mask_8x8 --num_visuals 1 --fullvc --text_seq_len 50 --dim 768 --pretrained_transformer openai_clip_visual --which_tokenizer simple --use_html --num_targets 8 --frame_num 8 --frame_step 4 --which_vae vqgan1024 --image_size 128 --log_root logs --use_cvae --iters 20 --batch_size 16 --n_per_sample 4 --n_sample 1 --no_debug --mp_T 20 --dalle_path vox_bert_text+draw_bs20_200k.pt
```


### Drawing and Mask
Training:
``` python
CUDA_VISIBLE_DEVICES=6,7 python3 train.py --name train_vox_draw+mask --image_text_folder data/vox-celeba-alex_v2 --dataset vox --attr_mode draw+mask2 --visual --vc_mode mask2_8x8 --num_visuals 2 --fullvc --batch_size 20 --text_seq_len 20 --dim 768 --pretrained_transformer openai_clip_visual --iters 200000 --learning_rate 1e-4 --random_resize_crop_lower_ratio 1 --clip_grad_norm 1 --lr_decay --lr_scheduler warmuplr --optimizer adam --weight_decay 0.0 --which_tokenizer simple --use_html --log_every 200 --sample_every 5000 --n_sample 4 --n_per_sample 4 --num_targets 8 --frame_num 8 --frame_step 4 --which_vae vqgan1024 --image_size 128 --beta_rel 0.5 --beta_vid 0.5 --beta_msm 7 --log_root logs --lr_scheduler_warmup 5000 --msm_strategy_prob 7,1,1,1 --msm_bernoulli_prob 0.2,0.2 --vid_strategy_prob 1,1,1,1 --dropout_vc 0.4 --dist_url tcp://localhost:10004 --vae_path pretrained_models/vae_vox.ckpt --cvae_path pretrained_models/cvae_vox.ckpt --rel_no_fully_masked --mask_predict_steps 10 20 30 --mask_predict_steps1 20 
```

Testing:
``` python
CUDA_VISIBLE_DEVICES=0 python3 test.py --name test_vox_draw+mask --image_text_folder data/vox-celeba-alex_v2 --dataset vox --attr_mode draw+mask2 --visual --vc_mode mask2_8x8 --num_visuals 2 --fullvc --text_seq_len 20 --dim 768 --pretrained_transformer openai_clip_visual --which_tokenizer simple --use_html --num_targets 8 --frame_num 8 --frame_step 4 --which_vae vqgan1024 --image_size 128 --log_root logs --use_cvae --iters 20 --batch_size 16 --n_per_sample 4 --n_sample 1 --no_debug --mp_T 20 --dalle_path vox_bert_draw+mask_bs20_92k.pt
```


### Image and Mask
Training:
``` python
CUDA_VISIBLE_DEVICES=4,5 python3 train.py --name train_vox_image+mask --image_text_folder data/vox-celeba-alex_v2 --dataset vox --attr_mode image+mask2 --visual --vc_mode mask2_8x8 --num_visuals 2 --fullvc --batch_size 20 --text_seq_len 20 --dim 768 --pretrained_transformer openai_clip_visual --iters 200000 --learning_rate 1e-4 --random_resize_crop_lower_ratio 1 --clip_grad_norm 1 --lr_decay --lr_scheduler warmuplr --optimizer adam --weight_decay 0.0 --which_tokenizer simple --use_html --log_every 200 --sample_every 5000 --n_sample 4 --n_per_sample 4 --num_targets 8 --frame_num 8 --frame_step 4 --which_vae vqgan1024 --image_size 128 --beta_rel 0.5 --beta_vid 0.5 --beta_msm 7 --log_root logs --lr_scheduler_warmup 5000 --msm_strategy_prob 7,1,1,1 --msm_bernoulli_prob 0.2,0.2 --vid_strategy_prob 1,1,1,1 --dropout_vc 0.4 --dist_url tcp://localhost:10005 --vae_path pretrained_models/vae_vox.ckpt --cvae_path pretrained_models/cvae_vox.ckpt --rel_no_fully_masked --mask_predict_steps 10 20 30 --mask_predict_steps1 20 
```

Testing:
``` python
CUDA_VISIBLE_DEVICES=1 python3 test.py --name test_vox_image+mask --image_text_folder data/vox-celeba-alex_v2 --dataset vox --attr_mode image+mask2 --visual --vc_mode mask2_8x8 --num_visuals 2 --fullvc --text_seq_len 20 --dim 768 --pretrained_transformer openai_clip_visual --which_tokenizer simple --use_html --num_targets 8 --frame_num 8 --frame_step 4 --which_vae vqgan1024 --image_size 128 --log_root logs --use_cvae --iters 20 --batch_size 16 --n_per_sample 4 --n_sample 1 --no_debug --mp_T 20 --dalle_path vox_bert_image+mask_bs20_100k.pt
```

### Text and Partial Image
Training:
``` python
CUDA_VISIBLE_DEVICES=2,3 python3 train.py --name train_vox_text+partial --image_text_folder data/vox-celeba-alex_v2 --dataset vox --attr_mode image+text_dropout --vc_mode face_8x8 --visual --num_visuals 1 --fullvc --batch_size 20 --text_seq_len 50 --dim 768 --pretrained_transformer openai_clip_visual --iters 200000 --learning_rate 1e-4 --random_resize_crop_lower_ratio 1 --clip_grad_norm 1 --lr_decay --lr_scheduler warmuplr --optimizer adam --weight_decay 0.0 --which_tokenizer simple --use_html --log_every 200 --sample_every 5000 --n_sample 4 --n_per_sample 4 --num_targets 8 --frame_num 8 --frame_step 4 --which_vae vqgan1024 --image_size 128 --beta_rel 0.5 --beta_vid 0.5 --beta_msm 7 --log_root logs --lr_scheduler_warmup 5000 --msm_strategy_prob 7,1,1,1 --msm_bernoulli_prob 0.2,0.2 --vid_strategy_prob 1,1,1,1 --dropout_vc 0.4 --dist_url tcp://localhost:10006 --vae_path pretrained_models/vae_vox.ckpt --cvae_path pretrained_models/cvae_vox.ckpt --rel_no_fully_masked --mask_predict_steps 10 20 30 --mask_predict_steps1 20 
```

Testing:
``` python
CUDA_VISIBLE_DEVICES=2 python3 test.py --name test_vox_text+partial --image_text_folder data/vox-celeba-alex_v2 --dataset vox --attr_mode image+text_dropout --visual --vc_mode face_8x8 --num_visuals 1 --fullvc --text_seq_len 50 --dim 768 --pretrained_transformer openai_clip_visual --which_tokenizer simple --use_html --num_targets 8 --frame_num 8 --frame_step 4 --which_vae vqgan1024 --image_size 128 --log_root logs --use_cvae --iters 20 --batch_size 16 --n_per_sample 4 --n_sample 1 --no_debug --mp_T 20 --dalle_path vox_bert_text+partial_bs20_98k.pt
```

### Image and Video
Training:
``` python
CUDA_VISIBLE_DEVICES=6,7 python3 train.py --name train_vox_image+video --image_text_folder data/vox-celeba-alex_v2 --dataset vox --attr_mode image+video33 --vc_mode face2_8x8 --visual --num_visuals 4 --fullvc --batch_size 32 --text_seq_len 20 --dim 768 --pretrained_transformer openai_clip_visual --iters 200000 --learning_rate 1e-4 --random_resize_crop_lower_ratio 1 --clip_grad_norm 1 --lr_decay --lr_scheduler warmuplr --optimizer adam --weight_decay 0.0 --which_tokenizer simple --use_html --log_every 200 --sample_every 5000 --n_sample 2 --n_per_sample 4 --num_targets 8 --frame_num 8 --frame_step 4 --which_vae vqgan1024 --image_size 128 --beta_rel 0.5 --beta_vid 0.5 --beta_msm 7 --log_root logs --lr_scheduler_warmup 5000 --msm_strategy_prob 7,1,1,1 --msm_bernoulli_prob 0.2,0.2 --vid_strategy_prob 1,1,1,1 --dropout_vc 0.4 --dist_url tcp://localhost:10007 --vae_path pretrained_models/vae_vox.ckpt --cvae_path pretrained_models/cvae_vox.ckpt --rel_no_fully_masked --visual_aug_mode motion_color 
```

Testing:
``` python
CUDA_VISIBLE_DEVICES=0 python3 test.py --name test_vox_image+video --image_text_folder data/vox-celeba-alex_v2 --dataset vox --attr_mode image+video33 --visual --vc_mode face2_8x8 --num_visuals 4 --fullvc --text_seq_len 20 --dim 768 --pretrained_transformer openai_clip_visual --which_tokenizer simple --use_html --num_targets 8 --frame_num 8 --frame_step 4 --which_vae vqgan1024 --image_size 128 --log_root logs --use_cvae --iters 20 --batch_size 16 --n_per_sample 4 --n_sample 1 --no_debug --mp_T 20 --dalle_path vox_bert_image+video_bs32_149k.pt
```

### Text and Drawing
Training:
``` python

```

Testing:
``` python

```

## iPER
### Long video generation (extrapolation)
Testing:
``` python
CUDA_VISIBLE_DEVICES=4 python3 test.py --name test_iper_text_long --image_text_folder data/iper2 --dataset iper --slow --text_seq_len 16 --dim 768 --pretrained_transformer openai_clip_visual --which_tokenizer simple --use_html --num_visuals 0 --num_targets 8 --frame_num 8 --frame_step 8 --which_vae vqgan1024 --image_size 128 --log_root logs --dataset_keys iper_test.txt --iters 20 --batch_size 1 --n_per_sample 1 --n_sample 1 --no_debug --mp_T 20 --dalle_path iper_bert_txtdrop_slow_180k.pt --eval_mode long --long_mode long --t_repeat 50 --t_overlap 7 
```

### Long video generation (interpolation)
Testing:
``` python
CUDA_VISIBLE_DEVICES=0 python3 test.py --name test_iper_text_interp --image_text_folder data/iper2 --dataset iper --slow --text_seq_len 16 --dim 768 --pretrained_transformer openai_clip_visual --which_tokenizer simple --use_html --num_visuals 0 --num_targets 8 --frame_num 8 --frame_step 8 --which_vae vqgan1024 --image_size 128 --log_root logs --dataset_keys iper_test.txt --iters 20 --batch_size 1 --n_per_sample 1 --n_sample 1 --no_debug --mp_T 20 --dalle_path iper_bert_txtdrop_slow_180k.pt --eval_mode long --long_mode interp_real --t_repeat 2 
```

## Shapes
### With Image Control (center-cropped IC for background)
Testing:
``` python
CUDA_VISIBLE_DEVICES=0 python3 test.py --name test_shape_text+image --image_text_folder data/shapes1c --dataset shape_attr --attr_mode text --visual --vc_mode shape_4x4 --num_visuals 1 --text_seq_len 20 --dim 768 --pretrained_transformer openai_clip_visual --which_tokenizer simple --use_html --n_sample 1 --n_per_sample 4 --num_targets 16 --frame_num 16 --frame_step 1 --which_vae vqgan1024 --image_size 64 --log_root logs --iters 20 --mp_T 20 --dalle_path shape_bert_text+image_60k.pt 
```

### Dependent
Testing:
``` python
CUDA_VISIBLE_DEVICES=0 python3 test.py --name test_shape_depend --image_text_folder data/shapes_v2 --attr_mode color+shape+background+rand --num_visuals 3 --dataset shape_attr --text_seq_len 32 --dim 768 --pretrained_transformer openai_clip_visual --bpe_path dalle_pytorch/data/variety.bpe --which_tokenizer yttm --use_html --num_targets 16 --frame_num 16 --frame_step 1 --which_vae vqgan1024 --image_size 64 --visual --fullvc --dataset_keys data/shapes_v2/large.txt --insert_sep --test_mode shapes --iters 20 --n_sample 1 --n_per_sample 4 --negvc --batch_size 2 --no_debug --dalle_path shape_bert_depend_init=norand.pt 
```

# Pretrained Models
[checkpoint folder](https://drive.google.com/drive/folders/1q_YdEBylrAWeuSleq6Jp58epE3KM-oXK?usp=sharing)

## Multi-Modal Vox-Celeb
