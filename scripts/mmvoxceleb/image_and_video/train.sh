CUDA_VISIBLE_DEVICES=6,7 python3 train.py --name train_vox_image+video --image_text_folder data/vox-celeba-alex_v2 --dataset vox --attr_mode image+video33 --vc_mode face2_8x8 --visual --num_visuals 4 --fullvc --batch_size 32 --text_seq_len 20 --dim 768 --pretrained_transformer openai_clip_visual --iters 200000 --learning_rate 1e-4 --random_resize_crop_lower_ratio 1 --clip_grad_norm 1 --lr_decay --lr_scheduler warmuplr --optimizer adam --weight_decay 0.0 --which_tokenizer simple --use_html --log_every 200 --sample_every 5000 --n_sample 2 --n_per_sample 4 --num_targets 8 --frame_num 8 --frame_step 4 --which_vae vqgan1024 --image_size 128 --beta_rel 0.5 --beta_vid 0.5 --beta_msm 7 --log_root logs --lr_scheduler_warmup 5000 --msm_strategy_prob 7,1,1,1 --msm_bernoulli_prob 0.2,0.2 --vid_strategy_prob 1,1,1,1 --dropout_vc 0.4 --dist_url tcp://localhost:10007 --vae_path pretrained_models/vae_vox.ckpt --cvae_path pretrained_models/cvae_vox.ckpt --rel_no_fully_masked --visual_aug_mode motion_color
