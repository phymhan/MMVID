CUDA_VISIBLE_DEVICES=0,1 python3 train.py --name train_vox_text --image_text_folder data/vox-celeba-alex_v2 --dataset video_text --batch_size 48 --text_seq_len 50 --dim 768 --pretrained_transformer openai_clip_visual --iters 200000 --learning_rate 1e-4 --random_resize_crop_lower_ratio 1 --clip_grad_norm 1 --lr_decay --lr_scheduler warmuplr --optimizer adam --weight_decay 0.0 --which_tokenizer simple --use_html --log_every 200 --sample_every 5000 --n_sample 4 --n_per_sample 4 --num_visuals 0 --num_targets 8 --frame_num 8 --frame_step 4 --which_vae vqgan1024 --image_size 128 --beta_rel 0.5 --beta_vid 0.5 --beta_msm 7 --log_root logs --lr_scheduler_warmup 5000 --msm_strategy_prob 7,1,1,1 --msm_bernoulli_prob 0.2,0.2 --vid_strategy_prob 1,1,1,1 --dropout_vc 0.4 --dist_url tcp://localhost:10001 --vae_path pretrained_models/vae_vox.ckpt --rel_no_fully_masked  --mask_predict_steps 10 20 30 --mask_predict_steps1 20