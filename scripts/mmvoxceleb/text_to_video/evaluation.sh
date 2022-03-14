python3 test.py --name test_vox_text --image_text_folder data/vox-celeba-alex_v2 --dataset video_text --text_seq_len 50 --dim 768 --pretrained_transformer openai_clip_visual --lr_decay --lr_scheduler warmuplr --optimizer adam --weight_decay 0.0 --which_tokenizer simple --use_html --num_visuals 0 --num_targets 8 --frame_num 8 --frame_step 4 --which_vae vqgan1024 --image_size 128 --log_root logs --iters 20 --batch_size 16 --n_per_sample 4 --n_sample 1 --no_debug --mp_T 20 --dalle_path vox_bert_text_bs48_100k.pt