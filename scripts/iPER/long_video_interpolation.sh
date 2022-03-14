python3 test.py --name test_iper_text_interp \
    --image_text_folder data/iper2 --dataset iper \
    --slow --text_seq_len 16 --dim 768 --pretrained_transformer openai_clip_visual \
    --which_tokenizer simple --use_html --num_visuals 0 --num_targets 8 \
    --frame_num 8 --frame_step 8 --which_vae vqgan1024 --image_size 128 \
    --log_root logs --dataset_keys iper_test.txt --iters 20 --batch_size 1 \
    --n_per_sample 1 --n_sample 1 --no_debug --mp_T 20 \
    --dalle_path iper_bert_txtdrop_slow_180k.pt --eval_mode long --long_mode interp_real --t_repeat 2
