python3 test.py --name test_shape_text+image --image_text_folder data/shapes1c \
    --dataset shape_attr --attr_mode text --visual --vc_mode shape_4x4 \
    --num_visuals 1 --text_seq_len 20 --dim 768 --pretrained_transformer openai_clip_visual \
    --which_tokenizer simple --use_html --n_sample 1 --n_per_sample 4 --num_targets 16 \
    --frame_num 16 --frame_step 1 --which_vae vqgan1024 --image_size 64 --log_root logs \
    --iters 20 --mp_T 20 --dalle_path shape_bert_text+image_60k.pt
