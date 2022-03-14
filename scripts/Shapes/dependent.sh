python3 test.py --name test_shape_depend \
    --image_text_folder data/shapes_v2 --attr_mode color+shape+background+rand \
    --num_visuals 3 --dataset shape_attr --text_seq_len 32 --dim 768 \
    --pretrained_transformer openai_clip_visual --bpe_path dalle_pytorch/data/variety.bpe \
    --which_tokenizer yttm --use_html --num_targets 16 --frame_num 16 --frame_step 1 \
    --which_vae vqgan1024 --image_size 64 --visual --fullvc \
    --dataset_keys data/shapes_v2/large.txt --insert_sep --test_mode shapes --iters 20 \
    --n_sample 1 --n_per_sample 4 --negvc --batch_size 2 --no_debug --dalle_path shape_bert_depend_init=norand.pt
