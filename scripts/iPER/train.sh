python3 train.py --name train_iper --image_text_folder data/iper2 \
    --dataset iper --batch_size 24 --text_seq_len 16 --use_html \
    --num_visuals 0 --num_targets 8 --frame_num 8 --frame_step 8 \
    --image_size 128 --msm_bernoulli_prob 0.1,0.3 --dropout_vc 0.4 \
    --dist_url tcp://localhost:10001 --vae_path pretrained_models/vae_iper.ckpt \
    --rel_no_fully_masked --mask_predict_steps 10 20 30 --num_workers 24 \
    --drop_sentence --slow --dataset_keys iper_train.txt 
