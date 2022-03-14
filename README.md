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

- [ ] TO DO: add explanation
- [ ] TO DO: add evaluation scripts

<details>
  <summary>Text-to-Video</summary>
  
  #### Training:
    bash scripts/mmvoxceleb/text_to_video/train.sh
  #### Testing:
    bash scripts/mmvoxceleb/text_to_video/test.sh
  #### For Quantitative Evaluation (FVD and PRD):
    bash scripts/mmvoxceleb/text_to_video/evaluation.sh
</details>

<details>
  <summary>Text and Mask</summary>
  
  #### Training:
    bash scripts/mmvoxceleb/text_and_mask/train.sh
  #### Testing:
    bash scripts/mmvoxceleb/text_and_mask/test.sh
  #### For Quantitative Evaluation (FVD and PRD):
    To Add
</details>

<details>
  <summary>Text and Drawing</summary>
  
  #### Training:
    bash scripts/mmvoxceleb/text_and_drawing/train.sh
  #### Testing:
    bash scripts/mmvoxceleb/text_and_drawing/test.sh
  #### For Quantitative Evaluation (FVD and PRD):
    To Add
</details>

<details>
  <summary>Drawing and Mask</summary>
  
  #### Training:
    bash scripts/mmvoxceleb/drawing_and_mask/train.sh
  #### Testing:
    bash scripts/mmvoxceleb/drawing_and_mask/test.sh
  #### For Quantitative Evaluation (FVD and PRD):
    To Add
</details>

<details>
  <summary>Image and Mask</summary>
  
  #### Training:
    bash scripts/mmvoxceleb/image_and_mask/train.sh
  #### Testing:
    bash scripts/mmvoxceleb/image_and_mask/test.sh
  #### For Quantitative Evaluation (FVD and PRD):
    To Add
</details>

<details>
  <summary>Text and Partial Image</summary>
  
  #### Training:
    bash scripts/mmvoxceleb/image_and_mask/train.sh
  #### Testing:
    bash scripts/mmvoxceleb/image_and_mask/test.sh
  #### For Quantitative Evaluation (FVD and PRD):
    To Add
</details>

<details>
  <summary>Image and Video</summary>
  
  #### Training:
    bash scripts/mmvoxceleb/image_and_mask/train.sh
  #### Testing:
    bash scripts/mmvoxceleb/image_and_mask/test.sh
  #### For Quantitative Evaluation (FVD and PRD):
    To Add
</details>



## iPER

<details>
  <summary>Long video generation </summary>
  
  #### Extrapolation:
    bash scripts/iPER/long_video_extrapolation.sh

  #### Interpolation:
    bash scripts/iPER/long_video_interpolation.sh
 
</details>



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
