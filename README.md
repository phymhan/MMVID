## MMVID<br><sub>Show Me What and Tell Me How: Video Synthesis via Multimodal Conditioning (CVPR 2022)</sub>

### [Project](https://snap-research.github.io/MMVID/) | [arXiv](https://arxiv.org/abs/2203.02573)

<div align="center">
  Generated Videos on Multimodal VoxCeleb
</div>

<div class="gif">
<p align="center">
<img src='images/demo.gif' align="center" width=400>
</p>
</div>

This repo contains the code for training and testing, models, and data for MMVID.

> [**Show Me What and Tell Me How: Video Synthesis via Multimodal Conditioning**](https://xxx)<br>
> [Ligong Han](https://phymhan.github.io/), [Jian Ren](https://alanspike.github.io/), [Hsin-Ying Lee](http://hsinyinglee.com/), [Francesco Barbieri](https://fvancesco.github.io/), [Kyle Olszewski](https://kyleolsz.github.io/), [Shervin Minaee](https://sites.google.com/site/shervinminaee/home), [Dimitris Metaxas](https://people.cs.rutgers.edu/~dnm/), [Sergey Tulyakov](http://www.stulyakov.com/)<br>
> Snap Inc., Rutgers University<br>
> CVPR 2022

# Multi-Modal Vox-Celeb Dataset

[Dataset](mm_vox_celeb/README.md)
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

<details>
  <summary>Inference </summary>
  
  #### With Image Control (center-cropped IC for background):
    bash scripts/Shapes/image_control.sh

  #### Dependent:
    bash scripts/Shapes/dependent.sh
 
</details>





# Pretrained Models
[checkpoint folder](https://drive.google.com/drive/folders/1q_YdEBylrAWeuSleq6Jp58epE3KM-oXK?usp=sharing)

## Multi-Modal Vox-Celeb

## Citation

If our code, data, or models help your work, please cite our paper:
```BibTeX
@article{han2022show,
  title={Show Me What and Tell Me How: Video Synthesis via Multimodal Conditioning},
  author={Han, Ligong and Ren, Jian and Lee, Hsin-Ying and Barbieri, Francesco and Olszewski, Kyle and Minaee, Shervin and Metaxas, Dimitris and Tulyakov, Sergey},
  journal={arXiv preprint arXiv:2203.02573},
  year={2022}
}
```