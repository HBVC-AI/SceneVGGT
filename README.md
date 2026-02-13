<div align="center">
  <h1>SceneVGGT: VGGT-based online 3D semantic SLAM for indoor scene understanding and navigation</h1>

  <p>
    <a href="https://arxiv.org/user/">Paper</a>
  </p>

  <p>
    Anna Gelencsér-Horváth<sup>*</sup><sup>&dagger;</sup> ·
    Gergely Dinya<sup>*</sup> ·
    Péter Halász ·
    Dorka Erős ·
    Islam Muhammad Muqsit ·
    Kristóf Karacs
  </p>

  <p>
    <sup>*</sup> Equal contribution.
    <sup>&dagger;</sup> Corresponding author.
  </p>
</div>



**SceneVGGT** is a spatio-temporal 3D scene understanding framework that combines SLAM with semantic mapping for autonomous and assistive navigation.
It supports online, real-time processing of streamed data (e.g., from an iPhone Pro).
The pipeline’s GPU memory usage remains under 17 GB, irrespectively of the length of the input sequence and achieves competitive point-cloud performance on the
ScanNet++ benchmark. Overall, SceneVGGT ensures robust semantic identification and is fast enough to support interactive assistive navigation with audio feedback.

## News

<!-- - **[2026/2/13]** Paper released on [arXiv](https://arxiv.org/abs/   ). -->
- **[2025/2/12]** Code release.

## Overview

SceneVGGT enables temporally coherent 3D semantic mapping by lifting 2D instance masks into 3D and tracking instances with the VGGT tracking head. Persistent object identities + timestamps provide computationally efficient, temporally consistent change detection, while floor-plane projection of object locations supports downstream assistive navigation—including a proof-of-concept navigation module.


### 3D semantic SLAM and navigation from Streaming Inputs
<p align="center">
  <img src="assets/lab.gif" width="45%" />
  <img src="assets/change.gif" width="45%" />
</p>

### Installation

1. Clone SceneVGGT
```bash
git clone git@github.com:HBVC-AI/SceneVGGT.git
cd SceneGGT
```
2. Create conda environment
```bash
conda create -n scenevggt python=3.10
conda activate SceneVGGT 
```

3. Install requirements
```bash
pip install -r requirements.txt
```

### Download Checkpoints
Please download VGG-T model from [here](https://huggingface.co/facebook/VGGT-1B/blob/main/model.pt).


### Evaluation codes
Coming soon.

## Citation

If you find this project helpful, please consider citing the following paper:
```
@article{scenevggt,
      title={SceneVGGT: VGGT-based Online 3D Semantic SLAM for Indoor Scene Understanding and Navigation}, 
      author={Anna Gelencsér-Horváth, Gergely Dinya, Dorka Boglárka Erős, Péter Halász, Islam Muhammad Muqsit, Kristóf Karacs},
      journal={arXiv preprint } 
      year={2026}
}
```
