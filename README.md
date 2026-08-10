<div align="center">
<h2>StereoCrafter: Diffusion-based Generation of Long and High-fidelity Stereoscopic 3D from Monocular Videos</h2>

Sijie Zhao*&emsp;
Wenbo Hu*&emsp;
Xiaodong Cun\*&emsp;
Yong Zhang&dagger;&emsp;
Xiaoyu Li&dagger;&emsp;<br>
Zhe Kong&emsp;
Xiangjun Gao&emsp;
Muyao Niu&emsp;
Ying Shan

&emsp;\* equal contribution &emsp; &dagger; corresponding author

<h3>Tencent AI Lab&emsp;&emsp;ARC Lab, Tencent PCG</h3>

<a href='https://arxiv.org/abs/2409.07447'><img src='https://img.shields.io/badge/arXiv-PDF-a92225'></a> &emsp;
<a href='https://stereocrafter.github.io/'><img src='https://img.shields.io/badge/Project_Page-Page-64fefe' alt='Project Page'></a> &emsp;
<a href='https://huggingface.co/TencentARC/StereoCrafter2'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Weights-yellow'></a>

</div>

## 💡 Abstract

We propose a novel framework to convert any 2D videos to immersive stereoscopic 3D ones that can be viewed on different display devices, like 3D Glasses, Apple Vision Pro and 3D Display. It can be applied to various video sources, such as movies, vlogs, 3D cartoons, and AIGC videos.

## 🛠️ Installation

#### 1. Set up the environment

We run our code on Python 3.13 and Cuda 12.8.

#### 2. Clone the repo

```bash
git clone --recursive https://github.com/Emanuel-de-Jong/StereoCrafter2-VR.git
cd StereoCrafter2-VR
```

#### 3. Install the requirements

```bash
conda create -n stereocrafter2 python=3.13 -y
conda activate stereocrafter2
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Dependencies

#### 1. Install customized 'Forward-Warp' package for forward splatting

```bash
cd ./dependency/Forward-Warp
# Manually copy the content of Forward-Warp-Overwrites into Forward-Warp
chmod a+x install.sh
CC=gcc-12 CXX=g++-12 ./install.sh
```

#### 2. Apply DepthCrafter changes

```bash
cd ./dependency/
# Manually copy the content of DepthCrafter-Overwrites into DepthCrafter
```

<!-- ### 3. Download Real-ESRGAN
1. Download this [Ubuntu release](https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan/releases/download/v0.2.0/realesrgan-ncnn-vulkan-v0.2.0-ubuntu.zip).
2. Unpack it in `dependency/` -->

#### 3. Download Video2X

1. Download this [AppImage](https://github.com/k4yt3x/video2x/releases/download/6.4.0/Video2X-x86_64.AppImage).
2. Put it in `dependency/` in a new folder `Video2X/`.
3. `chmod +x dependency/Video2X/Video2X-x86_64.AppImage`.

## 📦 Model Weights

<!-- TODO: 1. Download the small files from the release.
2. Extract the weights folder in the project root.
3. Download the big files individually: -->

#### 1. Download the [SVD img2vid model](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1) for the image encoder and VAE.

```bash
cd ../..
# In StereoCrafter2-VR project root directory
mkdir weights
cd ./weights
git lfs install
mkdir stable-video-diffusion-img2vid-xt-1-1
# Download the files model_index.json, image_encoder/config.json and image_encoder/model.fp16.safetensors and the folders feature_extractor and vae from https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1
# Put them in the folder
```

#### 2. Download the [DepthCrafter model](https://huggingface.co/tencent/DepthCrafter) for the video depth estimation.

```bash
mkdir DepthCrafter
# Download the files config.json and diffusion_pytorch_model.safetensors from https://huggingface.co/tencent/DepthCrafter
# Put them in the folder
```

#### 3. Download the [Wan2.1-VACE-14B-diffusers model](https://huggingface.co/Wan-AI/Wan2.1-VACE-14B-diffusers) for the text encoder and VAE.

```bash
mkdir Wan2.1-VACE-14B-diffusers
# Download the file model_index.json and the folders tokenizer, text_encoder and vae from https://huggingface.co/Wan-AI/Wan2.1-VACE-14B-diffusers
# Put them in the folder
```

#### 4. Download the [StereoCrafter2 model](https://huggingface.co/enoky/StereoCrafter2-FP8) for the stereo video generation.

```bash
mkdir StereoCrafter2-FP8
# Download the files config.json and diffusion_pytorch_model_fp8.pt from https://huggingface.co/enoky/StereoCrafter2-FP8
# Put them in the folder
```

## 🔄 Inference

Scripts:

```bash
conda activate stereocrafter2
# Then either
./run_inference.sh
# Or
./run_batch.sh
```

There are two main steps in these scripts for generating stereo video.

#### 1. Depth-Based Video Splatting Using the Video Depth from DepthCrafter

Execute the following command:

```bash
python s1_depth_splatting_inference.py --pre_trained_path [PATH] --unet_path [PATH]
                                    --input_video_path [PATH] --output_video_path [PATH]
```

Arguments:

- `--pre_trained_path`: Path to the SVD img2vid model weights (e.g., `./weights/stable-video-diffusion-img2vid-xt-1-1`).
- `--unet_path`: Path to the DepthCrafter model weights (e.g., `./weights/DepthCrafter`).
- `--input_video_path`: Path to the input video (e.g., `./input/vid.mp4`).
- `--output_video_path`: Path to the output video (e.g., `./outputs/vid_1_splatting.mp4`).
- `--max_disp`: Parameter controlling the maximum disparity between the generated right video and the input left video. Default value is `20` pixels.

The first step generates a video grid with input video, visualized depth map, occlusion mask, and splatting right video, as shown below:

#### 2. Stereo Video Inpainting of the Splatting Video

Execute the following command:

```bash
python s2_inpainting_inference.py --pre_trained_path [PATH] --transformer_path [PATH]
                               --input_video_path [PATH] --save_dir [PATH]
```

Arguments:

- `--pre_trained_path`: Path to the SVD img2vid model weights (e.g., `./weights/Wan2.1-VACE-14B-diffusers`).
- `--transformer_path`: Path to the StereoCrafter model weights (e.g., `./weights/StereoCrafter2-FP8`).
- `--input_video_path`: Path to the splatting video result generated by the first stage (e.g., `./outputs/vid_1_splatting.mp4`).
- `--save_dir`: Directory for the output stereo video (e.g., `./outputs`).
- `--tile_num`: The number of tiles in width and height dimensions for tiled processing, which allows for handling high resolution input without requiring more GPU memory. The default value is `1` (1 $\times$ 1 tile). For input videos with a resolution of 2K or higher, you could use more tiles to avoid running out of memory.

The stereo video inpainting generates the stereo video result in side-by-side format and anaglyph 3D format, as shown below:

## 🤝 Acknowledgements

We would like to express our gratitude to the following open-source projects:

- [Stable Video Diffusion](https://github.com/Stability-AI/generative-models): A latent diffusion model trained to generate video clips from an image or text conditioning.
- [DepthCrafter](https://github.com/Tencent/DepthCrafter): A novel method to generate temporally consistent depth sequences from videos.
- [Wan2.1-VACE-14B-diffusers](https://github.com/Wan-Video/Wan2.1): A comprehensive and open suite of video foundation models that pushes the boundaries of video generation.

## 📚 Citation

```bibtex
@article{zhao2024stereocrafter,
  title={Stereocrafter: Diffusion-based generation of long and high-fidelity stereoscopic 3d from monocular videos},
  author={Zhao, Sijie and Hu, Wenbo and Cun, Xiaodong and Zhang, Yong and Li, Xiaoyu and Kong, Zhe and Gao, Xiangjun and Niu, Muyao and Shan, Ying},
  journal={arXiv preprint arXiv:2409.07447},
  year={2024}
}
```
