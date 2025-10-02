# Cosmos-Transfer1-DiffusionRenderer


Cosmos-Transfer1-DiffusionRenderer is a dedicated video relighting framework based on [NVIDIA Cosmos World Foundation Models](https://www.nvidia.com/en-us/ai/cosmos/), designed for high-quality de-lighting and re-lighting of input image or videos.
It enables controllable video lighting manipulation, editing, and synthetic data augmentation—supporting physical AI systems to train perception and policy models with improved robustness to varying lighting conditions.
It is powered by NVIDIA’s Cosmos framework and builds on the research project [DiffusionRenderer](https://research.nvidia.com/labs/toronto-ai/DiffusionRenderer/), with an improved data pipeline and enhanced visual fidelity.

**[Paper](https://arxiv.org/abs/2501.18590) | [Project Page](https://research.nvidia.com/labs/toronto-ai/DiffusionRenderer/) | [Demo Video](https://www.youtube.com/watch?v=Q3xhYNbXM9c) | [Blog](https://blogs.nvidia.com/blog/cvpr-2025-ai-research-diffusionrenderer/)**

![img](asset/teaser.gif)


## 🚀 News 
-  [June 12, 2025] 🔥 Released Cosmos DiffusionRenderer code and model weights in this repo!  
-  [June 11, 2025] 🎬 Released our [video demo](https://www.youtube.com/watch?v=Q3xhYNbXM9c) and [blog](https://blogs.nvidia.com/blog/cvpr-2025-ai-research-diffusionrenderer/) on Cosmos DiffusionRenderer. 
-  [June 11, 2025] 🔥 Released the code and model weights for the academic version of DiffusionRenderer. This version reproduces the results in our paper. Check the [GitHub repo](https://github.com/nv-tlabs/diffusion-renderer) and [model weights](https://huggingface.co/collections/nexuslrf/diffusionrenderer-svd-68472d636e85c29b6c25422f)! 


## Installation

### Minimum requirements

- Python 3.10
- NVIDIA GPU with at least 16GB VRAM, recommend to have >=48GB VRAM 
- NVIDIA drivers and CUDA 12.0 or higher
- At least 70GB free disk space

The installation has been tested on:
- Ubuntu 20.04
- NVIDIA A100 GPU (80GB VRAM), NVIDIA A6000 GPU (48GB VRAM)


### Conda environment 

The below commands creates the `cosmos-predict1` conda environment and installs the dependencies for inference:
```bash
# Create the cosmos-predict1 conda environment.
conda env create --file cosmos-predict1.yaml
# Activate the cosmos-predict1 conda environment.
conda activate cosmos-predict1
# Install the dependencies.
pip install -r requirements.txt
# Patch Transformer engine linking issues in conda environments.
ln -sf $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/
ln -sf $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/python3.10
# Install Transformer engine.
pip install transformer-engine[pytorch]==1.12.0
```

If the [dependency](https://github.com/NVlabs/nvdiffrast/blob/main/docker/Dockerfile) is well taken care of, install `nvdiffrast` with:
```bash
# Patch dependency for nvdiffrast 
ln -sf $CONDA_PREFIX/lib/python3.10/site-packages/triton/backends/nvidia/include/crt $CONDA_PREFIX/include/
pip install git+https://github.com/NVlabs/nvdiffrast.git
```
For platforms other than ubuntu, check [nvdiffrast official documentation](https://nvlabs.github.io/nvdiffrast/) and their [Dockerfile](https://github.com/NVlabs/nvdiffrast/blob/main/docker/Dockerfile). 


### Download model weights (~56GB) 

The model weights are available on [Hugging Face](https://huggingface.co/collections/nvidia/cosmos-diffusionrenderer-6893a56d79182313b01c777e).

1. Generate a [Hugging Face](https://huggingface.co/settings/tokens) access token (if you haven't done so already). Set the access token to `Read` permission (default is `Fine-grained`).

2. Log in to Hugging Face with the access token:
   ```bash
   huggingface-cli login
   ```

3. Download the DiffusionRenderer model weights from [Hugging Face](https://huggingface.co/collections/nvidia/cosmos-diffusionrenderer-6893a56d79182313b01c777e):
   ```bash
   CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python scripts/download_diffusion_renderer_checkpoints.py --checkpoint_dir checkpoints
   ```

## Inference: Image examples 

This example demonstrates how to use DiffusionRenderer for delighting and relighting a set of images, using images placed in the `asset/examples/image_examples/` folder. The model will process each image in the folder; using fewer images will reduce the total processing time.

Approximately 16GB of GPU VRAM is recommended. If you encounter out-of-memory errors, add `--offload_diffusion_transformer --offload_tokenizer` to the command to reduce GPU memory usage. 

### Inverse rendering of images 

This will estimate albedo, metallic, roughness, depth, normals (G-buffers) from each input image using the pre-trained Inverse Renderer model. The inference script is `cosmos_predict1/diffusion/inference/inference_inverse_renderer.py`. 

To perform inverse rendering on a set of images, use the following command: 
```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/inference_inverse_renderer.py \
    --checkpoint_dir checkpoints --diffusion_transformer_dir Diffusion_Renderer_Inverse_Cosmos_7B \
    --dataset_path=asset/examples/image_examples/ --num_video_frames 1 --group_mode webdataset \
    --video_save_folder=asset/example_results/image_delighting/ --save_video=False
```

The configs here: 
- `--checkpoint_dir` specifies the directory containing model checkpoints, use default `checkpoints/`.
- `--diffusion_transformer_dir` selects the specific model variant to use.
- `--dataset_path` points to the folder with your input images.
- `--num_video_frames 1` processes each image individually (as a single frame).
- `--video_save_folder` sets the output directory for the results.
- `--save_video=False` disables saving a video file, since we're processing images. 

Explanation on additional arguments can be found inside the script. 
Additionally,  the `--inference_passes` argument controls which G-buffer maps are estimated and saved by the inverse renderer. By default, it runs on five passes: `basecolor`, `normal`, `depth`, `roughness`, and `metallic`. You can specify a subset to only compute certain outputs. 


### Relighting of images 

Using the gbuffer frames from the previous step `asset/example_results/image_delighting/gbuffer_frames`, we use Forward Renderer to relight images with user provided environment maps. 

```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/inference_forward_renderer.py \
    --checkpoint_dir checkpoints --diffusion_transformer_dir Diffusion_Renderer_Forward_Cosmos_7B \
    --dataset_path=asset/example_results/image_delighting/gbuffer_frames --num_video_frames 1 \
    --envlight_ind 0 1 2 3 --use_custom_envmap=True \
    --video_save_folder=asset/example_results/image_relighting/
```
Here, the `--envlight_ind 0 1 2 3` argument specifies which environment maps (HDRIs) to use for relighting. 
Each number corresponds to a different predefined lighting environment included with the code (check `ENV_LIGHT_PATH_LIST` in `inference_forward_renderer.py`). 

By providing multiple indices (e.g., `0 1 2 3`), the forward renderer will relight each input using all selected environment maps, producing multiple relit outputs per input. You can choose a subset (e.g., `--envlight_ind 0 2`) to use only specific lighting conditions. 
This script will produce results in `asset/example_results/image_relighting/`. 


### Illumination randomization of images 

When environment maps are not available, the command below allows to randomize illumination by changing random seeds. 
```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/inference_forward_renderer.py \
    --checkpoint_dir checkpoints --diffusion_transformer_dir Diffusion_Renderer_Forward_Cosmos_7B \
    --dataset_path=asset/example_results/image_delighting/gbuffer_frames --num_video_frames 1 \
    --envlight_ind 0 1 2 3 --use_custom_envmap=False \
    --video_save_folder=asset/example_results/image_relighting_random/
```
by setting `--use_custom_envmap` to `False`. 

The script will produce results in `asset/example_results/image_relighting_random/`. 


## Inference: Video examples 

This example uses videos placed in the `asset/examples/video_examples/` folder. The model will process each video in the folder; using fewer videos will reduce the total processing time.

The peak GPU memory usage is ~27GB. If you encounter out-of-memory errors, add `--offload_diffusion_transformer --offload_tokenizer` to the command to reduce GPU memory usage. 


### Extract frames from videos  

Before running the inverse renderer on videos, you need to extract individual frames from each video file. This step converts each video into a sequence of images, which are then used as input for the rendering pipeline.

The following command will process all videos in the `asset/examples/video_examples/` directory, extracting frames and saving them into the `asset/examples/video_frames_examples/` folder:
```bash
python scripts/dataproc_extract_frames_from_video.py --input_folder asset/examples/video_examples/ --output_folder asset/examples/video_frames_examples/ 
--frame_rate 24 --resize 1280x704 --max_frames=57
```

### Inverse rendering of videos

This step performs inverse rendering on a sequence of video frames to estimate the underlying G-buffer maps (such as basecolor, normal, depth, roughness, and metallic) for each frame. 

Example command:
```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/inference_inverse_renderer.py \
    --checkpoint_dir checkpoints --diffusion_transformer_dir Diffusion_Renderer_Inverse_Cosmos_7B \
    --dataset_path=asset/examples/video_frames_examples/ --num_video_frames 57 --group_mode folder \
    --video_save_folder=asset/example_results/video_delighting/ 
```


### Relighting of videos 

This step takes the G-buffer frames generated by the inverse renderer and applies novel lighting conditions to produce relit video frames. The command below uses four different environment maps (specified by `--envlight_ind 0 1 2 3`) to relight the video. 
```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/inference_forward_renderer.py \
    --checkpoint_dir checkpoints --diffusion_transformer_dir Diffusion_Renderer_Forward_Cosmos_7B \
    --dataset_path=asset/example_results/video_delighting/gbuffer_frames --num_video_frames 57 \
    --envlight_ind 0 1 2 3 --use_custom_envmap=True \
    --video_save_folder=asset/example_results/video_relighting/
```


We can also use a static frame and show relighting with a rotating environment light by specifying `--rotate_light=True --use_fixed_frame_ind=True`: 
```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/inference_forward_renderer.py \
    --checkpoint_dir checkpoints --diffusion_transformer_dir Diffusion_Renderer_Forward_Cosmos_7B \
    --dataset_path=asset/example_results/video_delighting/gbuffer_frames --num_video_frames 57 \
    --envlight_ind 0 1 2 3 --use_custom_envmap=True \
    --video_save_folder=asset/example_results/video_relighting_rotation/ --rotate_light=True --use_fixed_frame_ind=True
```


## License and Contact

Cosmos DiffusionRenderer source code is released under the [Apache 2 License](https://www.apache.org/licenses/LICENSE-2.0).
Models are released under the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license). 

For business inquiries, please visit our website and submit the form: [NVIDIA Research Licensing](https://www.nvidia.com/en-us/research/inquiries/).
For technical questions related to the model, please contact Zian Wang. 


## Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{DiffusionRenderer,
    author = {Ruofan Liang and Zan Gojcic and Huan Ling and Jacob Munkberg and 
        Jon Hasselgren and Zhi-Hao Lin and Jun Gao and Alexander Keller and 
        Nandita Vijaykumar and Sanja Fidler and Zian Wang},
    title = {DiffusionRenderer: Neural Inverse and Forward Rendering with Video Diffusion Models},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2025}
}
```
