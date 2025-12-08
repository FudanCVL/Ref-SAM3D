# Ref-SAM3D

arXiv: https://arxiv.org/abs/2511.19426

## TODO List

- [x] Release SAM3Agent + SAM3D code  
- [ ] Release quantitative evaluation code and results for Ref-SAM3D  
- [ ] Update the paper

## ⚙️ Installation
### 1.Installing SAM3D
```bash
conda env create -f environments/default.yml
conda activate ref-sam3d

# for pytorch/cuda dependencies
export PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"

# install sam3d-objects and core dependencies
pip install -e '.[dev]'
pip install -e '.[p3d]' # pytorch3d dependency on pytorch is broken, this 2-step approach solves it

# for inference
export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
pip install -e '.[inference]'

# patch things that aren't yet in official pip packages
./patching/hydra # https://github.com/facebookresearch/hydra/pull/2863
```

### 2.Getting Checkpoints

⚠️ Before using SAM 3D Objects, please request access to the checkpoints on the SAM 3D Objects
Hugging Face [repo](https://huggingface.co/facebook/sam-3d-objects). Once accepted, you
need to be authenticated to download the checkpoints. You can do this by running
the following [steps](https://huggingface.co/docs/huggingface_hub/en/quick-start#authentication)
(e.g. `hf auth login` after generating an access token).

```bash
pip install 'huggingface-hub[cli]<1.0'


TAG=hf
hf download \
  --repo-type model \
  --local-dir checkpoints/${TAG}-download \
  --max-workers 1 \
  facebook/sam-3d-objects
mv checkpoints/${TAG}-download/checkpoints checkpoints/${TAG}
rm -rf checkpoints/${TAG}-download
```
### 3.Installing SAM 3

⚠️ Before using SAM 3, please request access to the checkpoints on the SAM3 Hugging Face [repo](https://huggingface.co/facebook/sam3). 
Once accepted, you
need to be authenticated to download the checkpoints. You can do this by running
the following [steps](https://huggingface.co/docs/huggingface_hub/en/quick-start#authentication)
(e.g. `hf auth login` after generating an access token.)
```bash
git clone https://github.com/facebookresearch/sam3.git

cd sam3
pip install -e .
pip install -e ".[notebooks]"

```

## 📌 Getting Started
### Setup vLLM server 
This step is only required if you are using a model served by vLLM, skip this step if you are calling LLM using an API like Gemini and GPT.

* Install vLLM (in a separate conda env from SAM 3 to avoid dependency conflicts).
```bash
conda create -n vllm python=3.12
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu128
```
* Start vLLM server on the same machine
```bash
vllm serve Qwen/Qwen3-VL-4B-Instruct-FP8 --tensor-parallel-size 1 --allowed-local-media-path / --max-model-len 65536 --enforce-eager --port 8001 --gpu-memory-utilization 0.4
```
You can control vLLM's GPU memory usage via the max-model-len and gpu-memory-utilization parameters.


### Run inference code
```bash
python inference_pipeline.py --image_path <path_to_image_file> --prompt "<your_text_prompt>" --model <llm_model_name> --llm_server_url <llm_server_url>
```
The model's predictions and 3D reconstruction results will be saved in the output/ directory with the following structure:
```bash
output/
└── <image_name>/                 
    └── <prompt_text>/         
        ├── splat.ply              
        ├── mask_xxx.png           
        ├── agent_debug_out/       
        ├── none_out/             
        ├── sam_out/               
        └── ...
```    
### Visualization
To visualize the results, run the following command. This will launch a local Gradio interface where you can view the 3D model:
```bash     
python vis.py --ply_path <path_to_ply_file>
```


# Citation
If you find this work useful, we would greatly appreciate your citation:
```
@article{zhou2025ref,
  title={Ref-SAM3D: Bridging SAM3D with Text for Reference 3D Reconstruction},
  author={Zhou, Yun and Wang, Yaoting and Jie, Guangquan and Liu, Jinyu and Ding, Henghui},
  journal={arXiv preprint arXiv:2511.19426},
  year={2025}
}
```
