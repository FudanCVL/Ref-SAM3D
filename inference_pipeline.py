import sys
import os
import json
from pathlib import Path
import argparse
# import inference code
sys.path.append("notebook")
from inference import Inference, load_image, load_single_mask, load_mask, make_scene
from pycocotools import mask as mask_utils
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import sam3
from sam3.sam3 import build_sam3_image_model
from sam3.sam3.model.sam3_image_processor import Sam3Processor
from functools import partial
#from IPython.display import display, Image
from sam3.sam3.agent.client_llm import send_generate_request as send_generate_request_orig
from sam3.sam3.agent.client_sam3 import call_sam_service as call_sam_service_orig
from sam3.sam3.agent.inference import run_single_image_inference
from PIL import Image

def main(args):
    sam3_root = './sam3'
    bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
    model = build_sam3_image_model(bpe_path=bpe_path)
    processor = Sam3Processor(model, confidence_threshold=0.5)

    #LLM Config
    LLM_CONFIGS = {
        # vLLM-served models
        "qwen3_vl_8b_thinking": {
            "provider": "vllm",
            "model": "Qwen/Qwen3-VL-8B-Thinking",
        }, 
        "qwen3_vl_8b_instruct": {
            "provider": "vllm",
            "model": "Qwen/Qwen3-VL-8B-Instruct",
        },
        "qwen3_vl_4b_instruct_fp8": {
            "provider": "vllm",
            "model": "Qwen/Qwen3-VL-4B-Instruct-FP8",
        },
        # models served via external APIs
        # add your own
    }

    model = args.model
    LLM_API_KEY = "DUMMY_API_KEY"

    llm_config = LLM_CONFIGS[model]
    llm_config["api_key"] = LLM_API_KEY
    llm_config["name"] = model

    # setup API endpoint
    if llm_config["provider"] == "vllm":
        LLM_SERVER_URL = args.llm_server_url  # replace this with your vLLM server address as needed
    else:
        LLM_SERVER_URL = llm_config["base_url"]

    # prepare input args and run single image inference
    image = args.image_path
    prompt = args.prompt
    image = os.path.abspath(image)
    send_generate_request = partial(send_generate_request_orig, server_url=LLM_SERVER_URL, model=llm_config["model"], api_key=llm_config["api_key"])
    call_sam_service = partial(call_sam_service_orig, sam3_processor=processor)
    filename = os.path.splitext(os.path.basename(args.image_path))[0]
    output_root_dir = args.output_dir
    output_dir = os.path.join(output_root_dir, filename,prompt)
    output_image_path = run_single_image_inference(
        image, prompt, llm_config, send_generate_request, call_sam_service, 
        debug=True, output_dir = output_dir
    )

    image_basename = os.path.splitext(os.path.basename(image))[0]
    prompt_for_filename = prompt.replace("/", "_").replace(" ", "_")
    base_filename = f"{image_basename}_{prompt_for_filename}_agent_{llm_config['name']}"
    output_json_path = os.path.join(output_dir, f"{base_filename}_pred.json")

    with open(output_json_path, "r") as f:
        prediction = json.load(f)
    orig_h = int(prediction["orig_img_h"])
    orig_w = int(prediction["orig_img_w"])
    rle_masks = [
                {"size": (orig_h, orig_w), "counts": rle}
                for rle in prediction["pred_masks"]
            ]
    binary_masks = [mask_utils.decode(rle) for rle in rle_masks]
    for i, mask in enumerate(binary_masks):
        mask_uint8 = (mask * 255).astype('uint8')
        Image.fromarray(mask_uint8).save(f"{output_dir}/mask_{i:03d}.png")


    # load model
    tag = "hf"
    config_path = f"checkpoints/{tag}/pipeline.yaml"
    inference = Inference(config_path, compile=False)

    # load image 
    image = load_image(image)  #H*W*3
    masks = [load_mask(str(p)) for p in sorted(Path(output_dir).glob("mask_*.png"))]

    # run model
    outputs = [inference(image, mask, seed=42) for mask in masks]
    scene_gs = make_scene(*outputs)
    # export gaussian splat
    scene_gs.save_ply(f"{output_dir}/splat.ply")
    print("Your reconstruction has been saved to splat.ply")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default='output',required=False)
    parser.add_argument("--model", type=str, default='qwen3_vl_4b_instruct_fp8',required=False)
    parser.add_argument("--llm_server_url", type=str, default='http://10.176.42.36:8001/v1',required=False)
    args = parser.parse_args()
    main(args)

