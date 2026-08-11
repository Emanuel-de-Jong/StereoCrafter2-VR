import gc
import os

import numpy as np
import spaces
import gradio as gr
import torch
from diffusers.training_utils import set_seed

from depthcrafter.depth_crafter_ppl import DepthCrafterPipeline
from depthcrafter.unet import DiffusersUNetSpatioTemporalConditionModelDepthCrafter

import uuid
import random
from huggingface_hub import hf_hub_download

from depthcrafter.utils import read_video_frames, vis_sequence_depth, save_video

examples = [
    ["examples/example_01.mp4", 5, 1.0, 1024, -1, -1],
    ["examples/example_02.mp4", 5, 1.0, 1024, -1, -1],
    ["examples/example_03.mp4", 5, 1.0, 1024, -1, -1],
    ["examples/example_04.mp4", 5, 1.0, 1024, -1, -1],
    ["examples/example_05.mp4", 5, 1.0, 1024, -1, -1],
]


unet = DiffusersUNetSpatioTemporalConditionModelDepthCrafter.from_pretrained(
    "tencent/DepthCrafter",
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
)
pipe = DepthCrafterPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    unet=unet,
    torch_dtype=torch.float16,
    variant="fp16",
)
pipe.to("cuda")


@spaces.GPU(duration=120)
def infer_depth(
    video: str,
    num_denoising_steps: int,
    guidance_scale: float,
    max_res: int = 1024,
    process_length: int = -1,
    target_fps: int = -1,
    save_folder: str = "./demo_output",
    window_size: int = 110,
    overlap: int = 25,
    seed: int = 42,
    track_time: bool = True,
    save_npz: bool = False,
):
    set_seed(seed)
    pipe.enable_xformers_memory_efficient_attention()

    frames, target_fps = read_video_frames(video, process_length, target_fps, max_res)

    with torch.inference_mode():
        res = pipe(
            frames,
            height=frames.shape[1],
            width=frames.shape[2],
            output_type="np",
            guidance_scale=guidance_scale,
            num_inference_steps=num_denoising_steps,
            window_size=window_size,
            overlap=overlap,
            track_time=track_time,
        ).frames[0]
    res = res.sum(-1) / res.shape[-1]
    res = (res - res.min()) / (res.max() - res.min())
    vis = vis_sequence_depth(res)
    save_path = os.path.join(save_folder, os.path.splitext(os.path.basename(video))[0])
    print(f"==> saving results to {save_path}")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if save_npz:
        np.savez_compressed(save_path + ".npz", depth=res)
    save_video(res, save_path + "_depth.mp4", fps=target_fps)
    save_video(vis, save_path + "_vis.mp4", fps=target_fps)
    save_video(frames, save_path + "_input.mp4", fps=target_fps)

    gc.collect()
    torch.cuda.empty_cache()

    return [
        save_path + "_input.mp4",
        save_path + "_vis.mp4",
    ]


def construct_demo():
    with gr.Blocks(analytics_enabled=False) as depthcrafter_iface:
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                input_video = gr.Video(label="Input Video")

            with gr.Column(scale=2):
                with gr.Row(equal_height=True):
                    output_video_1 = gr.Video(
                        label="Preprocessed video",
                        interactive=False,
                        autoplay=True,
                        loop=True,
                        show_share_button=True,
                        scale=5,
                    )
                    output_video_2 = gr.Video(
                        label="Generated Depth Video",
                        interactive=False,
                        autoplay=True,
                        loop=True,
                        show_share_button=True,
                        scale=5,
                    )

        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                with gr.Row(equal_height=False):
                    with gr.Accordion("Advanced Settings", open=False):
                        num_denoising_steps = gr.Slider(
                            label="num denoising steps",
                            minimum=1,
                            maximum=25,
                            value=5,
                            step=1,
                        )
                        guidance_scale = gr.Slider(
                            label="cfg scale",
                            minimum=1.0,
                            maximum=1.2,
                            value=1.0,
                            step=0.1,
                        )
                        max_res = gr.Slider(
                            label="max resolution",
                            minimum=512,
                            maximum=2048,
                            value=1024,
                            step=64,
                        )
                        process_length = gr.Slider(
                            label="process length",
                            minimum=-1,
                            maximum=280,
                            value=60,
                            step=1,
                        )
                        process_target_fps = gr.Slider(
                            label="target FPS",
                            minimum=-1,
                            maximum=30,
                            value=15,
                            step=1,
                        )
                    generate_btn = gr.Button("Generate")
            with gr.Column(scale=2):
                pass

        gr.Examples(
            examples=examples,
            inputs=[
                input_video,
                num_denoising_steps,
                guidance_scale,
                max_res,
                process_length,
                process_target_fps,
            ],
            outputs=[output_video_1, output_video_2],
            fn=infer_depth,
            cache_examples="lazy",
        )
        generate_btn.click(
            fn=infer_depth,
            inputs=[
                input_video,
                num_denoising_steps,
                guidance_scale,
                max_res,
                process_length,
                process_target_fps,
            ],
            outputs=[output_video_1, output_video_2],
        )

    return depthcrafter_iface


if __name__ == "__main__":
    demo = construct_demo()
    demo.queue()
    demo.launch(share=True)
