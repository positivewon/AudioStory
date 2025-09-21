import warnings
warnings.filterwarnings("ignore")

# ===== Standard libs =====
import os
import re
import json
import time
import random
import logging
import shutil
import argparse
from datetime import datetime
from functools import partial
from typing import List, Tuple

# ===== Numeric / DL =====
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torch.distributed as dist
import torchdata.datapipes as dp

# ===== Audio/Video/Image =====
import librosa
import soundfile as sf
from PIL import Image
from pydub import AudioSegment
from moviepy import VideoFileClip, AudioFileClip
from tqdm import tqdm

# ===== Model / Config =====
import hydra
import pyrootutils
from omegaconf import OmegaConf
from safetensors.torch import load_file
from diffusers import AutoencoderOobleck
from transformers import WhisperFeatureExtractor, AutoModelForCausalLM, AutoTokenizer

# ===== Project model =====
from src.models.detokenizer.modeling_flux_gradio import TangoFlux

# ===== Google GenAI =====
os.environ["GOOGLE_API_KEY"] = ""  # NOTE: keep as your current setup 
from google import genai
from google.genai.types import HttpOptions 
import google.genai.types as types

# ===== UI =====
import gradio as gr

# ===== (Optional) misc http client placeholders =====
import requests
from venus_api_base.http_client import HttpClient
from venus_api_base.config import Config
import base64

# ---------------- Constants ----------------
BOI_TOKEN = '<t5>'
EOI_TOKEN = '</t5>'
AUD_TOKEN = '<t5_{:05d}>'

QWEN_BOS_INDEX = 151644
QWEN_EOS_INDEX = 151645

device = 'cuda:0'
dtype = torch.float16

MELBINS = 128
TARGET_LEN = 1024

EXAMPLE_VIDEO_PATHS = [
    "demos/examples/cats1.mp4",
    "demos/examples/duck1.mp4",
    "demos/examples/sora1.mp4",
    "demos/examples/bear1.mp4",
]

OVERLOADED_MESSAGE = """The model is overloaded. You can try the following:
1. Click the "Generate" button again to attempt re-generation.
2. Compress the video size, then re-upload and generate.
3. Replace the video with a different one, then re-upload and generate."""


def resize_video(video_path, output_path, max_size=448):
    # using ffmpeg to resize the video
    import subprocess
    import os

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(output_path):
        try:
            os.remove(output_path)
            print(f"Removed existing file: {output_path}")
        except OSError as e:
            print(f"Error removing existing file {output_path}: {e}")

    # Run ffmpeg command to resize the video
    command = [
        'ffmpeg', '-i', video_path, '-vf', f'scale={max_size}:-1', '-c:a', 'copy', output_path
    ]
    subprocess.run(command, check=True)
    print(f'Resized video saved to {output_path}')
    return output_path

    
# ---------------- Utilities ----------------
def chat_with_multi_modal(model: str, prompt: str, bucket_name: str, video_path: str, mimeType: str = 'video/mp4') -> str:
    """
    Call Gemini 2.5 Pro with a video + text prompt; return response text.
    """
    with open(video_path, 'rb') as f:
        video_bytes = f.read()

    client = genai.Client(http_options=HttpOptions(api_version="v1"))

    try:
        response = client.models.generate_content(
            model='models/gemini-2.5-pro',
            contents=types.Content(
                parts=[
                    types.Part(inline_data=types.Blob(data=video_bytes, mime_type='video/mp4')),
                    types.Part(text=prompt)
                ]
            )
        )
        print('===============================================')
        print('response:')
        print(response.text)
        print('===============================================')
        return response.text
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        status_code = None
        if hasattr(e, 'status_code'):
            status_code = e.status_code
        elif hasattr(e, 'response') and hasattr(e.response, 'status_code'):
            status_code = e.response.status_code
        elif hasattr(e, 'code'):
            status_code = e.code
        
        if status_code == 503:
            print("Detected 503 UNAVAILABLE error - model is overloaded")
            raise Exception("MODEL_OVERLOADED") from e
        raise



def smooth_concatenate(audio_tensors: List[torch.Tensor], sample_rate: int, transition_sec: float = 3.0) -> torch.Tensor:
    """
    Concatenate multiple audio tensors with a simple cross-fade.
    Each tensor should be [channels, samples].
    """
    if not audio_tensors:
        return torch.tensor([])

    audio_tensors = [a if a.dim() == 2 else a.unsqueeze(0) for a in audio_tensors]

    transition_samples = int(transition_sec * sample_rate)
    result = audio_tensors[0]

    for next_audio in audio_tensors[1:]:
        if result.shape[1] < transition_samples or next_audio.shape[1] < transition_samples:
            result = torch.cat([result, next_audio], dim=1)
            continue

        overlap_result = result[:, -transition_samples:]
        overlap_next = next_audio[:, :transition_samples]

        fade_out = torch.linspace(1, 0, transition_samples)
        fade_in = torch.linspace(0, 1, transition_samples)

        overlap_result = overlap_result * fade_out
        overlap_next = overlap_next * fade_in
        merged = overlap_result + overlap_next

        result = torch.cat([result[:, :-transition_samples], merged, next_audio[:, transition_samples:]], dim=1)

    return result


def process_and_save_audio(audio_tensors: List[torch.Tensor],
                           output_path: str,
                           original_sample_rate: int = 44100,
                           transition_sec: float = 3.0) -> torch.Tensor:
    """
    Build concatenated audio with transitions; return the waveform tensor.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    concatenated_audio = smooth_concatenate(audio_tensors, original_sample_rate, transition_sec)
    return concatenated_audio


def merge_audio_video(audio_path: str,
                      video_path: str,
                      output_path: str,
                      audio_sampling_rate: int = 44100,
                      overwrite: bool = False) -> str:
    """
    Replace video's audio with the generated audio and save a new mp4.
    """
    try:
        with VideoFileClip(video_path) as video_clip:
            audio_clip = AudioFileClip(audio_path)
            final_clip = video_clip.with_audio(audio_clip)
            final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
    except Exception as e:
        print(f"ERROR occurred when merging video: {e}")
    return output_path


# ---------------- Logging redirect ----------------
class PrintToLog:
    def write(self, message):
        if message.strip():
            logging.info(message.strip())

    def flush(self):
        pass


# ---------------- Model init ----------------
def init_audiostory():
    """
    Initialize TangoFlux and VAE models.
    """
    print('****************************')
    print('Begin to init AudioStory...')
    tangoflux_model = 'ckpt/AudioStory_detokenizer'

    # Load TangoFlux
    with open(f"{tangoflux_model}/config.json") as f:
        flux_config = json.load(f)
    flux_model = TangoFlux(config=flux_config)
    weights = load_file(f"{tangoflux_model}/model_1.safetensors")
    flux_model.load_state_dict(weights, strict=False)
    flux_model.eval().to(device, dtype=dtype)

    # Load VAE
    vae = AutoencoderOobleck()
    weights = load_file(f"{tangoflux_model}/model.safetensors")
    vae.load_state_dict(weights, strict=True)
    vae.eval().to(device, dtype=dtype)

    print('Init vae model Done')
    print('Init AudioStory Done')
    print('****************************')
    return flux_model, vae


# ---------------- Core dubbing pipeline ----------------
def video_dubbing(video_path: str, guidance: float, step: int):
    """
    1) Ask Gemini to segment video and return JSON {reasoning, audio_clips}
    2) Generate audio clips with TangoFlux + VAE
    3) Concatenate audio and mux back to the video
    4) Return: output_video_path, output_reasoning(str), combined_captions(str)
    """
    save_dir_concat = 'demos/gradio_generated'
    os.makedirs(save_dir_concat, exist_ok=True)
    random.seed(0)

    model = 'gemini-2.5-pro'
    object_key = video_path

    prompt = """
    Please watch a video and identify the temporal consecutive key events or scene changes in the video. When encountering different events, scene transitions, and exaggerated actions — including very brief movements like a 1-second scream, chasing, mouth movements, shouting, collisions, falling, etc. — each should be divided into a separate segment. For each segment, please provide background music associations that are in line with visual information. All time expressions should use seconds as a single unit (e.g., 130s) and avoid composite units like minutes:seconds (e.g., 2:10s). The results should be returned in JSON format and should include only the following two keys: 
    (1) "reasoning": Follow the narrative style of natural language. First, state the total duration of the entire video in the format of x.x seconds, and state how many consecutive key events the video can be divided into. And then elaborate on each point. Each point should first present a brief title and specify the time interval as [start_timestamp, end_timestamp](start_timestamp and end_timestamp are both floats with one decimal place.). **The time interval of each segment must be within 18 seconds and all segments should be consecutive. The total duration should be consistent with that of the original video.** Then state the Story Details and the derived Sound Effects. Each point MUST be within 20 words. 
    (2) "audio_clips": Imagine the background music or significant musical segments according to the visual content of each consecutive segment from 0.0 second. Output the "start_timestamp" and "end_timestamp", as well as "caption" for each music segment. The "caption" (MUST be within 32 words) should conduct a detailed description from four aspects: musical elements, rhythmic characteristics, connection with the picture, and atmosphere creation, aimed at high-quality sound synthesis. Here is an example: "A quiet, stealthy orchestral piece with tiptoeing pizzicato strings and a cautious, mischievous bassoon melody. Slow, deliberate rhythm builds suspense, matching Jerry's hesitant approach, creating an atmosphere of quiet tension and anticipation." If there are exaggerated actions such as chasing, mouth movements, shouting, collisions, falling, etc., additional explanations are required. The style should conform to the style of the cartoon "Tom and Jerry". 
    """
    time.sleep(10)


    # ret = chat_with_multi_modal(model=model, prompt=prompt, bucket_name='venus-gcp-aigc', video_path=object_key)
    try:
        ret = chat_with_multi_modal(model=model, prompt=prompt, bucket_name='venus-gcp-aigc', video_path=object_key)
    except Exception as e:
        # Check for our specific model overloaded exception
        if str(e) == "MODEL_OVERLOADED":
            # Re-raise to be caught in generate_video
            raise
        # Handle other exceptions
        print(f"Error in chat_with_multi_modal: {e}")
        raise


    # NOTE: retain original slicing; assumes response has surrounding tokens to strip.
    ret = ret[8:-3]
    element = json.loads(ret)

    audio_clips = element['audio_clips']
    output_reasoning = element['reasoning']

    with torch.no_grad():
        multi_audio_num = len(audio_clips)
        audio_tensors = []
        whole_duration = 0.0
        audio_clips_final = []

        for i, item in enumerate(audio_clips, start=1):
            caption = item['caption']
            duration = round(float(item['end_timestamp']) - float(item['start_timestamp']), 1)
            final_caption = f"{i}: [{item['start_timestamp']} - {item['end_timestamp']}]: {caption}\n"
            audio_clips_final.append(final_caption)

            print(f"Caption_{i}: {caption}")

            # Add 1s except the last one, to improve clip transitions
            duration = duration + 1 if i < len(audio_clips) else duration
            whole_duration += duration
            print(f"Caption_{i}_duration: {duration}")

            output_latents = flux_model.inference_flow(
                caption,
                duration=duration,
                num_inference_steps=step,
                guidance_scale=guidance,
            )
            wave = vae.decode(output_latents.transpose(2, 1)).sample.cpu()[0]
            waveform_end = int(duration * vae.config.sampling_rate)
            wave = wave[:, :waveform_end]

            wave_resampled = wave.numpy().astype('float')
            wave_resampled = torch.tensor(wave_resampled)
            audio_tensors.append(wave_resampled)

            torchaudio.save(f"{save_dir_concat}/output_wav_{i}.wav", wave_resampled, sample_rate=44100)

        print(f"whole duration: {whole_duration}")

        concatenated_audio = process_and_save_audio(
            audio_tensors, save_dir_concat, original_sample_rate=44100, transition_sec=1
        )
        print('vae.config.sampling_rate:', vae.config.sampling_rate)

        waveform_end = int(whole_duration * vae.config.sampling_rate)
        concatenated_audio = concatenated_audio[:, :waveform_end]  # ensure [C, T]
        output_audio_path = f"{save_dir_concat}/output_wav_whole.wav"
        torchaudio.save(output_audio_path, concatenated_audio, sample_rate=44100)

        output_video_path = merge_audio_video(
            audio_path=output_audio_path,
            video_path=video_path,
            output_path="demos/gradio_generated/gradio_output_merged_video.mp4",
            audio_sampling_rate=44100
        )

    # cleanup temp
    os.remove(video_path)
    print('Output video path:', output_video_path)

    combined_captions = ''.join(audio_clips_final)
    return output_video_path, output_reasoning, combined_captions


# ---------------- Gradio integration ----------------
def generate_video(steps, guidance_scale, video_input):
    """
    Generate video + storyline output for UI.
    Returns:
        generated_video_path, storyline_text
    """
    if video_input is None:
        return None, None

    # Accept both str path and temp File object
    input_path = video_input if isinstance(video_input, str) else getattr(video_input, "name", None)
    if not input_path or not os.path.exists(input_path):
        print(f"ERROR: files not existing - {input_path}")
        return None, "ERROR: input video file not found."

    # copy to working path
    temp_original_video = 'demos/gradio_generated/temp.mp4'
    shutil.copy(input_path, temp_original_video)

    compressed_video_path = 'demos/gradio_generated/temp_compressed.mp4'
    try:
        resize_video(
            video_path=temp_original_video,
            output_path=compressed_video_path,
            max_size=336 
        )
    except Exception as e:
        print(f"Video Compressing ERROR: {e}")
        compressed_video_path = temp_original_video

    try:
        generated_video, output_reasoning, combined_captions = video_dubbing(
            video_path=temp_original_video,
            guidance=guidance_scale,
            step=steps
        )
        print('Generated video finished!')

        # ====== Combine the two text outputs into one storyline with a blank line ======
        parts = []
        if output_reasoning:
            parts.append(str(output_reasoning).strip())
        if combined_captions:
            parts.append(str(combined_captions).strip())
        storyline_text = "\n\n".join(parts) if parts else ""

        return generated_video, storyline_text

    except Exception as e:
        print(f"ERROR occurred: {e}")
        # Check specifically for the model overloaded case
        if str(e) == "MODEL_OVERLOADED":
            return None, OVERLOADED_MESSAGE
        # For other errors, return the error message
        err = f'ERROR occurred during generation. {e}'
        return None, err


def clear_all():
    """
    Reset inputs/outputs for the UI.
    Returns should match the order of outputs in clear_btn.click.
    """
    temp_files = [
        'demos/gradio_generated/temp_original.mp4',
        'demos/gradio_generated/temp_compressed.mp4',
        'demos/gradio_generated/gradio_output_merged_video.mp4'
    ]
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
                print(f"Cleared temporary file: {temp_file}")
            except OSError as e:
                print(f"Error clearing temporary file {temp_file}: {e}")

    return None, 50, 4.0, None, None


# ============== Load models (global for pipeline) ==============
flux_model, vae = init_audiostory()


# ============== Gradio App ==============
with gr.Blocks(
    theme=gr.themes.Soft(),
    css="""
        .gr-button { border-radius: 8px; }
        .gr-button:hover { background-color: #ffcc00; }
        #output_storyline { font-family: 'Roboto', Arial, sans-serif; white-space: pre-wrap; }
    """
) as demo:
    gr.Markdown("# 🎶🤖 AudioStory: Generating Long-Form Narrative Audio with Large Language Models")
    gr.Markdown("## 🎞️ Video Dubbing")
    gr.Markdown("✅ Select and upload a video, then add music in the style of **Tom and Jerry** to it!")

    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="Input Video")

            gr.Examples(
                examples=EXAMPLE_VIDEO_PATHS,  
                inputs=video_input,            
                label="📺 Example Videos (Click to Use)",  
                examples_per_page=2,
                cache_examples=False
            )

            random_btn = gr.Button("Random Choose a Video")  # (Optional) not wired here
            steps = gr.Slider(minimum=1, maximum=100, value=50, label="Steps")
            guidance_scale = gr.Slider(minimum=0.1, maximum=10, value=4.0, label="Guidance Scale")

            with gr.Row():
                clear_btn = gr.Button("Clear")
                submit_btn = gr.Button("Generate", variant="primary")

        with gr.Column(scale=2):
            output_video = gr.Video(label="Generated Video")
            # ===== Replace two boxes with ONE merged storyline box =====
            output_storyline = gr.Textbox(label="Generated Storyline", lines=10, elem_id="output_storyline")
            

    # Submit: now only two outputs (video + merged storyline)
    submit_btn.click(
        fn=generate_video,
        inputs=[steps, guidance_scale, video_input],
        outputs=[output_video, output_storyline]
    )

    # Clear: return values must match outputs list here
    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[video_input, steps, guidance_scale, output_video, output_storyline]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=10556)
    parser.add_argument('--server', type=str, default='0.0.0.0')
    args = parser.parse_args()

    allowed_paths = ['demos/gradio_generated', 'demos/examples']

    demo.launch(server_name=args.server, share=False, server_port=args.port, allowed_paths=allowed_paths)