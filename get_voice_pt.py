import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import glob
import os
import json


cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False, use_flow_cache=False)


# 指定文件夹路径
folder_path = "/content/wavs"
prompt_path = "/content/prompt_text.json"

# 使用 glob 模式匹配 .wav 文件
wav_files = glob.glob(os.path.join(folder_path, "*.wav"))

prompt_dict = {}
with open(prompt_path, 'r') as f:
    prompt_dict = json.load(f)


# 遍历匹配到的文件
for file_path in wav_files:
    prompt_speech_16k = load_wav(file_path, 16000)
    index = file_path.split('/')[-1].split('.')[0]
    prompt_text = prompt_dict[index]
    cosyvoice.save_zero_shot_spk(prompt_text, prompt_speech_16k, index)
    print(f'{index=}, {prompt_text=}')
