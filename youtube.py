from faster_whisper import WhisperModel
from tqdm import tqdm
import yt_dlp
import time
import numpy as np
import pysubs2

# @param ["base","small","medium", "large-v1","large-v2"]
model_size = "large-v2"

url = input("Please type your YouTube Video Link Here and Press Enter : ")


video_path_local_list = []

# Run on GPU with FP16
model = WhisperModel(model_size)
ydl_opts = {
  'format': 'm4a/bestaudio/best',
  'outtmpl': './.tmp/%(id)s.%(ext)s',
  # ℹ️ See help(yt_dlp.postprocessor) for a list of available Postprocessors and their arguments
  'postprocessors': [{  # Extract audio using ffmpeg
    'key': 'FFmpegExtractAudio',
    'preferredcodec': 'wav',
  }],
}

# get name from url
name = url.split("=")[-1]

filename = "audio.wav"
title = "audio"

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    error_code = ydl.download(url)
    video_info = ydl.extract_info(url, download=False)
    title = video_info['title']
    filename = f"{video_info['id']}.wav"

input_directory = ".tmp"
output_directory = "output"
input_file = f"{input_directory}/{filename}"

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")
tic = time.time()

segments, info = model.transcribe(input_file, beam_size=5)
# segments is a generator so the transcription only starts when you iterate over it
# to use pysubs2, the argument must be a segment list-of-dicts
# Same precision as the Whisper timestamps.
total_duration = round(info.duration, 2)
results = []
with tqdm(total=total_duration, unit=" seconds") as pbar:
    for s in segments:
        segment_dict = {'start': s.start, 'end': s.end, 'text': s.text}
        results.append(segment_dict)
        segment_duration = s.end - s.start
        pbar.update(segment_duration)


# Time comsumed
toc = time.time()
print('识别完毕 Done')
print(f'Time consumpution {toc-tic}s')

srt_file = f"{output_directory}/{title}.srt"
subs = pysubs2.load_from_whisper(results)
subs.save(srt_file)
