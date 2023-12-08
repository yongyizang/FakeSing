# FakeSing: Enhance Your Live Vocal Performance with Real-Time Voice Tuning Against Pre-Recorded Track

DISCLAIMER: This project is a light-hearted, conceptual creation **meant for amusement** and not intended for professional use. It's a quick proof of concept with details that haven't been fully refined. Employ it at your own discretion!

## Introduction

Picture yourself as a seasoned singer facing the challenge of performing songs you wrote a decade or two ago. How do you adapt? Playing a pre-recorded vocal track in a live show might give it away, as after source separation, your voice's fundamental frequency might match the original track from your albums. Plus, there's the risk of lip syncing inaccurately. What's the solution?

This repository offers a conceptual tool that leverages a pre-recorded track and employs dynamic time warping (DTW) to determine your current position in the song in real-time. It's designed to be forgiving, allowing for minor errors with a customizable search buffer. The tool aims to match your voice to the reference audio frame's pitch, ensuring a natural sound without the overt effects of auto-tuning. It's our little secret!

Tested on an M3 Max Macbook Pro, this script achieves a Real-Time Factor (RTF) of 0.12, making it viable for live use!

## Getting Started

Begin by installing the required packages:
```
pip install numpy librosa soundfile dtw matplotlib tqdm
```

Next, execute the main script to generate visualizations of the fundamental pitch. 

Note: This repository **does not** offer actual pitch-shifting functionality. 
```
python main.py
```

Feel free to adjust the hyper-parameters in `main.py` as needed.