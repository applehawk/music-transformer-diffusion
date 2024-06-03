## Song Generation

Song generation models include difference kinds:
- Generation music (Meta/MaGNET)
- Generation instrumental music (MusicGen model).
- Generation mix of music+vocal (Jukebox model).
- Generation vocal/signing (Bark model).
- Generation audio (audio-effects) (AudioGen model).
- Generation song (text-to-lyrics+music+vocal) (Suno.Ai).
- Generation composition song (Muzic/Microsoft family models)

SOTA Approaches based on Transformer models and Diffusion models.

## How work with repository

Install prerequisites with command 'pip install -r requirements.txt'

Example of Training hierachical transfomer model of music generator presented in "example-train-tr-model.ipynb".

Example of Generation music with HuggingFace models availabe in "hf-music-gen-models".

Example of generated samples placed in "generated-samples" folder.

### Architecture/Method Each Models
- MusicGen Model, based on https://arxiv.org/pdf/2306.05284:
 - Architecture:
   - Audio Tokenization:
     - Converts audio into quantized tokens using [RVQ](https://arxiv.org/abs/2107.03312) (EnCodec).
     - EnCodec: Convolutional Auto-encoder with latent space quantized using Residual Vector Quantization (RVQ) and an adversarial reconstruction loss. EnCodec tokenizer with 4 codebooks sampled at 50 Hz.
    - Fours codebooks:
      - Codebook 1: Captures the high-level structure and broad aspects of the audio.
      - Codebook 2: Focuses on intermediate features, refining the details provided by the first codebook.
      - Codebook 3: Provides additional detail, working on the nuances not captured by the previous codebooks.
      - Codebook 4: Adds the final layer of detail, ensuring high-fidelity audio output.
   - Transformer single stage auto-regressive model trained over a 32kHz.
   - Transformer Decoder: An autoregressive model conditioned on text or melody.
 - Training:
   - use 20K hours of licensed music to train MusicGen.
   - internal dataset of 10K high-quality music tracks.
   - on the ShutterStock and Pond5 music data.
 - Samples page: https://ai.honu.io/papers/musicgen/
- Bark model (suno.ai): https://github.com/suno-ai/bark
 - As for text-to- speech synthesis, we leverage the Bark (Suno, 2023) model, which can generate realistic speech and is able to match the tone, pitch, emotion, and prosody of a given voice preset.
  - Architecture:
    - three Transformer models: coarse, text, fine. The same approach hierarchical modeling, described in [AudioML](https://arxiv.org/pdf/2209.03143).
      - Each Transformer model based on nanoGPT transformer.
- JukeBox model: (https://jukebox.openai.com/):
 - Architecture:
    - Audio Tokenization:
      - To compress Audio to lower dimension space are used 3 separate hierarchical [VQ-VAE](https://arxiv.org/pdf/1711.00937) models.
       - Three cascade models.
    - autoregresive Sparse Transformers.
    - autoregressive upsamplers to recreate the lost information at each level of compression.
 - Training:
    - For the music VQ-VAE, we use 3 levels of bottlenecks compressing 44 kHz audio in dimensionality by 8x, 32x, and 128x.
    - Codebooks size of 2048 for each level.
    - The VQ-VAE has 2 million parameters and is trained on 9-second audio clips on 256 V100 for 3 days.

### How improve quality of generation?

1. With help LLM generate text-to-prompts. Each prompt consist more precisly with much more characteristis of music batches. It can more detailed by including the instrument, tempo, genre, or emotion.
 - text-to-prompt (with instrumnets, genre, composition division)
 - text-to-lyrics (it can be part of general prompt).
In this case, we can generate music on prompt[+lyrics]-to-music[+singing] approach, compositionaly.

In song there are parts, so model should be trained generate music with understading [Position] in composition part of generation:
- Introduction/intro;
- Verse;
- Prechorus/bridge;
- Chorus;
- Post Chorus/Tag;
- Losing Break;
- Ending/morning.

2. Improve traning approach. We can train model based on separated audio and sining tracks, we help Demux model.
- need to investigate audio splitting costs for large example datasets to build a training dataset.
- train on (text, music[-singing]) and on (text, singing[-music]).
- we can use CLAP in labeling training audio. It can be instructed in to predict the most relevant text snippet, given an audio, without directly optimizing for the task. 

3. Architecture Imrovements:
 - use non-autoregression models (such in MagNET model), which shows 7-10 highly performance in inference.
 - use LoRA in Transformer model, make training faster and cheaper.
 - use Diffusion model approaches + Transformer model approach.
 - use combination Diffusion model and Transformer model.

### Examples of Dataset (MusicCap)

## References

### General models
Papers: ComputerScience.Sound (cs.SD) https://arxiv.org/list/cs.SD/recent

## Transformer pased generation:
(MagNET) Masked Audio Generation using a Single Non-Autoregressive Transformer
- Paper: https://arxiv.org/abs/2401.04577
- ModelCard: https://huggingface.co/models?other=magnet
- Samples: https://pages.cs.huji.ac.il/adiyoss-lab/MAGNeT/

Jukebox: A Generative Model for Music
- Paper: https://arxiv.org/pdf/2005.00341 (2020/OpenAI)
- Samples: https://jukebox.openai.com/?song=804331648

Simple and Controllable Music Generation (MusicGen)
- Paper: https://arxiv.org/pdf/2306.05284 (2024/MetaAI)
- Samples: https://audiocraft.metademolab.com/musicgen.html

MusicLM: Generating Music From Text
- Paper: https://arxiv.org/pdf/2301.11325 (2023/Google)
- Samples: https://google-research.github.io/seanet/musiclm/examples/

Music Transformer
- Paper: https://arxiv.org/pdf/1809.04281 (2018/Google Brain)

## Diffusion based generation:

Noise2Music: Text-conditioned Music Generation with Diffusion Models
- Paper: https://arxiv.org/abs/2302.03917 (2023)

### Vocal & Singing

Bark/SunoAI (Transformer based):
- Source: https://github.com/suno-ai/bark

RapVerse (scaling autoregressive multimodal transformers): Coherent Vocals and Whole-Body Motions Generations from Text
https://arxiv.org/pdf/2405.20336

### Audio Neural Codecs
High Fidelity Neural Audio Compression (EnCodec) - https://arxiv.org/pdf/2210.13438

### Convoluntional Models Audio Generation
WaveNet: A Generative Model for Raw Audio - https://arxiv.org/pdf/1609.03499

### Compositional Audio
WavJourney: Compositional Audio Creation with Large Language Models - https://arxiv.org/pdf/2307.14335