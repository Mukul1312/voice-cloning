\-\-\-
language:
\- ab
\- af
\- ak
\- am
\- ar
\- as
\- av
\- ay
\- az
\- ba
\- bm
\- be
\- bn
\- bi
\- bo
\- sh
\- br
\- bg
\- ca
\- cs
\- ce
\- cv
\- ku
\- cy
\- da
\- de
\- dv
\- dz
\- el
\- en
\- eo
\- et
\- eu
\- ee
\- fo
\- fa
\- fj
\- fi
\- fr
\- fy
\- ff
\- ga
\- gl
\- gn
\- gu
\- zh
\- ht
\- ha
\- he
\- hi
\- sh
\- hu
\- hy
\- ig
\- ia
\- ms
\- is
\- it
\- jv
\- ja
\- kn
\- ka
\- kk
\- kr
\- km
\- ki
\- rw
\- ky
\- ko
\- kv
\- lo
\- la
\- lv
\- ln
\- lt
\- lb
\- lg
\- mh
\- ml
\- mr
\- ms
\- mk
\- mg
\- mt
\- mn
\- mi
\- my
\- zh
\- nl
\- 'no'
\- 'no'
\- ne
\- ny
\- oc
\- om
\- or
\- os
\- pa
\- pl
\- pt
\- ms
\- ps
\- qu
\- qu
\- qu
\- qu
\- qu
\- qu
\- qu
\- qu
\- qu
\- qu
\- qu
\- qu
\- qu
\- qu
\- qu
\- qu
\- qu
\- qu
\- qu
\- qu
\- qu
\- qu
\- ro
\- rn
\- ru
\- sg
\- sk
\- sl
\- sm
\- sn
\- sd
\- so
\- es
\- sq
\- su
\- sv
\- sw
\- ta
\- tt
\- te
\- tg
\- tl
\- th
\- ti
\- ts
\- tr
\- uk
\- ms
\- vi
\- wo
\- xh
\- ms
\- yo
\- ms
\- zu
\- za
license: cc-by-nc-4.0
tags:
\- mms
\- wav2vec2
\- audio
\- voice
\- speech
\- forced-alignment
pipeline\_tag: automatic-speech-recognition
\-\-\-

\# Forced Alignment with Hugging Face CTC Models
This Python package provides an efficient way to perform forced alignment between text and audio using Hugging Face's pretrained models. it also features an improved implementation to use much less memory than TorchAudio forced alignment API.

The model checkpoint uploaded here is a conversion from torchaudio to HF Transformers for the MMS-300M checkpoint trained on forced alignment dataset

\## Installation

\`\`\`bash
pip install git+https://github.com/MahmoudAshraf97/ctc-forced-aligner.git
\`\`\`
\## Usage

\`\`\`python
import torch
from ctc\_forced\_aligner import (
 load\_audio,
 load\_alignment\_model,
 generate\_emissions,
 preprocess\_text,
 get\_alignments,
 get\_spans,
 postprocess\_results,
)

audio\_path = "your/audio/path"
text\_path = "your/text/path"
language = "iso" # ISO-639-3 Language code
device = "cuda" if torch.cuda.is\_available() else "cpu"
batch\_size = 16

alignment\_model, alignment\_tokenizer = load\_alignment\_model(
 device,
 dtype=torch.float16 if device == "cuda" else torch.float32,
)

audio\_waveform = load\_audio(audio\_path, alignment\_model.dtype, alignment\_model.device)

with open(text\_path, "r") as f:
 lines = f.readlines()
text = "".join(line for line in lines).replace("\\n", " ").strip()

emissions, stride = generate\_emissions(
 alignment\_model, audio\_waveform, batch\_size=batch\_size
)

tokens\_starred, text\_starred = preprocess\_text(
 text,
 romanize=True,
 language=language,
)

segments, scores, blank\_token = get\_alignments(
 emissions,
 tokens\_starred,
 alignment\_tokenizer,
)

spans = get\_spans(tokens\_starred, segments, blank\_token)

word\_timestamps = postprocess\_results(text\_starred, spans, stride, scores)
\`\`\`