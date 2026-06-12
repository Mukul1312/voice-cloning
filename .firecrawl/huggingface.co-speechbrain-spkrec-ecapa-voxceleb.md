# Speaker Verification with ECAPA-TDNN embeddings on Voxceleb

This repository provides all the necessary tools to perform speaker verification with a pretrained ECAPA-TDNN model using SpeechBrain.
The system can be used to extract speaker embeddings as well.
It is trained on Voxceleb 1+ Voxceleb2 training data.

For a better experience, we encourage you to learn more about
[SpeechBrain](https://speechbrain.github.io/). The model performance on Voxceleb1-test set(Cleaned) is:

| Release | EER(%) |
| :-: | :-: |
| 05-03-21 | 0.80 |

## Pipeline description

This system is composed of an ECAPA-TDNN model. It is a combination of convolutional and residual blocks. The embeddings are extracted using attentive statistical pooling. The system is trained with Additive Margin Softmax Loss. Speaker Verification is performed using cosine distance between speaker embeddings.

## Install SpeechBrain

First of all, please install SpeechBrain with the following command:

```
pip install git+https://github.com/speechbrain/speechbrain.git@develop
```

Please notice that we encourage you to read our tutorials and learn more about
[SpeechBrain](https://speechbrain.github.io/).

### Compute your speaker embeddings

```python
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
signal, fs =torchaudio.load('tests/samples/ASR/spk1_snt1.wav')
embeddings = classifier.encode_batch(signal)
```

The system is trained with recordings sampled at 16kHz (single channel).
The code will automatically normalize your audio (i.e., resampling + mono channel selection) when calling _classify\_file_ if needed. Make sure your input tensor is compliant with the expected sampling rate if you use _encode\_batch_ and _classify\_batch_.

### Perform Speaker Verification

```python
from speechbrain.inference.speaker import SpeakerRecognition
verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb")
score, prediction = verification.verify_files("tests/samples/ASR/spk1_snt1.wav", "tests/samples/ASR/spk2_snt1.wav") # Different Speakers
score, prediction = verification.verify_files("tests/samples/ASR/spk1_snt1.wav", "tests/samples/ASR/spk1_snt2.wav") # Same Speaker
```

The prediction is 1 if the two signals in input are from the same speaker and 0 otherwise.

### Inference on GPU

To perform inference on the GPU, add `run_opts={"device":"cuda"}` when calling the `from_hparams` method.

### Training

The model was trained with SpeechBrain (aa018540).
To train it from scratch follows these steps:

1. Clone SpeechBrain:

```bash
git clone https://github.com/speechbrain/speechbrain/
```

2. Install it:

```
cd speechbrain
pip install -r requirements.txt
pip install -e .
```

3. Run Training:

```
cd  recipes/VoxCeleb/SpeakerRec
python train_speaker_embeddings.py hparams/train_ecapa_tdnn.yaml --data_folder=your_data_folder
```

You can find our training results (models, logs, etc) [here](https://drive.google.com/drive/folders/1-ahC1xeyPinAHp2oAohL-02smNWO41Cc?usp=sharing).

### Limitations

The SpeechBrain team does not provide any warranty on the performance achieved by this model when used on other datasets.

#### Referencing ECAPA-TDNN

```
@inproceedings{DBLP:conf/interspeech/DesplanquesTD20,
  author    = {Brecht Desplanques and
               Jenthe Thienpondt and
               Kris Demuynck},
  editor    = {Helen Meng and
               Bo Xu and
               Thomas Fang Zheng},
  title     = {{ECAPA-TDNN:} Emphasized Channel Attention, Propagation and Aggregation
               in {TDNN} Based Speaker Verification},
  booktitle = {Interspeech 2020},
  pages     = {3830--3834},
  publisher = {{ISCA}},
  year      = {2020},
}
```

# **Citing SpeechBrain**

Please, cite SpeechBrain if you use it for your research or business.

```bibtex
@misc{speechbrain,
  title={{SpeechBrain}: A General-Purpose Speech Toolkit},
  author={Mirco Ravanelli and Titouan Parcollet and Peter Plantinga and Aku Rouhe and Samuele Cornell and Loren Lugosch and Cem Subakan and Nauman Dawalatabad and Abdelwahab Heba and Jianyuan Zhong and Ju-Chieh Chou and Sung-Lin Yeh and Szu-Wei Fu and Chien-Feng Liao and Elena Rastorgueva and François Grondin and William Aris and Hwidong Na and Yan Gao and Renato De Mori and Yoshua Bengio},
  year={2021},
  eprint={2106.04624},
  archivePrefix={arXiv},
  primaryClass={eess.AS},
  note={arXiv:2106.04624}
}
```

# **About SpeechBrain**

- Website: [https://speechbrain.github.io/](https://speechbrain.github.io/)
- Code: [https://github.com/speechbrain/speechbrain/](https://github.com/speechbrain/speechbrain/)
- HuggingFace: [https://huggingface.co/speechbrain/](https://huggingface.co/speechbrain/)

Downloads last month1,906,953

Inference Providers [NEW](https://huggingface.co/docs/inference-providers)

This model isn't deployed by any Inference Provider. [🙋5Ask for provider support](https://huggingface.co/spaces/huggingface/InferenceSupport/discussions/2631)

## Model tree for speechbrain/spkrec-ecapa-voxceleb

Finetunes

[7 models](https://huggingface.co/models?other=base_model:finetune:speechbrain/spkrec-ecapa-voxceleb)

Quantizations

[2 models](https://huggingface.co/models?other=base_model:quantized:speechbrain/spkrec-ecapa-voxceleb)

## Spaces using speechbrain/spkrec-ecapa-voxceleb100

## Paper for speechbrain/spkrec-ecapa-voxceleb

[Paper • 2106.04624 •Published Jun 8, 2021• 2](https://huggingface.co/papers/2106.04624)