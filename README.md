# Voice anti-spoofing LCNN
# was trained with **ASVspoof 2019 Logical Access**

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#results">Results</a> •
  <a href="#license">License</a>
</p>

## About

Spoofing detection for voice based on LightCNN

Task: bibary classification of audio files:
   0 - bonafide(original voice)
   1 - spoof(fake voice)

Architecture: LightCNN-9 with MFM activation, adaptive pooling, and regularization (Dropout and BatchNorm)

model is trained on the dataset **ASVspoof 2019 Logical Access**

## Installation

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `conda` version:

   ```bash
   # create env
   conda create -n project_env python=PYTHON_VERSION

   # activate env
   conda activate project_env
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env/bin/activate
   ```

1. repository cloning

```bash
git clone https://github.com/ddzaliva-web/Voice-Anti-spoofing.git
cd Voice-Anti-spoofing
```

## How To Use

### Training

```bash
python3 train.py -cn=baseline
```

## Inference

```bash
python3 inference.py -cn=inference
```

## CSV Checking

```bash
# Clone the course repository
git clone -b Summer_2026 https://github.com/Blinorot/deep-learning-research.git
cd deep-learning-research/hw

# Prepare the directory structure
mkdir -p students_solutions
mv path/to/dd_zaliva@mail.ru.csv students_solutions/

# Run the grading script
python3 grading.py
```

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

this project is licensed under the MIT License.
