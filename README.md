# len_gen_lm
## Train
1. Clone this repository
```bash
git clone git@github.com:kazemnejad/len_gen_lm.git
```
2. Create a conda environment
```bash
conda create -n len_gen_lm python=3.9
conda activate len_gen_lm
```
3. Install requirements
```bash
# Install pytorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
# Install other requirements
pip install -r requirements.txt
```
4. Copy `trainer_script.sh.template` to `trainer_script.sh` and edit `APP_EXP_DIR`, `TRANSFORMERS_CACHE`, `HF_DATASETS_CACHE`, and `NEPTUNE_API_TOKEN` variables.
Note that `trainer_script.sh` will not be tracked by git.
```bash
cp trainer_script.sh.template trainer_script.sh
nano trainer_script.sh
```
5. Submit jobs with `trainer_script.sh`. (Enable `len_gen_lm` conda environment for the job.)
```bash
./trainer_script.sh <PE_TYPE>
```

`<PE_TYPE>` can be chosen from:
- `t5_relative_bias`: T5 with relative bias
- `rotary`: Rotary
- `abs_sinusoid`: Absolute sinusoid
- `alibi`: Alibi
- `none`: NoPE

### What compute resources should be used?
- GPU: 2x A100 40GB (or 80GB). V100 doesn't work. 1x A100 80GB works but it's going to be slow.
- CPU: 6 cores
- Memory: 32GB

## Inference
4. Copy `inference_script.sh.template` to `inference_script.sh` and edit `APP_EXP_DIR`, `TRANSFORMERS_CACHE`, `HF_DATASETS_CACHE`, and `NEPTUNE_API_TOKEN` variables. Make sure these variables are the same as the ones in `trainer_script.sh`.
Note that `inference_script.sh` will not be tracked by git.
```bash
cp inference_script.sh.template inference_script.sh
nano inference_script.sh
```
5. Submit jobs with `inference_script.sh`. (Enable `len_gen_lm` conda environment for the job.)
```bash
./inference_script.sh <PE_TYPE>
```

`<PE_TYPE>` can be chosen from:
- `t5_relative_bias`: T5 with relative bias
- `rotary`: Rotary
- `abs_sinusoid`: Absolute sinusoid
- `alibi`: Alibi
- `none`: NoPE

### What compute resources should be used?
- GPU: **1x A100** 40GB (or 80GB). V100 doesn't work.
- CPU: 6 cores
- Memory: 32GB



