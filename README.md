# len_gen_lm
## Prepare Experiment Environment (Only for the first time)
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
4. Fill the environment variables in `env.sh` with
```bash
# We save checkpoints and logs here. It should be shared network storage accessible from all nodes.
export APP_EXP_DIR=/path/to/network/storage/len_gen_lm/exps

# Go to comet.ml and get your API token
export COMET_API_KEY="..."
```
## Train
```bash
./run_training.sh <pe> <size>
```

`<pe>` can be chosen from:
- `alibi`: Alibi
- `none`: NoPE

`<size>` can be chosen from:
- `100m`
- `300m`
- `1b`

### What compute resources should be used?
at least:
- CPU: 6 cores
- Memory: 32GB
- GPU: 1x A100 80gb

it will use all gpus available on the node. So, the more gpus you have, the faster it will be.