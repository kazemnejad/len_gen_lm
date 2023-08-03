module load StdEnv/2020 gcc/9.3.0 python/3.9 arrow/11
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index torch torchvision torchtext torchaudio
pip install --no-index transformers datasets evaluate accelerate sentencepiece scikit-learn comet_ml protobuf