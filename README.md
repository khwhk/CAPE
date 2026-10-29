# CAPE

Condition-Aligned Probabilistic Modeling for Time Series Forecasting with Exogenous Variables.

## Installation

```bash
git clone https://github.com/khwhk/CAPE.git
cd CAPE
conda env create -f environment.yml
conda activate cape
```

Alternatively:

```bash
conda create -n cape python=3.8 -y
conda activate cape
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Data

Download the [datasets](https://drive.google.com/file/d/1K2AvogpOpSz1PiQ53dPchzGv_PqlCWAK/view?usp=sharing)
and place the CSV files in `dataset/forecasting/`.

Experimental settings follow [DAG](https://github.com/decisionintelligence/DAG).

## Run CAPE

```bash
# Known future exogenous variables
bash scripts/w_future/Sdwpfh1.sh short
bash scripts/w_future/Sdwpfh1.sh long

# Historical exogenous variables only
bash scripts/wo_future/Sdwpfh1.sh short
bash scripts/wo_future/Sdwpfh1.sh long

# Choose a GPU
CUDA_VISIBLE_DEVICES=1 bash scripts/wo_future/Energy.sh long
```

Both script directories include `NP`, `PJM`, `BE`, `FR`, `DE`, `Energy`,
`Sdwpfm1`, `Sdwpfm2`, `Sdwpfh1`, `Sdwpfh2`, `Colbun`, and `Rapel`.
Each command runs the stored best configuration with seeds **1, 2, 3**.

The shared entry point supports the same settings:

```bash
bash scripts/w_future/run.sh Colbun short
bash scripts/wo_future/run.sh Colbun long

python run.py --mode wo_future --dataset NP --length short --seeds 1 2 3 \
  --evaluate-test --output-dir results/run_2
```

Results are saved under `results/<mode>/<dataset>/<length>/`.
`aggregate.json` contains the mean and standard deviation of MSE, MAE, CRPS, and QICE.
Use a fresh `--output-dir` when rerunning.

```bash
# Evaluate a saved final-stage checkpoint
python run.py --mode wo_future --dataset NP --length short --seeds 1 \
  --checkpoint path/to/model.pt --evaluate-test --output-dir results/evaluation

# Small CPU pipeline check
python run.py --mode w_future --dataset Colbun --length short \
  --device cpu --seeds 1 --smoke --evaluate-test --output-dir results/smoke
```

## Run baselines

```bash
# Download baseline implementations and install their dependencies
python scripts/baselines/run.py --model DAG --setup-only
pip install -r external/DAG/requirements.txt

# MODEL.sh MODE DATASET LENGTH
bash scripts/baselines/DAG.sh w_future Sdwpfh1 short
bash scripts/baselines/KITE.sh wo_future Sdwpfh1 long
bash scripts/baselines/GCGNet.sh w_future NP short
bash scripts/baselines/TimeXer.sh wo_future NP long
bash scripts/baselines/PatchTST.sh w_future Colbun short
bash scripts/baselines/DLinear.sh wo_future Colbun long

# Choose a GPU
bash scripts/baselines/KITE.sh w_future FR short --gpu 1
```

Baseline scripts also run seeds **1, 2, 3**. Outputs are saved under `results/baselines/`.
