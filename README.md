# LAD

Probabilistic time series forecasting with exogenous variables, using lag alignment
and state-conditioned diffusion response. No pretrained forecasting model is required.

## 1. Installation

Tested with Python 3.8, PyTorch 2.4.1 and CUDA 12.1.

```bash
conda create -n lad python=3.8 -y
conda activate lad
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## 2. Data

Download the [preprocessed datasets](https://drive.google.com/file/d/1K2AvogpOpSz1PiQ53dPchzGv_PqlCWAK/view?usp=sharing)
and place their CSV files under `dataset/forecasting/`.

Experimental settings follow [DAG](https://github.com/decisionintelligence/DAG).

## 3. Run

```bash
# Known future covariates; three seeds, short horizon
bash scripts/w_future/run.sh Sdwpfh1 short

# Historical covariates only
bash scripts/wo_future/run.sh Sdwpfh1 short

# Long horizon on another GPU
CUDA_VISIBLE_DEVICES=1 bash scripts/wo_future/run.sh BE long
```

Supported datasets: `NP`, `PJM`, `BE`, `FR`, `DE`, `Energy`, `Sdwpfm1`,
`Sdwpfm2`, `Sdwpfh1`, `Sdwpfh2`, `Colbun`, `Rapel`. Each supports `short` and `long`.

Settings are in `scripts/{w_future,wo_future}/presets.json`. Override them with:

```bash
python run.py --mode wo_future --dataset BE --length short --seeds 1 2 3 \
  --evaluate-test --set lr=0.001 batch_size=64
```

For a quick CPU pipeline check, run `python run.py --device cpu --seeds 1 --smoke`.
Test data is evaluated only with `--evaluate-test` (enabled by the shell scripts).
Each run saves a checkpoint, configuration, history and MSE/MAE/CRPS/QICE;
`aggregate.json` reports seed means and standard deviations. Use a new
`--output-dir` when rerunning. MSE/MAE use the actual sample mean, without recentering.

`w_future` supplies the dataset's true future covariates; `wo_future` never reads
them as model inputs. The scripts train standalone models. Archived multi-stage
search results are not claimed as from-scratch reproductions by these scripts.

## Code

`models/` builds LAD; `layers/` contains the backbone, lag alignment and state
response; `data_provider/`, `exp/` and `utils/` handle data, training and metrics.
Data, weights, baselines and historical tuning logs are not bundled.

## License

MIT; see [LICENSE](LICENSE). The data-reader code is adapted from
[DAG](https://github.com/decisionintelligence/DAG); its copyright notice is retained.
