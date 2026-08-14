# SimToolReal Local Environment

This workspace contains a local SimToolReal checkout and an isolated Python 3.11 environment.

## Paths

```text
Workspace: /data/SUN_ht/Isaac_Gym/SimToolReal_workspace
Repo:      /data/SUN_ht/Isaac_Gym/SimToolReal_workspace/simtoolreal
Env:       /data/SUN_ht/Isaac_Gym/SimToolReal_workspace/env_simtoolreal
IsaacLab:  /data/SUN_ht/Isaac_Gym/IsaacLab_v2.2.1
```

## Activate

```bash
cd /data/SUN_ht/Isaac_Gym/SimToolReal_workspace/simtoolreal
conda activate /data/SUN_ht/Isaac_Gym/SimToolReal_workspace/env_simtoolreal
export OMNI_KIT_ACCEPT_EULA=YES
export MPLCONFIGDIR=/data/SUN_ht/Isaac_Gym/SimToolReal_workspace/mpl_cache
export PIP_CACHE_DIR=/data/SUN_ht/Isaac_Gym/SimToolReal_workspace/pip_cache
```

## Verified

```bash
python isaacsimenvs/tests/test_load_isaacsim.py
python isaacsimenvs/tests/test_gym_register.py
```

Both commands launch Isaac Sim through Isaac Lab and complete in this environment.

## GPU Note

The current execution session cannot access the NVIDIA driver:

```text
nvidia-smi: NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.
torch.cuda.is_available(): False
```

Because of that, full environment stepping fails at simulation creation with:

```text
RuntimeError: No CUDA GPUs are available
```

Run the training or full smoke test from a shell/session where `nvidia-smi` works.

## Full Smoke Test

```bash
python isaacsimenvs/tests/test_simtoolreal_env_smoke.py \
  --num_envs 8 \
  --num_assets_per_type 2 \
  --steps 10
```

## Training Entry

```bash
python isaacsimenvs/train.py \
  --task Isaacsimenvs-SimToolReal-Direct-v0 \
  --agent rl_games_cfg_entry_point \
  --headless
```

