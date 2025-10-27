# pomdp_coder

This repo contains the code release for the paper
[LLM-Guided Probabilistic Program Induction for POMDP Model Estimation](https://arxiv.org/pdf/2505.02216)

## Setup

```
conda create -n "uncertain_worms" python=3.11
python -m pip install -e .
```

## Minigrid

### Hardcoded

```
python main.py --config-dir=uncertain_worms/config/approaches/hardcoded --config-name=hardcoded_po_planning_agent.yaml
```

### Ours

```
python main.py --config-dir=uncertain_worms/config/approaches/ours --config-name=llm_TROI_po_planning_agent.yaml
```

## Real spot experiments configs

### Ours (model learning)

```
python main.py --config-dir=uncertain_worms/config/approaches/ours --config-name=spot_llm_TROI_po_planning_agent.yaml
```

### Fixed (Ours after model learning)

```
python main.py --config-dir=uncertain_worms/config/approaches/hardcoded --config-name=spot_hardcoded_po_planning_fixed_agent.yaml
```

### Uniform

```
python main.py --config-dir=uncertain_worms/config/approaches/hardcoded --config-name=spot_hardcoded_po_planning_agent.yaml
```

### Direct LLM

```
python main.py --config-dir=uncertain_worms/config/approaches/direct --config-name=spot_direct_llm_po_agent.yaml
```

### Behavior Cloning

```
python main.py --config-dir=uncertain_worms/config/approaches/behavior_cloning --config-name=spot_bc_po_agent.yaml
```

### Tabular

```
python main.py --config-dir=uncertain_worms/config/approaches/tabular --config-name=spot_tabular_TROI_po_planning_agent.yaml
```

## Toy problem

### Hardcoded

```
python main.py --config-dir=uncertain_worms/config/approaches/hardcoded --config-name=tiger_hardcoded.yaml
```

```
python main.py --config-dir=uncertain_worms/config/approaches/hardcoded --config-name=rocksample_hardcoded.yaml
```

### Ours

```
python main.py --config-dir=uncertain_worms/config/approaches/ours --config-name=tiger_llm_TROI_po_planning_agent.yaml
```

```
python main.py --config-dir=uncertain_worms/config/approaches/ours --config-name=rocksample_llm_T_po_planning_agent.yaml
```

## Segmentation Server

```
python uncertain_worms/environments/spot/segmentation_server.py
```

## Testing

Make sure these pass before merging in

```
python -m pytest tests/test_configs.py
```

## Tensorboard

```
tensorboard --logdir=outputs
```
