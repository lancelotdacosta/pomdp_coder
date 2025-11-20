#!/bin/bash

trap "echo '🛑 CTRL+C pressed → stopping all experiments…'; jobs -p | xargs kill; exit 1" INT 
## Normally process run in background when we do "Ctrl+C"; this will help terminate the process

echo "──────────────────────────────────────────────"
echo "    Launching Multi-Seed Experiment Runner    "
echo "──────────────────────────────────────────────"

# Delay in seconds between launches
DELAY=10

## Same as range; will take 0, 1, 2, 3
for SEED in {1..4}
do
    echo -e "▶ Starting run for seed $SEED}"

    LOG_DIR="outputs/four_rooms/ours/\${now:%Y-%m-%d}/\${now:%H-%M-%S}_seed${SEED}"

    # Print the directory it's going to use
    echo -e "➤ Output directory: ${LOG_DIR}"

    python main.py \
        --config-dir=uncertain_worms/config/approaches/ours \
        --config-name=four_rooms_llm_TROI_po_planning_agent.yaml \
        seed=$SEED \
        agent.use_openrouter=true \
        save_log=true \
        "hydra.run.dir=${LOG_DIR}" &

    echo -e "✔ Started seed $SEED in background"

    # Delay before starting next job
    echo -e "Waiting ${DELAY}s before next seed..."
    sleep $DELAY
done

echo "──────────────────────────────────────────────"


echo -e " All runs launched! Waiting for all processes to finish..."
wait
echo -e " All experiments finished successfully!"
