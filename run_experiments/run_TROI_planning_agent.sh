#!/bin/bash

trap "echo '🛑 CTRL+C pressed → stopping all experiments…'; jobs -p | xargs kill; exit 1" INT 
## Normally process run in background when we do "Ctrl+C"; this will help terminate the process

echo "──────────────────────────────────────────────"
echo "    Launching Multi-Seed Experiment Runner    "
echo "──────────────────────────────────────────────"

# Delay in seconds between launches
DELAY=10
ATTEMPTS=5
ENVIRONMENT="corners"
APPROACH="ours_O"

# Same as range; will take 0, 1, 2, 3
for SEED in {0..0}
do
    echo -e "▶ Starting run for seed $SEED}"

    LOG_DIR="outputs/exp_pho/${ENVIRONMENT}/${APPROACH}/\${now:%Y-%m-%d}/\${now:%H-%M-%S}_seed${SEED}"

    # Print the directory it's going to use
    echo -e "➤ Output directory: ${LOG_DIR}"

    cmd_args=(
        python main.py
        --config-dir="uncertain_worms/config/approaches/$APPROACH"
        --config-name="${ENVIRONMENT}_${APPROACH}.yaml"
        "+extras.environment=$ENVIRONMENT"
        "+extras.approach=$APPROACH"
        "seed=$SEED"
        "save_log=true"
        "hydra.run.dir=${LOG_DIR}"
    )

    # Conditionally append arguments if APPROACH is "ours"
    case "$APPROACH" in
        "ours" | "ours_O" | "ours_I" | "ours_OI" )
            cmd_args+=(
                "agent.use_openrouter=true"
                "agent.num_model_attempts=$ATTEMPTS"
                "agent.num_online_model_attempts=$ATTEMPTS"
            )
            ;;
    esac


    # Execute the command using the array -> "${cmd_args[@]}" expands to the list of arguments exactly as defined
    "${cmd_args[@]}" &


    echo -e "✔ Started seed $SEED in background"

    # Delay before starting next job
    echo -e "Waiting ${DELAY}s before next seed..."
    sleep $DELAY
done

echo "──────────────────────────────────────────────"


echo -e " All runs launched! Waiting for all processes to finish..."
wait
echo -e " All experiments finished successfully!"
