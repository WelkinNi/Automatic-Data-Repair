#!/bin/bash

# Define variables
TASK_LIST=("hospital" "beers" "flights" "rayyan")
NUMS_LIST=(10 30 50 70 90)
CNTS_LIST=(1 2 3)

# Create an array to store the process IDs of the background commands
pids=()

for CNT in "${CNTS_LIST[@]}"
do
    # Loop through error rates and run holoclean_run.py on dataset
    for TASK in "${TASK_LIST[@]}"
    do        
        for NUM in "${NUMS_LIST[@]}"
        do                
            DIRTY_DATA="./data with dc_rules/${TASK}/noise/${TASK}-inner_error-"
            CLEAN_DATA="./data with dc_rules/${TASK}/clean.csv"
            RULE="./data with dc_rules/${TASK}/dc_rules-validate-fd-horizon.txt"
            
            PYTHON="/data/nw/DC_ED/References_inner_and_outer/SCAREd/scared.py"
            TASK_NAME="${TASK}${CNT}"
            DIRTY_DATA_PATH="${DIRTY_DATA}${NUM}.csv"
            /home/dell/anaconda3/envs/torch110/bin/python $PYTHON --task_name "$TASK_NAME" --rule_path "$RULE" --onlyed 0 --perfected 0 --dirty_path "$DIRTY_DATA_PATH" --clean_path "$CLEAN_DATA" &
            
            DIRTY_DATA="./data with dc_rules/${TASK}/noise/${TASK}-outer_error-"
            DIRTY_DATA_PATH="${DIRTY_DATA}${NUM}.csv"
            /home/dell/anaconda3/envs/torch110/bin/python $PYTHON --task_name "$TASK_NAME" --rule_path "$RULE" --onlyed 0 --perfected 0 --dirty_path "$DIRTY_DATA_PATH" --clean_path "$CLEAN_DATA" &
        done
        wait
    done
done


# for pid in "${pids[@]}"
# do
#     wait $pid
#     exit_code=$?
#     if [ $exit_code -eq 124 ]; then
#         echo "COMMAND time out: ${COMMAND}" >> "./aggre_results/timeout_log.txt"
#     fi
# done
# pids=()