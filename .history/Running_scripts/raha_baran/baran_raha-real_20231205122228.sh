#!/bin/bash

# TASK_LIST=("hospital" "beers" "rayyan" "flights")
TASK_LIST=("rayyan" "beers")
# TASK_LIST=("beers")
# NUMS_LIST=(10 30 50 70 90)
NUMS_LIST=('01')
CNTS_LIST=(1 2 3)

for CNT in "${CNTS_LIST[@]}"
do
    # Loop through error rates and run holoclean_run.py on dataset
    for NUM in "${NUMS_LIST[@]}"
    do        
        for TASK in "${TASK_LIST[@]}"
        do                
            DIRTY_DATA="./data with dc_rules/${TASK}/noise/${TASK}-inner_outer_error-"
            CLEAN_DATA="./data with dc_rules/${TASK}/clean.csv"
            PYTHON="/data/nw/DC_ED/References_inner_and_outer/raha-master/correction_with_raha.py"
            TASK_NAME="${TASK}${CNT}"
            DIRTY_DATA_PATH="${DIRTY_DATA}${NUM}.csv"
            /home/dell/anaconda3/envs/torch110/bin/python $PYTHON --task_name "$TASK_NAME" --dirty_path "$DIRTY_DATA_PATH" --clean_path "$CLEAN_DATA" &
            sleep 600
        done
    done
    wait
done

