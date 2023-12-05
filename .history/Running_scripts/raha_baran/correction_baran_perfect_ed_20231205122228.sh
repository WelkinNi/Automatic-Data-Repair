#!/bin/bash

TASK_LIST=("hospital" "beers" "rayyan" "flights")
NUMS_LIST=('01')
CNTS_LIST=(1 2 3)

for NUM in "${NUMS_LIST[@]}"
do
    # Loop through error rates and run holoclean_run.py on dataset
    for CNT in "${CNTS_LIST[@]}"
    do        
        for TASK in "${TASK_LIST[@]}"
        do                
            DIRTY_DATA="./data with dc_rules/${TASK}/noise/${TASK}-inner_outer_error-"
            CLEAN_DATA="./data with dc_rules/${TASK}/clean.csv"
            PYTHON="/data/nw/DC_ED/References_inner_and_outer/raha-master/correction.py"
            TASK_NAME="${TASK}${CNT}"
            DIRTY_DATA_PATH="${DIRTY_DATA}${NUM}.csv"
            
            COMMAND="timeout 1d /home/dell/anaconda3/envs/torch110/bin/python $PYTHON --task_name \"$TASK_NAME\" --dirty_path \"$DIRTY_DATA_PATH\" --clean_path \"$CLEAN_DATA\" \&"
            { eval "$COMMAND" || echo "Timeout reached: $(date) ${COMMAND}" ; } 2>&1 | tee -a "./aggre_results/timeout_log.txt" 
        done
        # wait
    done
done

bash ./Running_scripts_inner_and_outer/raha_baran/running_correction_baran_raha.sh


