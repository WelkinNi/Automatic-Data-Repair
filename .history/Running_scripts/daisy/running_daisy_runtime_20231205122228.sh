#!/bin/bash
TASK_LIST=("tax")
CNTS_LIST=(1)
NUMS_LIST=()
for ((i=2; i<=20; i+=2)); 
do
  NUMS_LIST+=("$(printf "%04d" $i)k") 
done

for CNT in "${CNTS_LIST[@]}"
do
    PYTHON="/data/nw/DC_ED/References_inner_and_outer/Daisy/daisy.py"
    # Loop through error rates and run holoclean_run.py on dataset
    for NUM in "${NUMS_LIST[@]}"
    do
        echo ${NUM}
        for TASK in "${TASK_LIST[@]}"
        do 
            DIRTY_DATA="./data with dc_rules/tax/split_data/tax-dirty-original_error-${NUM}.csv"
            CLEAN_DATA="./data with dc_rules/tax/split_data/tax-clean-clean_data_ori-${NUM}.csv"
            RULE="./data with dc_rules/${TASK}/dc_rules_holoclean.txt"
                    
            TASK_NAME="${TASK}${CNT}"
            DIRTY_DATA_PATH=${DIRTY_DATA}
            COMMAND="timeout 1d /home/dell/anaconda3/envs/torch110/bin/python $PYTHON --task_name \"$TASK_NAME\" --rule_path \"$RULE\" --onlyed 0 --perfected 0 --dirty_path \"$DIRTY_DATA_PATH\" --clean_path \"$CLEAN_DATA\" "
            if ! eval "$COMMAND"; then
                if [ $? -eq 124 ]; then
                    echo "Timeout reached: $(date) ${COMMAND}" | tee -a "./aggre_results/timeout_log.txt"
                fi
            fi &
        done
    done
    wait
done


