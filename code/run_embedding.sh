GPU=${1}
MODEL=${2}
LOAD_PATH=${3}
TASK_NAME=${4}
OUTPUT_DIR=${5}
BATCH_SIZE=${6:-32}
TASK_TYPE=${7:-CR}
EMBE_TYPE=${8:-task}
GLUE_TASK=${9:-False}

if [[ ${TASK_TYPE} == 'CR' && ${EMBE_TYPE} == 'task' ]]; then
  RUN_FN='code/run_taskemb_CR.py'
elif [[ ${TASK_TYPE} == 'CR' && ${EMBE_TYPE} == 'text' ]]; then
  RUN_FN='code/run_textemb_CR.py'
elif [[ ${TASK_TYPE} == 'SL' && ${EMBE_TYPE} == 'task' ]]; then
  RUN_FN='code/run_taskemb_SL.py'
elif [[ ${TASK_TYPE} == 'SL' && ${EMBE_TYPE} == 'text' ]]; then
  RUN_FN='code/run_textemb_SL.py'
fi

if [[ $MODEL == 'bert' ]]; then
  MODEL_NAME='bert-base-cased'
elif [[ $MODEL == 'hkust' ]]; then
  MODEL_NAME='yiyanghkust/finbert-pretrain'
elif [[ $MODEL == 'prosus' ]]; then
  MODEL_NAME='prosusfinbert'
elif [[ $MODEL == 'bert_uncased' ]]; then
  MODEL_NAME='bert-base-uncased'
fi

SEED_ID=42
DATA_DIR=/path/to/MRPC/data/dir
MODEL_TYPE=bert
USE_LABELS=True # set to False to sample from the model's predictive distribution
CACHE_DIR='cache'
# we start from a fine-tuned task-specific BERT so no need for further fine-tuning
FURTHER_FINETUNE_CLASSIFIER=False
FURTHER_FINETUNE_FEATURE_EXTRACTOR=False

if [[ ${EMBE_TYPE} == 'task' ]]; then
  SPECIAL_CMD="--use_labels ${USE_LABELS} --finetune_classifier ${FURTHER_FINETUNE_CLASSIFIER} \
               --finetune_feature_extractor ${FURTHER_FINETUNE_FEATURE_EXTRACTOR} \
               --batch_size=${BATCH_SIZE} --learning_rate 2e-5 --num_epochs 1"
elif [[ ${EMBE_TYPE} == 'text' ]]; then
  SPECIAL_CMD="--per_gpu_train_batch_size=32 --num_train_epochs 1"
fi

cmd="CUDA_VISIBLE_DEVICES=${GPU} python ${RUN_FN} \
    --model ${MODEL_NAME} \
    --glue_task ${GLUE_TASK} \
    --model_type ${MODEL_TYPE} \
    --model_name_or_path ${LOAD_PATH} \
    --task_name ${TASK_NAME} \
    --do_lower_case \
    --data_dir ${DATA_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --cache_dir ${CACHE_DIR} \
    --seed ${SEED_ID} ${SPECIAL_CMD}
    --overwrite_output_dir"

echo ${cmd}
eval ${cmd}