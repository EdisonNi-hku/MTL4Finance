WORKDIR="/cluster/work/sachan/leonhard/jingwei/ni2/MTL4Finance/code"
export PYTHONPATH=$WORKDIR

GPU=${1}
MODEL=${2}
TASK=${3}
EPOCH=${4:-20}
SAVE_STRATEGY=${5:-steps}

if [[ $TASK == 'fpb' ]]; then
  SCRIPT='code/run_sc.py'
elif [[ $TASK == 'srl' ]]; then
  SCRIPT='code/run_fsrl.py'
elif [[ $TASK == 'asa' ]]; then
  SCRIPT='code/run_tsa.py'
elif [[ $TASK == 'nc' ]]; then
  SCRIPT='code/run_nc.py'
elif [[ $TASK == 'na' ]]; then
  SCRIPT='code/run_nad.py'
elif [[ $TASK == 'cau' ]]; then
  SCRIPT='code/run_cd.py'
fi

if [[ $MODEL == 'bert' ]]; then
  MODEL_NAME='bert-base-cased'
elif [[ $MODEL == 'hkust' ]]; then
  MODEL_NAME='yiyanghkust/finbert-pretrain'
elif [[ $MODEL == 'prosus' ]]; then
  MODEL_NAME='prosusfinbert'
elif [[ $MODEL == 'bert_uncased' ]]; then
  MODEL_NAME='bert-base-uncased'
elif [[ $MODEL == 'financial' ]]; then
  MODEL_NAME='ahmedrachid/FinancialBERT'
fi

for i in {1..5}
do
  SAVE_PATH=${MODEL}_${TASK}_${i}
  cmd="CUDA_VISIBLE_DEVICES=${GPU} python ${SCRIPT} --save ${SAVE_PATH} --strategy ${SAVE_STRATEGY} \
  --model ${MODEL_NAME} --seed ${i} --epoch ${EPOCH} --pool --no_sep > ${SAVE_PATH}.log"
  echo "${cmd}"
  eval "${cmd}"
  arr=($(ls ${SAVE_PATH} | tr " " "\n"))
  arr1=($(echo ${arr[0]} | tr "-" "\n"))
  arr2=($(echo ${arr[1]} | tr "-" "\n"))
  if [ ${arr1[1]} -gt ${arr2[1]} ]; then
    EVAL_CHECKPOINT='checkpoint-'${arr2[1]}
  else
    EVAL_CHECKPOINT='checkpoint-'${arr1[1]}
  fi
  CUDA_VISIBLE_DEVICES=${GPU} python ${SCRIPT} --test ${SAVE_PATH}/${EVAL_CHECKPOINT} \
  --model ${MODEL_NAME} --seed ${i} --pool --no_sep > ${SAVE_PATH}.eval
done
