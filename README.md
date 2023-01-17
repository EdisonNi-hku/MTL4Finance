# when-does-mtl-work
We conduct a case study on Financial NLP, exploring when will MTL work from a perspective of the relations between tasks and skills to be aggregated.

### Configuring the environment
```shell
conda create -n mtl python==3.7 --yes
conda activate mtl
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install -r code/requirements.txt
```

### Data Preparation
```shell
# First prepare data of TSA, SC, FSRL and CD.
cd data
unzip dataset.zip
# Datasets of NC and NAD requires sending an request to their authors. See:
# NC: https://sites.google.com/nlg.csie.ntu.edu.tw/finnum3/data
# NAD: https://sites.google.com/nlg.csie.ntu.edu.tw/finnum2020/data
# NAD has its official train/dev/test sets. NC provides official train/test sets.
# To replicate our validation set of
# NC: split 20% for dev from the official training data, using sklearn.train_test_split with a random seed of 42
```

### Pre-trained Model Preparation
```shell
# Download P-FinBERT, save it at when-does-mtl-work/prosusfinbert
wget https://prosus-public.s3-eu-west-1.amazonaws.com/finbert/language-model/pytorch_model.bin
# Download other models
python code/model_cache.py
```

### STL experiments
```shell
# Run SC with P-FinBERT (this command will train and test 5 random seeds at once).
bash code/run_exp.sh 0,1,2,3 prosus fpb 40
```
- Field 1. ID(s) of GPU(s) to use, here we use four GPUs.
- Field 2. Pre-trained model used. Choices: {P-FinBERT: prosus, Y-FinBERT: hkust, FinancialBERT: financial, BERT-cased: bert, BERT-uncased: bert_uncased}
- Field 3. Task to run. Choices: {TSA: asa, SC: fpb, FSRL: srl, NC: nc, NAD: na, CD: cau}
- Field 4. Number of epochs to run.
- The command will save checkpoints in directories like prosus_fpb_{random_seed}, and test results in prosus_fpb_{random_seed}.eval

### MTL experiments
```shell
# MTL with naive P-FinBERT
python code/run_mtl.py --save all_1 --epoch 40 --seed 1 --task_list asa,fpb,na,nc,srl,cau
# MTL with SPAL-FinBERT
python code/run_mtl.py --save all_p1 --freeze --epoch 40 --seed 1 --task_list asa,fpb,na,nc,srl,cau --tunning_size 204
```
- save: the directory to save the checkpoints.
- epoch: number of epochs to train.
- seed: random seed to use.
- task_list: task to be aggregated in MTL. Choices: {TSA: asa, SC: fpb, FSRL: srl, NC: nc, NAD: na, CD: cau}
- freeze: only used for SPAL-FinBERT. Set the flag to freeze the pre-trained part.
- tunning_size: SPAL hidden size, default to be 204

```shell
# Test MTL with naive P-FinBERT
bash code/test_mtl.sh 0 all_1 asa,fpb,nc,na,srl,cau prosus
```
- Field 1. ID(s) of GPU(s) to use, here we use one GPU.
- Field 2. MTL model to test.
- Field 3. Task to test. Choices: {TSA: asa, SC: fpb, FSRL: srl, NC: nc, NAD: na, CD: cau}
- Field 4. The pre-trained model that the MTL system based on.

```shell
# Test MTL with SPAL-FinBERT
bash code/test_mtl.sh 0 all_p1 asa,fpb,nc,na,srl,cau prosus 1 204
```
- Field 5. whether to use SPAL-FinBERT. {1: True, 0: False}
- Field 6. SPAL hidden size, default to be 204.