#!/bin/sh


python3 lm_inference.py --head 6 --ppl 20 --config config_nrp.yaml 2>&1|tee lm_train_head_6_ppl_20_2.log
python3 lm_inference.py --head 6 --ppl 20 --config config_nrp.yaml 2>&1|tee lm_train_head_6_ppl_20_3.log
python3 lm_inference.py --head 6 --ppl 20 --config config_nrp.yaml 2>&1|tee lm_train_head_6_ppl_20_4.log
python3 lm_inference.py --head 6 --ppl 20 --config config_nrp.yaml 2>&1|tee lm_train_head_6_ppl_20_5.log

python3 lm_inference.py --head 6 --ppl 30 --config config_nrp.yaml 2>&1|tee lm_train_head_6_ppl_30_1.log
python3 lm_inference.py --head 6 --ppl 30 --config config_nrp.yaml 2>&1|tee lm_train_head_6_ppl_30_2.log
python3 lm_inference.py --head 6 --ppl 30 --config config_nrp.yaml 2>&1|tee lm_train_head_6_ppl_30_3.log
python3 lm_inference.py --head 6 --ppl 30 --config config_nrp.yaml 2>&1|tee lm_train_head_6_ppl_30_4.log
python3 lm_inference.py --head 6 --ppl 30 --config config_nrp.yaml 2>&1|tee lm_train_head_6_ppl_30_5.log