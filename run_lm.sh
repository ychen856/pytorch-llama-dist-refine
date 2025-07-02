#!/bin/sh

python3 lm_inference.py --head 1 --ppl 20 --config config_nrp.yaml 2>&1|tee lm_inference_head_1_ppl_20_1.log
python3 lm_inference.py --head 1 --ppl 20 --config config_nrp.yaml 2>&1|tee lm_inference_head_1_ppl_20_2.log
python3 lm_inference.py --head 1 --ppl 20 --config config_nrp.yaml 2>&1|tee lm_inference_head_1_ppl_20_3.log
python3 lm_inference.py --head 1 --ppl 20 --config config_nrp.yaml 2>&1|tee lm_inference_head_1_ppl_20_4.log
python3 lm_inference.py --head 1 --ppl 20 --config config_nrp.yaml 2>&1|tee lm_inference_head_1_ppl_20_5.log

python3 lm_inference.py --head 2 --ppl 20 --config config_nrp.yaml 2>&1|tee lm_inference_head_2_ppl_20_1.log
python3 lm_inference.py --head 2 --ppl 20 --config config_nrp.yaml 2>&1|tee lm_inference_head_2_ppl_20_2.log
python3 lm_inference.py --head 2 --ppl 20 --config config_nrp.yaml 2>&1|tee lm_inference_head_2_ppl_20_3.log
python3 lm_inference.py --head 2 --ppl 20 --config config_nrp.yaml 2>&1|tee lm_inference_head_2_ppl_20_4.log
python3 lm_inference.py --head 2 --ppl 20 --config config_nrp.yaml 2>&1|tee lm_inference_head_2_ppl_20_5.log

python3 lm_inference.py --head 4 --ppl 20 --config config_nrp.yaml 2>&1|tee lm_inference_head_4_ppl_20_1.log
python3 lm_inference.py --head 4 --ppl 20 --config config_nrp.yaml 2>&1|tee lm_inference_head_4_ppl_20_2.log
python3 lm_inference.py --head 4 --ppl 20 --config config_nrp.yaml 2>&1|tee lm_inference_head_4_ppl_20_3.log
python3 lm_inference.py --head 4 --ppl 20 --config config_nrp.yaml 2>&1|tee lm_inference_head_4_ppl_20_4.log
python3 lm_inference.py --head 4 --ppl 20 --config config_nrp.yaml 2>&1|tee lm_inference_head_4_ppl_20_5.log

python3 lm_inference.py --head 6 --ppl 20 --config config_nrp.yaml 2>&1|tee lm_inference_head_6_ppl_20_1.log
python3 lm_inference.py --head 6 --ppl 20 --config config_nrp.yaml 2>&1|tee lm_inference_head_6_ppl_20_2.log
python3 lm_inference.py --head 6 --ppl 20 --config config_nrp.yaml 2>&1|tee lm_inference_head_6_ppl_20_3.log
python3 lm_inference.py --head 6 --ppl 20 --config config_nrp.yaml 2>&1|tee lm_inference_head_6_ppl_20_4.log
python3 lm_inference.py --head 6 --ppl 20 --config config_nrp.yaml 2>&1|tee lm_inference_head_6_ppl_20_5.log
