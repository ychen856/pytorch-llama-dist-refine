#!/bin/sh

python3.8 -u client_jetson.py --config config_jetson2.yaml 2>&1 | tee jetson_2_device_ppl_10_weight_6_1.log
sleep 10
python3.8 -u client_jetson.py --config config_jetson2.yaml 2>&1 | tee jetson_2_device_ppl_10_weight_6_2.log
sleep 10
python3.8 -u client_jetson.py --config config_jetson2.yaml 2>&1 | tee jetson_2_device_ppl_10_weight_6_3.log
sleep 10
python3.8 -u client_jetson.py --config config_jetson2.yaml 2>&1 | tee jetson_2_device_ppl_10_weight_6_4.log
sleep 10
python3.8 -u client_jetson.py --config config_jetson2.yaml 2>&1 | tee jetson_2_device_ppl_10_weight_6_5.log
sleep 10
python3.8 -u client_jetson.py --config config_jetson2.yaml 2>&1 | tee jetson_2_device_ppl_10_weight_6_6.log
sleep 10
python3.8 -u client_jetson.py --config config_jetson2.yaml 2>&1 | tee jetson_2_device_ppl_10_weight_6_7.log
sleep 10
python3.8 -u client_jetson.py --config config_jetson2.yaml 2>&1 | tee jetson_2_device_ppl_10_weight_6_8.log
sleep 10
python3.8 -u client_jetson.py --config config_jetson2.yaml 2>&1 | tee jetson_2_device_ppl_10_weight_6_9.log
sleep 10
python3.8 -u client_jetson.py --config config_jetson2.yaml 2>&1 | tee jetson_2_device_ppl_10_weight_6_10.log
sleep 10

