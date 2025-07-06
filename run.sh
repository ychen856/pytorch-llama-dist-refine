#!/bin/sh
python3.8 client_jetson.py --ppl 10 --weight 0.0 --config config_jetson2.yaml --log test.log
sleep 10
python3.8 client_jetson.py --ppl 10 --weight 0.0 --config config_jetson2.yaml --log 06_jetson2_wlan_ppl_10_default_2.log
sleep 10
python3.8 client_jetson_adp.py --ppl 10 --mode fixed --config config_jetson2.yaml --log 06_jetson2_wlan_ppl_10_fixed_2.log
sleep 10
python3.8 client_jetson_adp.py --ppl 10 --mode linear-exit-rate --config config_jetson2.yaml --log 06_jetson2_wlan_ppl_10_linear_2.log
sleep 10
python3.8 client_jetson_adp.py --ppl 10 --mode exit-rate --config config_jetson2.yaml --log 06_jetson2_wlan_ppl_10_exit_2.log
sleep 10
python3.8 client_jetson_adp.py --ppl 10 --mode bandwidth-aware --config config_jetson2.yaml --log 06_jetson2_wlan_ppl_10_bandwidth_2.log
sleep 10

python3.8 client_jetson.py --ppl 30 --weight 0.0 --config config_jetson2.yaml --log 06_jetson2_wlan_ppl_30_default_2.log
sleep 10
python3.8 client_jetson_adp.py --ppl 30 --mode fixed --config config_jetson2.yaml --log 06_jetson2_wlan_ppl_30_fixed_2.log
sleep 10
python3.8 client_jetson_adp.py --ppl 30 --mode linear-exit-rate --config config_jetson2.yaml --log 06_jetson2_wlan_ppl_30_linear_2.log
sleep 10
python3.8 client_jetson_adp.py --ppl 30 --mode exit-rate --config config_jetson2.yaml --log 06_jetson2_wlan_ppl_30_exit_2.log
sleep 10
python3.8 client_jetson_adp.py --ppl 30 --mode bandwidth-aware --config config_jetson2.yaml --log 06_jetson2_wlan_ppl_30_bandwidth_2.log
sleep 10
