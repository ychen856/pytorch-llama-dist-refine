#!/bin/sh
python3.8 client_jetson_adp.py --ppl 10 --mode default --config config_jetson2.yaml --log test.log
sleep 10

python3.8 client_jetson_adp.py --ppl 10 --mode default --config config_jetson2.yaml --log 07_jetson2_wlan_ppl_10_fixed_1.log
sleep 10
python3.8 client_jetson_adp.py --ppl 10 --mode fixed --config config_jetson2.yaml --log 07_jetson2_wlan_ppl_10_fixed_1.log
sleep 10
python3.8 client_jetson_adp.py --ppl 10 --mode linear-exit-rate --config config_jetson2.yaml --log 07_jetson2_wlan_ppl_10_linear_1.log
sleep 10
python3.8 client_jetson_adp.py --ppl 10 --mode exit-rate --config config_jetson2.yaml --log 07_jetson2_wlan_ppl_10_exit_1.log
sleep 10
python3.8 client_jetson_adp.py --ppl 10 --mode bandwidth-aware --config config_jetson2.yaml --log 07_jetson2_wlan_ppl_10_bandwidth_1.log
sleep 10


python3.8 client_jetson_adp.py --ppl 10 --mode default --config config_jetson2.yaml --log 07_jetson2_wlan_ppl_10_default_2.log
sleep 10
python3.8 client_jetson_adp.py --ppl 10 --mode fixed --config config_jetson2.yaml --log 07_jetson2_wlan_ppl_10_fixed_2.log
sleep 10
python3.8 client_jetson_adp.py --ppl 10 --mode linear-exit-rate --config config_jetson2.yaml --log 07_jetson2_wlan_ppl_10_linear_2.log
sleep 10
python3.8 client_jetson_adp.py --ppl 10 --mode exit-rate --config config_jetson2.yaml --log 07_jetson2_wlan_ppl_10_exit_2.log
sleep 10
python3.8 client_jetson_adp.py --ppl 10 --mode bandwidth-aware --config config_jetson2.yaml --log 07_jetson2_wlan_ppl_10_bandwidth_2.log
sleep 10
