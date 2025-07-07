#!/bin/sh
python3.8 client_jetson_adp.py --ppl 20 --mode default --config config_jetson2.yaml --log test.log
sleep 10

python3.8 client_jetson_adp.py --ppl 20 --mode default --config config_jetson2.yaml --log 07_jetson2_wlan_ppl_20_fixed_1.log
sleep 10
python3.8 client_jetson_adp.py --ppl 20 --mode fixed --config config_jetson2.yaml --log 07_jetson2_wlan_ppl_20_fixed_1.log
sleep 10
python3.8 client_jetson_adp.py --ppl 20 --mode linear-exit-rate --config config_jetson2.yaml --log 07_jetson2_wlan_ppl_20_linear_1.log
sleep 10
python3.8 client_jetson_adp.py --ppl 20 --mode exit-rate --config config_jetson2.yaml --log 07_jetson2_wlan_ppl_20_exit_1.log
sleep 10
python3.8 client_jetson_adp.py --ppl 20 --mode bandwidth-aware --config config_jetson2.yaml --log 07_jetson2_wlan_ppl_20_bandwidth_1.log
sleep 10

python3.8 client_jetson_adp.py --ppl 30 --mode default --config config_jetson2.yaml --log 07_jetson2_wlan_ppl_30_fixed_1.log
sleep 10

python3.8 client_jetson_adp.py --ppl 20 --mode default --config config_jetson2.yaml --log 07_jetson2_wlan_ppl_20_fixed_2.log
sleep 10
python3.8 client_jetson_adp.py --ppl 20 --mode fixed --config config_jetson2.yaml --log 07_jetson2_wlan_ppl_20_fixed_2.log
sleep 10
python3.8 client_jetson_adp.py --ppl 20 --mode linear-exit-rate --config config_jetson2.yaml --log 07_jetson2_wlan_ppl_20_linear_2.log
sleep 10
python3.8 client_jetson_adp.py --ppl 20 --mode exit-rate --config config_jetson2.yaml --log 07_jetson2_wlan_ppl_20_exit_2.log
sleep 10
python3.8 client_jetson_adp.py --ppl 20 --mode bandwidth-aware --config config_jetson2.yaml --log 07_jetson2_wlan_ppl_20_bandwidth_2.log
sleep 10
