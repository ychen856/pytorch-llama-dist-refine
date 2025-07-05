#!/bin/sh
python3.8 client_jetson_adp.py --ppl 10 --config config_jetson2.yaml --log test.log
sleep 10
python3.8 client_jetson_adp.py --ppl 10 --config config_jetson2.yaml --log jetson2_wlan_ppl_10_fixed_5.log
sleep 10
python3.8 client_jetson_adp.py --ppl 10 --config config_jetson2.yaml --log jetson2_wlan_ppl_10_fixed_6.log
sleep 10
python3.8 client_jetson_adp.py --ppl 10 --config config_jetson2.yaml --log jetson2_wlan_ppl_10_fixed_7.log
sleep 10
python3.8 client_jetson_adp.py --ppl 10 --config config_jetson2.yaml --log jetson2_wlan_ppl_10_fixed_8.log