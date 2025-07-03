#!/bin/sh

sleep 10
python3.8 -u client_jetson.py --config config_jetson2.yaml --ppl 30 --weight 0.0 --log jetson_2_device_wlan2_ppl_30_weight_0_1.log
sleep 10
python3.8 -u client_jetson.py --config config_jetson2.yaml --ppl 30 --weight 0.2 --log jetson_2_device_wlan2_ppl_30_weight_2_1.log
sleep 10
python3.8 -u client_jetson.py --config config_jetson2.yaml --ppl 30 --weight 0.4 --log jetson_2_device_wlan2_ppl_30_weight_4_1.log
sleep 10
python3.8 -u client_jetson.py --config config_jetson2.yaml --ppl 30 --weight 0.6 --log jetson_2_device_wlan2_ppl_30_weight_6_1.log


#sleep 10
#python3.8 -u client_jetson.py --config config_jetson2.yaml --ppl 10 --weight 0.0 --log jetson_2_device_wlan2_ppl_10_weight_0_1.log
#sleep 10
#python3.8 -u client_jetson.py --config config_jetson2.yaml --ppl 10 --weight 0.2 --log jetson_2_device_wlan2_ppl_10_weight_2_1.log
#sleep 10
#python3.8 -u client_jetson.py --config config_jetson2.yaml --ppl 10 --weight 0.4 --log jetson_2_device_wlan2_ppl_10_weight_4_1.log
#sleep 10
#python3.8 -u client_jetson.py --config config_jetson2.yaml --ppl 10 --weight 0.6 --log jetson_2_device_wlan2_ppl_10_weight_6_1.log

