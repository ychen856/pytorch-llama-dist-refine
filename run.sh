  #!/bin/sh
python3.8 client_jetson_adp.py --ppl 10 --mode default --config config_jetson2.yaml --log test.log
sleep 10

python3.8 client_jetson_adp.py --ppl 10 --mode bandwidth-aware --config config_jetson2.yaml --log 13_jetson2_wlan_ppl_10_SM_bandwidth_1.log
sleep 10
python3.8 client_jetson_adp.py --ppl 20 --mode bandwidth-aware --config config_jetson2.yaml --log 13_jetson2_wlan_ppl_10_SM_bandwidth_1.log
sleep 10
python3.8 client_jetson_adp.py --ppl 30 --mode bandwidth-aware --config config_jetson2.yaml --log 13_jetson2_wlan_ppl_10_SM_bandwidth_1.log
sleep 10

python3.8 client_jetson_adp.py --ppl 10 --mode exit-rate --config config_jetson2.yaml --log 13_jetson2_wlan_ppl_10_SM_exit_1.log
sleep 10
python3.8 client_jetson_adp.py --ppl 20 --mode exit-rate --config config_jetson2.yaml --log 13_jetson2_wlan_ppl_10_SM_exit_1.log
sleep 10
python3.8 client_jetson_adp.py --ppl 30 --mode exit-rate --config config_jetson2.yaml --log 13_jetson2_wlan_ppl_10_SM_exit_1.log
sleep 10

python3.8 client_jetson_adp.py --ppl 10 --mode default --config config_jetson2.yaml --log 13_jetson2_wlan_ppl_10_SM_default_1.log
sleep 10
python3.8 client_jetson_adp.py --ppl 20 --mode default --config config_jetson2.yaml --log 13_jetson2_wlan_ppl_10_SM_default_1.log
sleep 10
python3.8 client_jetson_adp.py --ppl 30 --mode default --config config_jetson2.yaml --log 13_jetson2_wlan_ppl_10_SM_default_1.log
sleep 10