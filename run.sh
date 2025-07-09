  #!/bin/sh
python3.8 client_jetson_adp.py --ppl 10 --mode default --config config_jetson2.yaml --log test.log
sleep 10

python3.8 client_jetson_adp.py --ppl 10 --mode exit-rate --config config_jetson2.yaml --log 08_jetson2_wlan_ppl_10_mid_exit_TTT_1.log
sleep 10
python3.8 client_jetson_adp.py --ppl 10 --mode bandwidth-aware --config config_jetson2.yaml --log 08_jetson2_wlan_ppl_10_mid_bandwidth_TTT_1.log
sleep 10


python3.8 client_jetson_adp.py --ppl 10 --mode exit-rate --config config_jetson2.yaml --log 08_jetson2_wlan_ppl_10_mid_exit_TTT_2.log
sleep 10
python3.8 client_jetson_adp.py --ppl 10 --mode bandwidth-aware --config config_jetson2.yaml --log 08_jetson2_wlan_ppl_10_mid_bandwidth_TTT_2.log
sleep 10



python3.8 client_jetson_adp.py --ppl 20 --mode exit-rate --config config_jetson2.yaml --log 08_jetson2_wlan_ppl_20_mid_exit_TTT_2.log
sleep 10
python3.8 client_jetson_adp.py --ppl 20 --mode bandwidth-aware --config config_jetson2.yaml --log 08_jetson2_wlan_ppl_20_mid_bandwidth_TTT_2.log
sleep 10



python3.8 client_jetson_adp.py --ppl 20 --mode exit-rate --config config_jetson2.yaml --log 08_jetson2_wlan_ppl_20_mid_exit_TTT_2.log
sleep 10
python3.8 client_jetson_adp.py --ppl 20 --mode bandwidth-aware --config config_jetson2.yaml --log 08_jetson2_wlan_ppl_20_mid_bandwidth_TTT_2.log
sleep 10



python3.8 client_jetson_adp.py --ppl 30 --mode exit-rate --config config_jetson2.yaml --log 08_jetson2_wlan_ppl_30_mid_exit_TTT_2.log
sleep 10
python3.8 client_jetson_adp.py --ppl 30 --mode bandwidth-aware --config config_jetson2.yaml --log 08_jetson2_wlan_ppl_30_mid_bandwidth_TTT_2.log
sleep 10



python3.8 client_jetson_adp.py --ppl 30 --mode exit-rate --config config_jetson2.yaml --log 08_jetson2_wlan_ppl_30_mid_exit_TTT_2.log
sleep 10
python3.8 client_jetson_adp.py --ppl 30 --mode bandwidth-aware --config config_jetson2.yaml --log 08_jetson2_wlan_ppl_30_mid_bandwidth_TTT_2.log
sleep 10
