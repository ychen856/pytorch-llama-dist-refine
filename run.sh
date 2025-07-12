  #!/bin/sh
python3.8 client_jetson_adp.py --ppl 10 --mode default --config config_jetson2.yaml --log test.log
sleep 10

python3.8 client_jetson_adp.py --ppl 10 --mode exit-rate --config config_jetson2.yaml --log 11_jetson2_wlan_ppl_10_mid_exit_TTT_2.log
sleep 10
python3.8 client_jetson_adp.py --ppl 20 --mode exit-rate --config config_jetson2.yaml --log 11_jetson2_wlan_ppl_10_mid_exit_TTT_2.log
sleep 10
python3.8 client_jetson_adp.py --ppl 30 --mode exit-rate --config config_jetson2.yaml --log 11_jetson2_wlan_ppl_10_mid_exit_TTT_2.log
sleep 10
