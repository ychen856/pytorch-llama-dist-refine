  #!/bin/sh
#python3.8 client_jetson_adp.py --ppl 10 --mode default --config config_jetson2.yaml --log test.log
#sleep 10

#python3.8 client_jetson_adp2.py --ppl 10 --mode bandwidth-aware --config config_jetson2.yaml --log 13_jetson2_wlan_ppl_10_server_sudden_1.log
#sleep 10
#python3.8 client_jetson_adp.py --ppl 20 --mode bandwidth-aware --config config_jetson2.yaml --log 13_jetson2_wlan_ppl_10_SM_fixed_end_4.log
#sleep 10
#python3.8 client_jetson_adp.py --ppl 30 --mode bandwidth-aware --config config_jetson2.yaml --log 13_jetson2_wlan_ppl_10_SM_fixed_end_4.log
sleep 10
python3.8 client_jetson_adp2.py --ppl 10 --mode exit-rate --config config_jetson2.yaml --log 10_3_device_3_hop_test_63_1.log

sleep 10
python3.8 client_jetson_adp2.py --ppl 10 --mode exit-rate --config config_jetson2.yaml --log 10_3_device_3_hop_test_63_2.log

sleep 10
python3.8 client_jetson_adp2.py --ppl 10 --mode exit-rate --config config_jetson2.yaml --log 10_3_device_3_hop_test_63_3.log

sleep 10
python3.8 client_jetson_adp2.py --ppl 10 --mode exit-rate --config config_jetson2.yaml --log 10_3_device_3_hop_test_63_4.log

sleep 10
python3.8 client_jetson_adp2.py --ppl 10 --mode exit-rate --config config_jetson2.yaml --log 10_3_device_3_hop_test_63_5.log