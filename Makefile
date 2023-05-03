
SHELL := /bin/bash

DATE:=$(shell date +"%Y_%m_%d_%T" | tr ':' '_')

$(DATE): 
	mkdir $(DATE)

define make_capture_in_out_target
master_$i_$o: $(DATE)
	sleep 1
	python3 main.py -i $i -o $(DATE)/$o.mp4 -t 300 -g $(DATE)/$o.csv -f 50 -m 3 -c

slave_$i_$o: $(DATE)
	python3 main.py -i $i -o $(DATE)/$o.mp4 -t 300 -g $(DATE)/$o.csv -f 50 -s 3 -c
endef
$(foreach o, 0 1 2 3, $(foreach i,0 1,$(eval $(call make_capture_in_out_target))))

master_slave: master_0_0 slave_1_1

slave_slave: slave_0_2 slave_1_3
