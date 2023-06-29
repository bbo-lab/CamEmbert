
SHELL := /bin/bash
TIME := 6000
TRIGGER:=4
EXPOSURE_COLOR:=1000
EXPOSURE_GRAY:=1000
OUTPUT:=~/Basler

DATE:=$(shell date +"%Y_%m_%d_%T" | tr ':' '_')

$(OUTPUT)/$(DATE):
	mkdir -p $(OUTPUT)/$(DATE)

define make_capture_in_out_target
color_master_$i_$o: $(OUTPUT)/$(DATE)
	sleep 1
	python3 main.py -i $i -o $(OUTPUT)/$(DATE)/$o.mp4 -t $(TIME) -g $(OUTPUT)/$(DATE)/$o.csv -f 90 -m $(TRIGGER) -c --exposure $(EXPOSURE_COLOR) -p

color_slave_$i_$o: $(OUTPUT)/$(DATE)
	python3 main.py -i $i -o $(OUTPUT)/$(DATE)/$o.mp4 -t $(TIME) -g $(OUTPUT)/$(DATE)/$o.csv -f 90 -s $(TRIGGER) -c --exposure $(EXPOSURE_COLOR) -p
endef
$(foreach o, 0 1 2 3, $(foreach i,0 1,$(eval $(call make_capture_in_out_target))))

color_master_slave: color_master_0_0 color_slave_1_1

color_slave_slave: color_slave_0_2 color_slave_1_3

define make_capture_in_out_target
gray_master_$i_$o: $(OUTPUT)/$(DATE)
	sleep 1
	python3 main.py -i $i -o $(OUTPUT)/$(DATE)/$o.mp4 -t $(TIME) -g $(OUTPUT)/$(DATE)/$o.csv -f 150 -m $(TRIGGER) --exposure $(EXPOSURE_GRAY) -p

gray_slave_$i_$o: $(OUTPUT)/$(DATE)
	python3 main.py -i $i -o $(OUTPUT)/$(DATE)/$o.mp4 -t $(TIME) -g $(OUTPUT)/$(DATE)/$o.csv -f 150 -s $(TRIGGER) --exposure $(EXPOSURE_GRAY) -p
endef
$(foreach o, 0 1 2 3, $(foreach i,0 1,$(eval $(call make_capture_in_out_target))))

gray_master_master: gray_master_0_0 gray_master_1_1

gray_master_slave: gray_master_0_0 gray_slave_1_1

gray_slave_slave: gray_slave_0_2 gray_slave_1_3
