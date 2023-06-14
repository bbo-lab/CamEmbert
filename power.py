import psutil
import time
#import os
import beepy
while True:
	if not psutil.sensors_battery().power_plugged:
		#os.system('spd-say "running on battery"')
		beepy.beep(sound="error")
	time.sleep(2)
