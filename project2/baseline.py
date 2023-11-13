#!/usr/bin/python

# Script to get the baseline values for different PRBS

# -*- coding:utf-8 -*-

import time
import ADS1256
import DAC8532
import RPi.GPIO as GPIO
import numpy as np
import matplotlib.pyplot as plt

# Correlation function
def xcor(array1, array2):
	xcor = np.zeros(array1.size)
	for i in range(array1.size):
		array2_delay = np.roll(array2, i)
		products = array1 * array2_delay
		xcor[i] = np.sum(products)

	return xcor

# Generate PRBS sequence given taps
def PRBS(taps, start=1):
	maximal_length = 2 ** len(taps) - 1
	taps = int(taps, 2)
	prbs = ""
	count = 0
	lfsr = start

	while True:
        	lsb = lfsr & 1
        	prbs += str(lsb)
        	lfsr = lfsr >> 1

	        if lsb == 1:
        		lfsr = lfsr ^ taps
        	count +=1

        	if lfsr == start:
        		break

	print(f'PRBS{maximal_length}: {prbs}')
	print(f'Count: {count}')

	if count == maximal_length:
        	print(f"Polynomial is maximal: {maximal_length}")
	else:
        	print(f"Polynomial is not maximal. Maximal length is {maximal_length}")

	PRBS = []
	for bit in prbs:
		if bit == '1':
			PRBS.append(1)
		else:
			PRBS.append(0)

	return np.array(PRBS, dtype=int)


# Drive all pins to value of PRBS
def drive_and_sense_pins(ADC, pins, shifted_prbs, sense_pin):
	sense_array = []
	ADC.ADS1256_SetChannal(sense_pin)
	for i in range(PRBS_SIZE):
		T1 = time.time()
		GPIO.output(pins[0], shifted_prbs[0][i])
		GPIO.output(pins[1], shifted_prbs[1][i])
		GPIO.output(pins[2], shifted_prbs[2][i])
		GPIO.output(pins[3], shifted_prbs[3][i])
		GPIO.output(pins[4], shifted_prbs[4][i])


		ADC.ADS1256_WriteCmd(ADS1256.CMD['CMD_SYNC'])
		ADC.ADS1256_WriteCmd(ADS1256.CMD['CMD_WAKEUP'])
		sense_array.append(ADC.ADS1256_Read_ADC_Data() *5.0 / 0x7fffff)

		T2 = time.time()
		dt = T2 - T1
		#print(f'Freq: {1/dt}')
	return sense_array


# Drive Pins
drive_pins = [7, 12, 16, 20, 21]

prbs_len = ['511', '255', '127', '63', '31', '15', '7']
taps = ['100010000', '10111000', '1100000', '110000', '10100', '1100', '110']

ADC = ADS1256.ADS1256()
DAC = DAC8532.DAC8532()
ADC.ADS1256_init()

DAC.DAC8532_Out_Voltage(0x30, 3)
DAC.DAC8532_Out_Voltage(0x34, 3)

# Setup GPIOs
GPIO.setmode(GPIO.BCM)
for i in drive_pins:
	GPIO.setup(i, GPIO.OUT)

# For all PRBS find baseline
for i in range(len(prbs_len)):
	prbs_num = prbs_len[i]
	prbs = PRBS(taps[i])
	PRBS_SIZE = prbs.size

	# Create delay spacing for PRBS for each derive lines
	delay_spacing = np.linspace(0, PRBS_SIZE - 1, num=5, dtype=np.int16)

	# Make matrix of shifted PRBS
	shifted_prbs = []
	for delay in delay_spacing:
		prbs_delayed = np.roll(prbs, delay)
		prbs_delayed = prbs_delayed.tolist()
		shifted_prbs.append(prbs_delayed)

	count = 0
	dt = 0
	sense = np.zeros(shape=(7, PRBS_SIZE))
	sense_array = np.zeros(shape=(7, PRBS_SIZE))
	while True:
		# PRBS in Drive Pins and sense one row at a time
		T1 = time.time()
		for sense_row in range(1, 8):
			sense[7 - sense_row] = np.array(drive_and_sense_pins(ADC, drive_pins, shifted_prbs, sense_row))
		T2 = time.time()
		dt += (T2 - T1)

		count += 1
		sense_array = sense_array + sense

		if count == 50:
			sense_array = sense_array/count
			avg_period = dt/count
			avg_fps = 1/avg_period
			print(f'FPS for PRBS{prbs_num}: {avg_fps}')
			break



	sense_npz = np.array(sense_array)

	with open(f'notouch_prbs{prbs_num}.npy', 'wb') as f:
		np.save(f, sense_npz)

"""
FPS for PRBS511: 1.073053152565141
FPS for PRBS255: 2.1515564279619603
FPS for PRBS127: 4.319205488556428
FPS for PRBS63: 8.705741807243275
FPS for PRBS31: 17.663965435380135
FPS for PRBS15: 36.320376887089566
FPS for PRBS7: 77.11280009119021
"""
