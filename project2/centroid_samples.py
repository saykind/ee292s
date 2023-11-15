#!/usr/bin/python
# -*- coding:utf-8 -*-

#Mean: (1.896, 3.476)
#STD: (0.0099, 0.026)

import time
import ADS1256
import DAC8532
import RPi.GPIO as GPIO
import numpy as np
import matplotlib.pyplot as plt

"""
FPS without plots
FPS for PRBS511: 1.073053152565141
FPS for PRBS255: 2.1515564279619603
FPS for PRBS127: 4.319205488556428
FPS for PRBS63: 8.705741807243275
FPS for PRBS31: 17.663965435380135
FPS for PRBS15: 36.320376887089566
FPS for PRBS7: 77.11280009119021
"""
threshold = 2
spacing = 9 # Spacing in mm

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


# Initiate Plots
def plot_init(PRBS_SIZE):
	fig, ax = plt.subplots(1, 2, figsize=(6, 3))
	fig.subplots_adjust(left=0,
						bottom=0.1, 
						right=0.85, 
						top=0.9, 
						wspace=0.2, 
						hspace=0.4)

	touch_screen = np.zeros(shape=(7, 5))

	x = np.arange(0, PRBS_SIZE, 1)
	cor_plots = []

	heatmap = ax[0].imshow(touch_screen, vmin=0, vmax=PRBS_SIZE, cmap='inferno', interpolation='nearest', animated=True)
	cbar = ax[0].figure.colorbar(heatmap, ax=ax, location='left')

	ax[0].set_title('Touch Heatmap')
	ax[0].set_xticks([0, 1, 2, 3, 4])

	for i in range(7):
		(cor_plot, ) = ax[1].plot(x, x, animated=True)
		cor_plots.append(cor_plot)

	ax[1].set_title('Correlation')
	ax[1].set_xticks(np.arange(0, PRBS_SIZE + 1, (PRBS_SIZE + 1)/4))
	ax[1].grid()

	# Legend
	ax[1].legend(['0', '1', '2', '3', '4', '5', '6'], loc='center left', bbox_to_anchor=(1, 0.5))

	ax[0].figure.canvas.draw()
	ax[1].figure.canvas.draw()

	heatmap_bg = fig.canvas.copy_from_bbox(ax[0].bbox)
	cor_bg = fig.canvas.copy_from_bbox(ax[1].bbox)
	plt.show(block=False)
	plt.pause(0.000001)

	ax[0].draw_artist(heatmap)
	ax[1].draw_artist(cor_plot)

	fig.canvas.blit(fig.bbox)

	return fig, ax, heatmap_bg, cor_bg, heatmap, cor_plots


# Update heatmap
def update_heatmap(fig, plot, ax, bg, data):
	fig.canvas.restore_region(bg)
	plot.set_data(data)
	ax.draw_artist(plot)
	fig.canvas.blit(fig.bbox)
	fig.canvas.flush_events()


# Update plot
def update_plot(fig, plots, ax, bg, ydata):
	fig.canvas.restore_region(bg)
	for i, plot in enumerate(plots):
		plot.set_ydata(ydata[i])
		ax.draw_artist(plot)
	fig.canvas.blit(fig.bbox)
	fig.canvas.flush_events()



# Drive all pins to value of PRBS
def drive_and_sense_pins(ADC, pins, shifted_prbs, sense_pin):
	sense_array = []
	ADC.ADS1256_SetChannal(sense_pin)
	for i in range(PRBS_SIZE):
		#T1 = time.time()
		GPIO.output(pins[0], shifted_prbs[0][i])
		GPIO.output(pins[1], shifted_prbs[1][i])
		GPIO.output(pins[2], shifted_prbs[2][i])
		GPIO.output(pins[3], shifted_prbs[3][i])
		GPIO.output(pins[4], shifted_prbs[4][i])

		ADC.ADS1256_WriteCmd(ADS1256.CMD['CMD_SYNC'])
		ADC.ADS1256_WriteCmd(ADS1256.CMD['CMD_WAKEUP'])
		sense_array.append(ADC.ADS1256_Read_ADC_Data() *5.0 / 0x7fffff)

		#T2 = time.time()
		#dt = T2 - T1
		#print(f'Freq: {1/dt}')
	return sense_array


# Calculate centroid
def centroid(caps):
	total_weight = np.sum(caps)

	x = 0
	y = 0

	# X Coordinate
	for i in range(5):
		x_weight = np.sum(caps[:, i])
		x += i*(x_weight/total_weight)

	# Y Coordinate
	for i in range(7):
		y_weight = np.sum(caps[i, :])
		y += i*(y_weight/total_weight)

	digit_x = round(x)
	digit_y = round(y)

	# Major/Minor Axes
	x_axes = 2 * np.std(caps[:, digit_x]) / np.sum(caps[:, digit_x]) * spacing
	y_axes = 2 * np.std(caps[digit_y, :]) / np.sum(caps[digit_y, :]) * spacing

	coords = (x * spacing, y * spacing)
	major_minor_axes = (max(x_axes, y_axes), min(x_axes, y_axes))

	return coords, major_minor_axes


# Drive Pins
drive_pins = [7, 12, 16, 20, 21]

# Create PRBS -> Smaller PRBS has worse SNR ratio, more difficult to distinguish between touch in sense line from threshold

# PRBS511 (1 FPS)
# taps = "100010000"

# PRBS255 (2 FPS)
# taps = '10111000'

# PRBS127 (4 FPS)
# taps = '1100000'

# PRBS63 (8 FPS)
# taps = '110000'

# PRBS31 (16 FPS) -> Smallest with best SNR, better SNR from more signal gain
#taps = '10100'

# PRBS15 (36 FPS) -> Smallest PRBS we can use (any lower and the delays are too close to each other -- some leakage into other delays) (spaced for enough for leakage between delays to not happen)
taps = '1100'

# PRBS7 (72 FPS) -> Some leakage in other delays, difficult to distinguish minor axes
# taps = '110'
prbs = PRBS(taps)
PRBS_SIZE = prbs.size
prbs_name = f'prbs{PRBS_SIZE}'

# Create delay spacing for PRBS for each drive lines
delay_spacing = np.linspace(0, prbs.size - 1, num=5, dtype=np.int16)

print(delay_spacing)
print(f'Delays: {delay_spacing}')
# Make matrix of shifted PRBSes
shifted_prbs = []
for delay in delay_spacing:
	prbs_delayed = np.roll(prbs, delay)
	prbs_delayed = prbs_delayed.tolist()
	shifted_prbs.append(prbs_delayed)


ADC = ADS1256.ADS1256()
DAC = DAC8532.DAC8532()
ADC.ADS1256_init()

DAC.DAC8532_Out_Voltage(0x30, 3)
DAC.DAC8532_Out_Voltage(0x34, 3)

# Setup GPIOs
GPIO.setmode(GPIO.BCM)

for i in drive_pins:
	GPIO.setup(i, GPIO.OUT)

count = 0

# Baseline
with open(f'notouch_{prbs_name}.npy', 'rb') as f:
	baseline = np.load(f)
	baseline0 = baseline[0]


sense_array = np.zeros(shape=(7, PRBS_SIZE))
touch_screen = np.zeros(shape=(7, 5))

# Setup Plots
fig, ax, heatmap_bg, cor_bg, heatmap, cor_plots = plot_init(PRBS_SIZE)

count = 0
x = []
y = []
while(1):
	T1 = time.time()
	sense = np.zeros(shape=(7, PRBS_SIZE))
	cor = np.zeros(shape=(7, PRBS_SIZE))

	# PRBS in Drive Pins and sense one row at a time
	for sense_row in range(1, 8):
		sense[7 - sense_row] = np.array(drive_and_sense_pins(ADC, drive_pins, shifted_prbs, sense_row))

	sensed = sense - baseline

	for i in range(7):
		cor[i] = xcor(sensed[i], prbs)
		cor[i] = cor[i] - np.min(cor[i])

	caps = cor[:, delay_spacing]

	# Set thresholds (change threshold depending on PRBS -- look at correlation graph)
	touch_screen = np.copy(caps)
	touch_screen[touch_screen < threshold] = 0

	# Update Plots
	update_heatmap(fig, heatmap, ax[0], heatmap_bg, touch_screen)
	update_plot(fig, cor_plots, ax[1], cor_bg, cor)
	T2 = time.time()
	FPS = 1/(T2-T1)


	# Get centroid coordinates
	if np.any(touch_screen):
		centroid_coords, mm_axes = centroid(caps)

		x.append(centroid_coords[0])
		y.append(centroid_coords[1])
		count += 1

		if count == 1000:
			x = np.array(x)
			y = np.array(y)

			x_mean = np.mean(x) * spacing
			y_mean = np.mean(y) * spacing

			x_std = np.std(x) * spacing
			y_std = np.std(y) * spacing

			print(f'Mean: ({x_mean}, {y_mean})\nSTD: ({x_std}, {y_std}))')

			with open('x_data.npy', 'wb') as f1:
				np.save(f1, x)
			with open('y_data.npy', 'wb') as f2:
				np.save(f2, y)
			break
		print(f'Centroid Coordinates (X, Y): {centroid_coords}')
		#print(f'Major Axes: {mm_axes[0]}')
		#print(f'Minor Axes: {mm_axes[1]}')

	# print('FPS:', FPS)

"""
plt.imshow(caps, cmap='hot', interpolation='nearest', animated=True)
plt.title(f'Heatmap of Touch for PRBS{PRBS_SIZE} - Touch at Top Left Corner')
plt.show()

for i in range(len(acor)):
	plt.plot(acor[i])

plt.legend(['Row 1','Row 2','Row 3','Row 4','Row 5','Row 6', 'Row 7'])
plt.title(f'Correlation for Sense Rows at PRBS{PRBS_SIZE} - Touch at Top Left Corner')
plt.grid()
plt.show()
break
"""



