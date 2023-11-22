#!/usr/bin/python
# -*- coding:utf-8 -*-


import time
import ADS1256
import RPi.GPIO as GPIO
import numpy as np
import matplotlib.pyplot as plt


### Constants
CHANNEL = 7
MAX_FREQ = 95
SCAN_PERIOD = 1


### Plot functions
def plot_init():
  """ Creates axes to be used for real time data plotting."""
  fig, axs = plt.subplots(2, 2, figsize=(6, 4), sharex='col')
  
  titles = [["Signal", "FFT X"], ["Processor frequency", "FFT Y"]]
  ylabels = [["a.u.", "a.u"], ["freq, Hz", "freq, Hz"]]
  for i in range(2):
    for j in range(2):
      axs[i,j].grid()
      axs[i,j].set_title(titles[i][j])
      axs[i,j].set_ylabel(ylabels[i][j])
  axs[1,0].set_xlabel("Time, sec")
  axs[1,1].set_xlabel("Freq, Hz")

  N = 100
  domain = np.linspace(0, N/MAX_FREQ, N)
  val = 1+5*np.linspace(0,1,N)
  (ln00,) = axs[0,0].plot(domain, val, 'k', animated=True)
  val = 85+20*np.linspace(0,1,N)
  (ln10,) = axs[1,0].plot(domain, val, 'r', animated=True)
  val = N*domain
  (ln01,) = axs[0,1].plot(domain, val, 'k', animated=True)
  val = N*domain
  (ln11,) = axs[1,1].plot(domain, val, 'k', animated=True)
  lns = np.array([[ln00, ln01], [ln10, ln11]])

  plt.tight_layout()
  plt.show(block=False)
  plt.pause(.05)
  bg = fig.canvas.copy_from_bbox(fig.bbox)
  axs[0,0].draw_artist(ln00)
  axs[1,0].draw_artist(ln10)
  axs[0,1].draw_artist(ln01)
  axs[1,1].draw_artist(ln11)
  fig.canvas.blit(fig.bbox)
  return fig, axs, lns, bg, domain
  
def plot_flush(fig, axs, lns, bg, vals):
  """ Blitting function."""
  fig.canvas.restore_region(bg)
  for i in range(2):
    for j in range(2):
      lns[i,j].set_ydata(vals[i,j])
      axs[i,j].draw_artist(lns[i,j])
  fig.canvas.blit(fig.bbox)
  fig.canvas.flush_events()


### Main function
if __name__ == '__main__':
  fig, axs, lns, bg, domain = plot_init()
  signal = np.zeros(domain.size)
  freqs = np.zeros(domain.size)
  
  try:
    ADC = ADS1256.ADS1256()
    ADC.ADS1256_init()
    #ADC.ADS1256_EnableInputBuffer()
    
    
    while(True):
      t0 = time.time()
      for i in range(domain.size):
        signal[i] = ADC.ADS1256_GetChannelValue(CHANNEL)*5.0/0x7fffff
      
        #Measure processor frequency
        t1 = time.time()
        dt = t1-t0
        t0 = t1
        freqs[i] = 1./dt
       
      #Make a plot
      fft = np.fft.fft(signal)
      values = np.array([[signal,fft.real],[freqs,fft.imag]])
      plot_flush(fig, axs, lns, bg, values)
        
  except Exception as err:
    print(f"Error: {err=}, {type(err)=}")
    GPIO.cleanup()
    print ("\r\nProgram end     ")
    exit()
