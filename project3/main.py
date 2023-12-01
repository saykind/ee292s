#!/usr/bin/python
# -*- coding:utf-8 -*-


import time
import ADS1256
import RPi.GPIO as GPIO
import numpy as np
import matplotlib.pyplot as plt


### Constants
CHANNEL = 7
N = 60000 # number of points
RATE = 300 # Hz, approximate
PERIOD = N/RATE # seconds, approximate


### Plot functions
def plot_init():
  """ Creates axes to be used for real time data plotting."""
  fig, axs = plt.subplots(2, 2, figsize=(6, 4), sharex='col')
  
  titles = [["Signal", "FFT R"], ["Processor frequency", "FFT X"]]
  ylabels = [["a.u.", "a.u"], ["freq, Hz", "freq, Hz"]]
  for i in range(2):
    for j in range(2):
      axs[i,j].grid()
      axs[i,j].set_title(titles[i][j])
      axs[i,j].set_ylabel(ylabels[i][j])
  axs[1,0].set_xlabel("Time, sec")
  axs[1,1].set_xlabel("Freq, Hz")

  domain_t = np.linspace(0, PERIOD, N)
  val = -.02+.1*np.linspace(0,1,N)
  (ln00,) = axs[0,0].plot(domain_t, val, 'k', animated=True)
  val = 295+20*np.linspace(0,1,N)
  (ln10,) = axs[1,0].plot(domain_t, val, 'r.', animated=True)
  
  N_f = N//40
  domain_f = np.linspace(0, RATE*(N_f/N), N_f)
  val = np.linspace(0,N_f/8,N_f)
  (ln01,) = axs[0,1].plot(domain_f, val, 'k', animated=True)
  val = np.linspace(0,N_f/8,N_f)
  (ln11,) = axs[1,1].plot(domain_f, val, 'k', animated=True)
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
  return fig, axs, lns, bg, domain_t, domain_f
  
def plot_flush(fig, axs, lns, bg, vals_t, vals_f):
  """ Blitting function."""
  fig.canvas.restore_region(bg)
  for i in range(2):
    lns[i,0].set_ydata(vals_t[i])
    axs[i,0].draw_artist(lns[i,0])
  for i in range(2):
    lns[i,1].set_ydata(vals_f[i])
    axs[i,1].draw_artist(lns[i,1])
  fig.canvas.blit(fig.bbox)
  fig.canvas.flush_events()

def make_filename():
  """ Makes a filename for saving data."""
  t = time.localtime()
  filename = f"data{RATE}Hz/" \
             f"data{N // RATE}_{t.tm_year}-{t.tm_mon}-{t.tm_mday}_" \
             f"{t.tm_hour}-{t.tm_min}-{t.tm_sec}.npz"
  return filename


### Main function
if __name__ == '__main__':
  fig, axs, lns, bg, domain_t, domain_f = plot_init()
  N_f = domain_f.size
  signal = np.zeros(domain_t.size)
  freqs = np.zeros(domain_t.size)
  
  try:
    ADC = ADS1256.ADS1256()
    ADC.ADS1256_init()
    #ADC.ADS1256_EnableInputBuffer()
    
    
    while(True):
      t0 = time.time()
      for i in range(domain_t.size):
        signal[i] = ADC.ADS1256_GetChannelValue(CHANNEL)*5.0/0x7fffff
      
        #Measure processor frequency
        t1 = time.time()
        dt = t1-t0
        t0 = t1
        freqs[i] = 1./dt
       
      #Make a plot
      signal = signal - np.mean(signal)
      fft = np.fft.fft(signal)[:N_f]
      vals_t = np.array([[signal],[freqs]])
      vals_f = np.array([[np.abs(fft)],[fft.real]])
      plot_flush(fig, axs, lns, bg, vals_t, vals_f)
      np.savez(make_filename(), signal=signal, freqs=freqs)
        
  except Exception as err:
    print(f"Error: {err=}, {type(err)=}")
    GPIO.cleanup()
    print ("\r\nProgram end     ")
    exit()
