#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
This program measures acceleration of RPI, then uses gyro to calculate its orientation
and subtracts contribution from gravity.
Code assumes that at t=0 there is only gravity (no other acceleration).
It plots acceleration (gravity excluded).
"""
from header import *
GyroA = np.array([0,0,0])

def plot_init():
  """ Creates axes to be used for real time data plotting."""
  fig, axs = plt.subplots(2, 2, figsize=(6, 4), sharex='col')
  
  titles = [["Acc X", "Acc Y"], ["Processor frequency", "Acc Z"]]
  ylabels = [["acc, g", "acc, g"], ["frequency, Hz", "acc, g"]]
  for i in range(2):
    for j in range(2):
      axs[i,j].grid()
      axs[i,j].set_title(titles[i][j])
      axs[i,j].set_ylabel(ylabels[i][j])
      axs[1,j].set_xlabel("Time, sec")
  
  domain = np.linspace(0,10,100)
  val = np.linspace(-1.5,1.5,100)
  val_freq = 5+2*domain
  (ln00,) = axs[0,0].plot(domain, val, 'k', animated=True)
  (ln10,) = axs[1,0].plot(domain, val_freq, 'r', animated=True)
  (ln01,) = axs[0,1].plot(domain, val, 'k', animated=True)
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
  
def polar_angle(n):
  return math.degrees(math.acos(n[2]/np.linalg.norm(n)))

def azimuth_angle(n = Accel):
  return math.degrees(math.acos(n[0]/np.linalg.norm(n[:2])))
  
def rotate_GyroA(dt):
  """ Changes value of the global variable GyroA.
      I follow this order of operations 
      https://msl.cs.uiuc.edu/planning/node102.html
  """
  gyro_sens = 32.8	
  a = -math.radians(Gyro[2]*dt/gyro_sens)
  b = -math.radians(Gyro[1]*dt/gyro_sens)
  c = -math.radians(Gyro[0]*dt/gyro_sens)
  
  sa = math.sin(a)
  ca = math.cos(a)
  sb = math.sin(b)
  cb = math.cos(b)
  sc = math.sin(c)
  cc = math.cos(c)
  g = np.array(GyroA).copy()
  GyroA[0] = ca*cb*g[0] + (ca*sb*sc-sa*cc)*g[1] + (ca*sb*cc+sa*sc)*g[2]
  GyroA[1] = sa*cb*g[0] + (sa*sb*sc+ca*cc)*g[1] + (sa*sb*cc-ca*sc)*g[2]
  GyroA[2] =   -sb*g[0] +            cb*sc*g[1] +            cb*cc*g[2]



"""           """
"""   MAIN    """
"""           """
if __name__ == '__main__':
  icm20948=ICM20948()

  fig, axs, lns, bg, domain = plot_init()
  acc  = np.zeros(3)
  accX = np.zeros(domain.size)
  accY = np.zeros(domain.size)
  accZ = np.zeros(domain.size)
  freqs = np.zeros(domain.size)
  
  # Initialize gyro direction
  icm20948.icm20948_Gyro_Accel_Read()
  GyroA = np.array(Accel).copy()
  
  dt0 = time.time()
  while True:
    icm20948.icm20948_Gyro_Accel_Read()

    # Measure effective processor frequency
    dt1 = time.time()
    dt = dt1-dt0
    dt0 = dt1
    freqs[:-1] = freqs[1:]
    freqs[-1] = 1./dt
    
    # Calculate angle from gyroscope
    rotate_GyroA(dt)
    
    # Substract gravity from acceleration vector
    acc_sense = 16384
    acc = np.array(Accel-GyroA).copy()
    
    # Plot acceleration components
    accX[:-1] = accX[1:]
    accY[:-1] = accY[1:]
    accZ[:-1] = accZ[1:]
    accX[-1]  = acc[0]/acc_sense
    accY[-1]  = acc[1]/acc_sense
    accZ[-1]  = acc[2]/acc_sense
    
    
    vals = np.array([[accX, accY],[freqs, accZ]])
    plot_flush(fig, axs, lns, bg, vals)

