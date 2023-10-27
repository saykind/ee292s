#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
This program measures acceleration of RPI, then uses gyro to calculate its orientation
and subtracts contribution from gravity.
Code assumes that at t=0 there is only gravity (no other acceleration).
Next it usess Kalman filter to double-integrate acceleration and measure distance.
It plots distance.
"""
from header import *
from filterpy.kalman import KalmanFilter
#plt.rcParams['text.usetex'] = True

GyroA = np.array([0,0,0])

def plot_init():
  """ Creates axes to be used for real time data plotting."""
  fig, axs = plt.subplots(2, 2, figsize=(6, 4), sharex='col')
  
  titles = [["X", "Y"], ["Processor frequency", "Z"]]
  ylabels = [["SI units", "SI units"], ["frequency, Hz", "SI units"]]
  for i in range(2):
    for j in range(2):
      axs[i,j].grid()
      axs[i,j].set_title(titles[i][j])
      axs[i,j].set_ylabel(ylabels[i][j])
      axs[1,j].set_xlabel("Time, sec")
  
  domain = np.linspace(0,10,100)
  val = np.linspace(-1,1,100)
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

def generate_F(dt):
  return np.array([[1, 0, 0, dt, 0, 0, dt*dt/2, 0, 0],
                   [0, 1, 0, 0, dt, 0, 0, dt*dt/2, 0],
                   [0, 0, 1, 0, 0, dt, 0, 0, dt*dt/2],
                   [0, 0, 0, 1, 0, 0, dt, 0, 0],
                   [0, 0, 0, 0, 1, 0, 0, dt, 0],
                   [0, 0, 0, 0, 0, 1, 0, 0, dt],
                   [0, 0, 0, 0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 1]])

def kalman_distance_init(dt):
  kf = KalmanFilter(dim_x=9, dim_z=3)
  # State transition matrix
  kf.F = generate_F(dt)
  # Measurement array
  kf.H = np.array([[0, 0, 0, 0, 0, 0, 1., 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 1., 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 1.],])
  # Initial conditions (values x, variances p)
  kf.x = np.zeros(kf.dim_x)
  kf.P = np.eye(kf.dim_x)
  # Variance of measurement
  kf.Q = np.eye(kf.dim_x)
  kf.R = np.eye(kf.dim_z)
  
  return kf
  
def stop_motion(kf):
  kf.x[3:6] = 0

  

def kalman_accel_init(x0):
  kf = KalmanFilter(dim_x=3, dim_z=6)
  # State transition matrix
  kf.F = np.eye(kf.dim_x)
  # Measurement array
  kf.H = np.array([[1., 0, 0],
                   [0, 1., 0],
                   [0, 0, 1.],
                   [1., 0, 0],
                   [0, 1., 0],
                   [0, 0, 1.]])
  # Initial conditions (values x, variances p)
  kf.x = x0
  kf.P = 1e3*np.eye(kf.dim_x)
  # Variance of measurement
  kf.Q = 1e4*np.eye(kf.dim_x)
  kf.R = 1e4*np.eye(kf.dim_z)
  
  return kf



"""           """
"""   MAIN    """
"""           """
if __name__ == '__main__':
  icm20948=ICM20948()

  fig, axs, lns, bg, domain = plot_init()
  dim_x = 9
  dim_z = 3
  acc = np.zeros(dim_z)
  pos = np.zeros((dim_z, domain.size))
  x   = np.zeros((dim_x, domain.size))
  freqs = np.zeros(domain.size)
  
  # Initialize gyro direction
  icm20948.icm20948_Gyro_Accel_Read()
  GyroA = np.array(Accel).copy()
  GyroA0 = GyroA.copy()
  
  # Initialize Kalman
  kf = kalman_distance_init(.05)
  kfa = kalman_accel_init(GyroA0)
  
  dt0 = time.time()
  while True:
    icm20948.icm20948_Gyro_Accel_Read()

    # Measure effective processor frequency
    dt1 = time.time()
    dt = dt1-dt0
    dt0 = dt1
    freqs[:-1] = freqs[1:]
    freqs[-1] = 1./dt
    
    # Update gravity vector used by gyro
    if np.linalg.norm(np.var(x[-3:,-20:], axis=1)) < 1e-5:
      GyroA = np.array(Accel)
      stop_motion(kf)
    else:
      rotate_GyroA(dt)
    
    # Update acceleration using Kalman
    kfa.predict()
    z = np.array([Accel, GyroA])
    z = z.reshape((1,6))
    kfa.update(z)
    
    # Substract gravity from acceleration vector
    acc_sens = 16384/9.81
    acc = (kfa.x-GyroA)/acc_sens
    
    # Kalman update
    kf.K = generate_F(dt)
    kf.predict()
    kf.update(acc)
    
    # Plot acceleration components
    x[:,:-1] = x[:,1:]
    x[:,-1]  = kf.x
    x[-3:,-1] = acc
    
    vals = np.array([[x[0,:], x[1,:]],[freqs, x[2,:]]]) #position
    #vals = np.array([[x[3,:], x[4,:]],[freqs, x[5,:]]]) #velocity
    #vals = np.array([[x[6,:], x[7,:]],[freqs, x[8,:]]]) #acceleration
    plot_flush(fig, axs, lns, bg, vals)

