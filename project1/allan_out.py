import numpy as np
import time
import allantools
import matplotlib.pyplot as plt

if __name__ == '__main__':
  imu_vals = np.load('IMU_output2.npz')

  AX = imu_vals['AX']
  AY = imu_vals['AY']
  AZ = imu_vals['AZ']
  GX = imu_vals['GX']
  GY = imu_vals['GY']
  GZ = imu_vals['GZ']

  print(AX)
  print(AY)
  print(AZ)
  print(GX)
  print(GY)
  print(GZ)

  (ax_tau_out, ax_adev, adeverr, n) = allantools.adev(AX, rate=10, data_type='freq', taus='all')
  (ay_tau_out, ay_adev, adeverr, n) = allantools.adev(AY, rate=10, data_type='freq', taus='all')
  (az_tau_out, az_adev, adeverr, n) = allantools.adev(AZ, rate=10, data_type='freq', taus='all')

  (gx_tau_out, gx_adev, adeverr, n) = allantools.adev(GX, rate=10, data_type='freq', taus='all')
  (gy_tau_out, gy_adev, adeverr, n) = allantools.adev(GY, rate=10, data_type='freq', taus='all')
  (gz_tau_out, gz_adev, adeverr, n) = allantools.adev(GZ, rate=10, data_type='freq', taus='all')

  plt.figure()
  plt.loglog(ax_tau_out, ax_adev, label='ax')
  plt.loglog(ay_tau_out, ay_adev, label='ay')
  plt.loglog(az_tau_out, az_adev, label='az')
  plt.legend()
  plt.grid()
  plt.xlabel('taus')
  plt.ylabel('ADEV [ms^-2]')


  plt.figure()
  plt.loglog(gx_tau_out, gx_adev, label='gx')
  plt.loglog(gy_tau_out, gy_adev, label='gy')
  plt.loglog(gz_tau_out, gz_adev, label='gz')
  plt.legend()
  plt.grid()
  plt.xlabel('taus')
  plt.ylabel('ADEV [dps]')


  plt.legend()
  plt.show()


