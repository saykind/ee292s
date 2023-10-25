#!/usr/bin/python
# -*- coding:utf-8 -*-
from filterpy.kalman import KalmanFilter
from header import *
import numpy as np

def KalmanFilterSetup():
  measVar = 0.25   # Noise is 0.5 m/s^2 
  kf = KalmanFilter(dim_x=3,dim_z=1)
  kf.F = np.array([[1, 0.1, 0.5 * 0.1 ** 2],
                  [0, 1, 0.1],
                  [0, 0, 1]])
  kf.H = np.array([[0, 0, 1]])

  # Iniial conditions: At rest, no acceleration, no velocity, pos = 0
  # Guess on covariance matrix: 0.1 m in position, 0.1 m/s in velocity, 0.1 m/s^2 in acceleration
  kf.x = ([0, 0, 0])
  kf.P = np.array([[0.1, 0, 0],
                  [0, 0.1, 0],
                  [0, 0, 0.1]])

  kf.Q = np.array([[0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0]])

  kf.R = np.array([[measVar]]) # Variance of measurement

  return kf

if __name__ == '__main__':
  kf = KalmanFilterSetup()

  icm20948=ICM20948()

  sleep_time = 0.1
  calc = False

  dt0 = time.time()
  while True:
    icm20948.icm20948_Gyro_Accel_Read()
    if calc:
      sleep_time = 0.1 - (time.time() - dt0)
    time.sleep(sleep_time)

    # Measure effective processor frequency
    dt1 = time.time()
    dt = dt1-dt0
    dt0 = dt1

    print(f'Sample Time: {dt} s')

    # Convert Acceleratometer Output
    AccelX = Accel[0]/16384 * 9.81

    # Kalman Filter
    kf.predict()
    kf.update([AccelX])


    print(f'Position: {kf.x[0]} m\nVelocity: {kf.x[1]} m/s\nAcceleration: {kf.x[2]} m/s^2')
    print(f'Sensor Acceleration: {AccelX}')
    print(f'=================================================================')

    calc = True
