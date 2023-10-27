#!/usr/bin/python
# -*- coding:utf-8 -*-
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from header import *
import numpy as np

GyroA = np.array([0,0,0])

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

  kf.Q = Q_discrete_white_noise(3, dt=0.1, var=1)
  kf.R = np.array([[measVar]]) # Variance of measurement

  return kf


def x_gravity():
  AccelX = Accel[0]/16384 * 9.81
  AccelY = Accel[1]/16384 * 9.81
  AccelZ = Accel[2]/16384 * 9.81

  g = 9.81 * math.sin(math.acos(AccelZ / (math.sqrt(AccelX ** 2 + AccelY ** 2 + AccelZ ** 2))))

  return g


def calibrateSensors():
  print('Calibrating Sensors')
  N = 0
  AccelXVals = []

  while N != 500:
    icm20948.icm20948_Gyro_Accel_Read()
    time.sleep(0.1)

    AccelXVals.append(Accel[0]/16384*9.81)
    N += 1

  return sum(AccelXVals)/N


def polar_angle(n):
  return math.degrees(math.acos(n[2]/np.linalg.norm(n)))


def rotate_GyroA(dt):
  """ Changes value of the global variable GyroA.
      I follow this order of operations 
      https://msl.cs.uiuc.edu/planning/node102.html
  """
  gyro_sens = 32.8
  a = -math.radians(Gyro[2]*dt/gyro_sens)
  b = math.radians(Gyro[1]*dt/gyro_sens)
  c = math.radians(Gyro[0]*dt/gyro_sens)

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


if __name__ == '__main__':
  kf = KalmanFilterSetup()

  icm20948=ICM20948()

  icm20948.icm20948_Gyro_Accel_Read()
  GyroA = np.array(Accel).copy()

  accelCalX = calibrateSensors()


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

    # Gravity level
    g = x_gravity()
    print(f'Leakage Gravity: {g}')

    # Convert Acceleratometer Output
    AccelX = Accel[0]/16384 * 9.81 - accelCalX

    # Kalman Filter
    kf.predict()
    kf.update([AccelX])


    print(f'Position: {kf.x[0]} m\nVelocity: {kf.x[1]} m/s\nAcceleration: {kf.x[2]} m/s^2')
    print(f'Sensor Acceleration: {AccelX}')
    print(f'=================================================================')

    calc = True
