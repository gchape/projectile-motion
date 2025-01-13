# Ball Trajectory Simulation with Air Resistance

This project simulates the trajectory of a ball thrown with an initial velocity and angle, considering the effects of gravity and air resistance (drag). The simulation uses numerical methods to approximate the motion of the ball over time.


_shoot dem balls_ 🚀

https://github.com/user-attachments/assets/6b39fbe9-5f70-40db-afc8-adec8fbed86f

## Problem Setup

The ball is thrown with an initial velocity at a specific angle, and the following forces are considered:

- **Gravity**: Acts downward, with a constant acceleration of 9.81 m/s².
- **Air Resistance (Drag)**: Acts opposite to the direction of motion and affects both horizontal and vertical components of the ball’s velocity. The drag force is proportional to the square of the ball’s velocity.

The motion is governed by equations for both the horizontal and vertical directions, which are influenced by gravity and drag. The simulation uses numerical integration to approximate the ball’s trajectory over time.

## Motion Model

The motion of the ball is described by the following key components:

- **Horizontal Motion**: The ball’s horizontal velocity is affected by drag, causing a reduction in its horizontal speed over time.
- **Vertical Motion**: In addition to drag, gravity also influences the vertical motion, pulling the ball downward while air resistance reduces its upward speed.

Both of these components are updated at each time step using numerical methods.

## Numerical Integration

Euler's method is used for numerical integration, updating the velocity and position of the ball iteratively at each time step. The velocity components are updated based on the forces acting on the ball, and the positions are updated based on the velocities.

## Features

- **Air Resistance**: Accounts for the drag force that opposes the ball's motion, affecting both horizontal and vertical directions.
- **Multiple Targets**: The simulation can be configured to track whether the ball hits multiple targets in the trajectory.
- **Realistic Physics**: Incorporates both gravity and drag to model more realistic ball motion compared to idealized projectile motion.

## Algorithm Overview

1. **Initialization**: The initial velocity and angle of the ball are set. The ball's initial position is also defined.
2. **Time Steps**: At each time step, the velocity and position of the ball are updated using the forces of gravity and drag.
3. **Collision Detection**: The simulation checks if the ball has reached a target or collided with the ground.
4. **Repeat**: The process continues iteratively until the ball reaches its destination or all targets are hit.


![xyz](https://github.com/user-attachments/assets/32404159-2345-42c9-be0b-529c2fad84f7)
