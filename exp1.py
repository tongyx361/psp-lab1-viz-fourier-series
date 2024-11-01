#!/usr/bin/env python
# coding: utf-8

# # Visualization of Fourier Series

import math
import os

import imageio
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import cm
from tqdm import tqdm


class FourierSeriesVisualizer:
    """
    Parameters
    ----------
    signal_name: str
        optional, implement visualization for semi-circle
    N_Fourier: int
        1. Change N_Fourier to 2, 4, 8, 16, 32, 64, 128, get visualization results with differnet number of Fourier Series
    num_samples: int
        Number of samples for the original function.
    """

    def __init__(
        self,
        signal_name: str = "square",
        N_Fourier: int = 64,
        num_samples: int = 1000,
    ):
        self.signal_name = signal_name
        self.N_Fourier = N_Fourier
        self.num_samples = num_samples

    def fourier_coefficient(self, n):
        """DONE: 4. Calculate the nth Fourier coefficient for either square wave or semi-circle wave.

        For a periodic function f(t), the Fourier series is:
        f(t) = a0/2 + Σ(an*cos(nωt) + bn*sin(nωt))

        This function returns coefficients in the following order:
        n = 0: returns a0 (DC component)
        n = 1: returns b1 (first sine coefficient)
        n = 2: returns a1 (first cosine coefficient)
        n = 3: returns b2 (second sine coefficient)
        n = 4: returns a2 (second cosine coefficient)
        And so on...

        For square wave:
        - a0 = 0.5 (mean value)
        - an = 0 (all cosine terms are zero due to odd symmetry)
        - bn = 2/(nπ) for n odd, 0 for n even (due to half-wave symmetry)

        For semi-circle:
        Uses numerical integration (trapezoidal rule) to compute coefficients:
        - a0 = (1/2π) ∫f(t)dt over [0,2π]
        - an = (1/π) ∫f(t)cos(nt)dt over [0,2π]
        - bn = (1/π) ∫f(t)sin(nt)dt over [0,2π]
        """
        if self.signal_name == "square":
            # For square wave, coefficients have analytical solutions
            if n == 0:
                return 0.5  # DC component (a0)
            elif n % 2 == 0:
                return 0  # Even coefficients are zero due to symmetry
            else:
                # Odd coefficients follow 2/(nπ) pattern for sine terms only
                return 2 / (math.pi * (n + 1) / 2) if n % 4 == 1 else 0

        elif self.signal_name == "semicircle":
            # For semi-circle, use numerical integration
            x = np.linspace(0, 2 * math.pi, self.num_samples)  # Sample points
            y = np.zeros(self.num_samples, dtype=float)

            # Calculate function values at sample points
            for i in range(self.num_samples):
                y[i] = self.semi_circle_wave(x[i])

            if n == 0:
                # Calculate a0 coefficient (mean value)
                return np.trapezoid(y, x) / (2 * math.pi)
            elif n % 2 == 0:
                # Calculate an coefficients (cosine terms)
                for i in range(1000):
                    y[i] = y[i] * math.cos(n / 2 * x[i])
                return np.trapezoid(y, x) / math.pi
            else:
                # Calculate bn coefficients (sine terms)
                for i in range(1000):
                    y[i] = y[i] * math.sin((n + 1) / 2 * x[i])
                return np.trapezoid(y, x) / math.pi
        else:
            raise Exception("Unknown Signal")

    def square_wave(self, t):
        """DONE: 3. implement the signal function"""
        delta_t = (t + 2 * math.pi) % (2 * math.pi)
        y = 0 if delta_t <= 0 or delta_t >= math.pi else 1
        return y

    def semi_circle_wave(self, t):
        """DONE: optional. implement the semi circle wave function"""
        delta_t = (t + 2 * math.pi) % (2 * math.pi)
        y = np.sqrt(math.pi * math.pi - (delta_t - math.pi) ** 2)
        return y

    def function(self, t):
        if self.signal_name == "square":
            return self.square_wave(t)
        elif self.signal_name == "semicircle":
            return self.semi_circle_wave(t)
        else:
            raise Exception("Unknown Signal")

    def visualize(self, frames: int = 100, output_dir: str = "results"):
        """
        Parameters
        ----------
        frames: int
            Number of frames for the animation
        """
        signal_dir = os.path.join("tmp", self.signal_name)
        os.makedirs(signal_dir, exist_ok=True)

        # x and y are for drawing the original function
        x = np.linspace(0, 2 * math.pi, self.num_samples)
        y = np.zeros(self.num_samples, dtype=float)
        for i in range(self.num_samples):
            y[i] = self.function(x[i])

        for i in range(frames):
            figure, axes = plt.subplots()
            color = iter(cm.rainbow(np.linspace(0, 1, 2 * self.N_Fourier + 1)))

            time = 2 * math.pi * i / frames
            point_pos_array = np.zeros((2 * self.N_Fourier + 2, 2), dtype=float)
            radius_array = np.zeros((2 * self.N_Fourier + 1), dtype=float)

            point_pos_array[0, :] = [0, 0]
            radius_array[0] = self.fourier_coefficient(0)
            point_pos_array[1, :] = [0, radius_array[0]]

            circle = patches.Circle(
                point_pos_array[0], radius_array[0], fill=False, color=next(color)
            )
            axes.add_artist(circle)

            f_t = self.function(time)
            for j in range(self.N_Fourier):
                # calculate circle for a_{n}
                radius_array[2 * j + 1] = self.fourier_coefficient(2 * j + 1)
                point_pos_array[2 * j + 2] = [
                    point_pos_array[2 * j + 1][0]
                    + radius_array[2 * j + 1] * math.cos((j + 1) * time),  # x axis
                    point_pos_array[2 * j + 1][1]
                    + radius_array[2 * j + 1] * math.sin((j + 1) * time),
                ]  # y axis
                circle = patches.Circle(
                    point_pos_array[2 * j + 1],
                    radius_array[2 * j + 1],
                    fill=False,
                    color=next(color),
                )
                axes.add_artist(circle)

                # calculate circle for b_{n}
                radius_array[2 * j + 2] = self.fourier_coefficient(2 * j + 2)
                point_pos_array[2 * j + 3] = [
                    point_pos_array[2 * j + 2][0]
                    + radius_array[2 * j + 2] * math.sin((j + 1) * time),  # x axis
                    point_pos_array[2 * j + 2][1]
                    + radius_array[2 * j + 2] * math.cos((j + 1) * time),
                ]  # y axis
                circle = patches.Circle(
                    point_pos_array[2 * j + 2],
                    radius_array[2 * j + 2],
                    fill=False,
                    color=next(color),
                )
                axes.add_artist(circle)

            plt.plot(point_pos_array[:, 0], point_pos_array[:, 1], "o-")
            plt.plot(x, y, "-")
            plt.plot(
                [time, point_pos_array[-1][0]],
                [f_t, point_pos_array[-1][1]],
                "-",
                color="r",
            )
            plt.gca().set_aspect("equal", adjustable="box")
            plt.savefig(os.path.join(signal_dir, f"{i}.png"))
            # plt.show()
            plt.close()

        images = []
        for i in range(frames):
            images.append(imageio.imread(os.path.join(signal_dir, f"{i}.png")))
        os.makedirs(output_dir, exist_ok=True)
        imageio.mimsave(
            os.path.join(output_dir, f"{self.signal_name}-{self.N_Fourier}.mp4"), images
        )


if __name__ == "__main__":
    visualizer = FourierSeriesVisualizer()
    for signal_name in ["square", "semicircle"]:
        for N_Fourier in tqdm([2, 4, 8, 16, 32, 64, 128], desc=signal_name):
            visualizer.signal_name = signal_name
            visualizer.N_Fourier = N_Fourier
            visualizer.visualize()
