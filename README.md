# PSP-Lab1: Visualization of Fourier Series

Lab 1 _Visualization of Fourier Series_ in course _Principles of Signal Processing_ by Prof. Jia Jia at DSCT, THU

## Quick Start

To setup the environment, please run the following command:

```bash
conda create -n viz-fs python=3.11 -y
conda activate viz-fs
pip install -r requirements.txt
# pip install -r requirements-dev.txt # for development, check the comments for more details
```

Then you can run the [`exp1.ipynb`](./exp1.ipynb) or run the script:

```bash
python exp1.py
```

Finally, check the videos in the [`results`](./results) folder.

# PSP-Lab1 报告: 傅⾥叶级数的可视化

## 任务⼀: 可视化方波信号

### 方波信号

$$f(t) = 0.5 \text{sgn}(\sin(t)) + 0.5$$

### 方波信号的傅里叶级数

计算得到

$$f(t) = 0.5 + \frac{2}{\pi} * (\sin t + \frac{1}{3} \sin 3t + ... + \frac{1}{n} \sin nt)$$

### 实现思路

对 $n$ 分类返回对应常数即可：

```python
# For square wave, coefficients have analytical solutions
if n == 0:
    return 0.5  # DC component (a0)
elif n % 2 == 0:
    return 0  # Even coefficients are zero due to symmetry
else:
    # Odd coefficients follow 2/(nπ) pattern for sine terms only
    return 2 / (math.pi * (n + 1) / 2) if n % 4 == 1 else 0
```

## 任务二: 可视化半圆波信号（选做）

### 半圆波信号

$$f(t) = \sqrt{\pi^2 - (t - \pi)^2}$$

### 半圆波信号的傅里叶级数

表达式如下：

$$
\begin{aligned}
a_0=&\frac{1}{T_1} \int_{t_0}^{t_0+T_1} f(t) d t \\
a_n=&\frac{2}{T_1} \int_{t_0}^{t_0+T_1} f(t) \cos (n \omega_1 t) d t \\
b_n=&\frac{2}{T_1} \int_{t_0}^{t_0+T_1} f(t) \sin (n \omega_1 t) d t
\end{aligned}
$$

### 实现思路

难以直接求解，调用数值算法 `np.trapzoid` 积分

```python
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
```
