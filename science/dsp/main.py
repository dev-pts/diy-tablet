import math
import numpy as np
import scipy.signal
import scipy.fft
import time
import random

# Input parameters
frequency = 10000
adc_max = 1650
adc_resolution = 16384
amplitude = 1320
sample_rate = 30000
samples = 512

# Constants
end_time = samples / sample_rate
timel = np.arange(0, end_time, 1 / sample_rate)
theta = 2 * np.random.random() * math.pi
omega = 2 * math.pi * frequency

print("Input parameters:")
print(f"\tsignal frequency: {frequency}")
print(f"\tsignal amplitude: {amplitude}")
print(f"\tADC sample rate: {sample_rate}")
print(f"\tADC samples: {samples}")
print(f"\tADC resolution: {adc_resolution}")
print(f"\tSampling time (sec): {end_time}")

# https://pdfs.semanticscholar.org/489d/3fa67ad82a13a0856030d013fba64aaf7451.pdf
def estimator_fast_4_point_better_3x_sr(frame):
    def A(i):
        a = float(frame[i + 0])
        b = float(frame[i + 1])
        c = float(frame[i + 2])
        d = float(frame[i + 3])
        m1 = a**4 + b**4 + c**4 + d**4 - 2 * (a**2 * b**2 + a**2 * c**2 + a**2 * d**2 + b**2 * c**2 + b**2 * d**2 + c**2 * d**2) + 8 * a * b * c * d
        m2 = a**3 * b * c * d + a * b**3 * c * d + a * b * c**3 * d + a * b * c * d**3 - (a**2 * b**2 * c**2 + a**2 * b**2 * d**2 + a**2 * c**2 * d**2 + b**2 * c**2 * d**2)
        return -2 * math.sqrt(m1 * m2) / m1
    r = 0
    n = 64
    for i in range(n):
        r += A(i)
    r /= n
    return r

# https://www.tandfonline.com/doi/abs/10.1080/00207217.2011.582452
def estimator_4_point(frame):
    def e(i):
        return frame[i + 1] - frame[i]
    def A(i):
        return math.sqrt((e(i + 1)**2 - e(i) * e(i + 2)) / (2 * (1 + (e(i) + e(i + 2)) / (2 * e(i + 1))) * (1 - (e(i) + e(i + 2)) / (2 * e(i + 1)))**2))
    return A(0)

# https://ieeexplore.ieee.org/document/5256302
def estimator_5_point_need_3x_sr(y):
    def G(i):
        return y[i]**2 + y[i + 1]**2 + y[i + 2]**2 - (y[i] * y[i + 1] + y[i] * y[i + 2] + y[i + 1] * y[i + 2])
    n = 3
    r = 0
    for i in range(n):
        r += G(i)
    return round(2 * math.sqrt(r / n) / 3)

def fft(wfdata):
    lflim = frequency - 1000
    uflim = frequency + 1000
    nfft = len(wfdata)
    sr = sample_rate

    window = scipy.signal.windows.flattop(nfft)
    wfwind = [wf * wd for wf, wd in zip(wfdata, window)]
    spectrum = np.fft.rfft(wfwind, nfft)
    fvals = np.fft.rfftfreq(nfft, 1 / sr)
    power = np.abs(spectrum)

    lindex = int(np.floor((len(fvals) / fvals[-1]) * lflim))
    uindex = int(np.ceil((len(fvals) / fvals[-1]) * uflim))
    pslice = power[lindex:uindex]
    fslice = fvals[lindex:uindex]

    f0x = fslice[np.argmax(pslice)]
    f0y = np.max(pslice)
    return f0y
    # https://www.dsprelated.com/showarticle/155.php
    return pslice[2] - 0.94247 * (pslice[1] - pslice[3]) + 0.44247 * (pslice[0] - pslice[4])

def all_maxes_mean(frame):
    result = []
    for i in range(1, len(frame) - 1):
        if frame[i - 1] <= frame[i] >= frame[i + 1]:
            result.append(frame[i])
        elif frame[i - 1] >= frame[i] <= frame[i + 1]:
            result.append(abs(frame[i]))
    return round(np.mean(result))

def all_3_maxes_mean(frame):
    result = []
    for i in range(1, len(frame) - 1):
        if frame[i - 1] <= frame[i] >= frame[i + 1]:
            result.append(frame[i - 1])
            result.append(frame[i])
            result.append(frame[i + 1])
        elif frame[i - 1] >= frame[i] <= frame[i + 1]:
            result.append(abs(frame[i - 1]))
            result.append(abs(frame[i]))
            result.append(abs(frame[i + 1]))
    return round(np.mean(result))

def parabola(frame):
    def getone(frame, i):
        x1 = i - 1
        x2 = i
        x3 = i + 1

        y1 = frame[x1]
        y2 = frame[x2]
        y3 = frame[x3]

        a = y2 - (y1 + y3) / 2
        b = (y3 - y1) / 4
        if math.isnan(a) or a == 0:
            return y2
        xExtr = b / a
        return y2 + b * xExtr

    result = []
    for i in range(1, len(frame) - 1):
        if frame[i - 1] <= frame[i] >= frame[i + 1]:
            result.append(getone(frame, i))
        elif frame[i - 1] >= frame[i] <= frame[i + 1]:
            result.append(abs(getone(frame, i)))
    return round(np.mean(result))

def sine_wave_fitting(frame):
    def sinfunc(t, A, w, p, c):
        return A * np.sin(w * t + p) + c

    guess = np.array([np.max(frame), 2. * np.pi * frequency, 0., 0])
    popt, pcov = scipy.optimize.curve_fit(sinfunc, timel, frame, p0=guess)
    A, w, p, c = popt
    return round(abs(A))

def sine_2_points(frame):
    def get_sine(y1, y2, t):
        Y = math.sqrt(y1**2 + y2**2 - 2 * y1 * y2 * math.cos(t)) / abs(math.sin(t))
        #phi = 2 * math.pi - 1 / math.tan((y2 * math.sin(omega * t1) - y1 * math.sin(omega * t2)) / (y2 * math.cos(omega * t1) - y1 * math.cos(omega * t2)))
        return Y

    result = []
    for i in range(samples - 1):
        result.append(get_sine(frame[i], frame[i + 1], omega / sample_rate))
    return round(np.mean(result))

def goertzel(frame):
    window = scipy.signal.windows.flattop(len(frame))
    frame = [wf * wd for wf, wd in zip(frame, window)]

    k = int(0.5 + ((len(frame) * frequency) / sample_rate))
    omega = (2 * math.pi * k) / len(frame)
    sine = math.sin(omega)
    cosine = math.cos(omega)
    coeff = 2 * cosine
    q0 = 0
    q1 = 0
    q2 = 0

    for i in range(len(frame)):
        q0 = coeff * q1 - q2 + frame[i]
        q2 = q1
        q1 = q0

    return math.sqrt(q1**2 + q2**2 - q1 * q2 * coeff)

def testone(method, frame):
    start = time.time_ns()
    result = method(frame)
    end = time.time_ns()
    return result, (end - start) / 1000

def gen_frame(phase_random, phase_noise, adc_values, adc_noise, amplitude):
    global theta

    if phase_random:
        theta = 2 * np.random.random() * math.pi
    _theta = theta + random.uniform(-phase_noise, phase_noise) / 180 * math.pi
    frame = amplitude * np.sin(2 * np.pi * frequency * timel + _theta)
    if adc_values:
        frame = np.int32(frame / adc_max * adc_resolution)
        if adc_noise != 0:
            frame += np.int32((2 + 2 * adc_noise) * (np.random.random(size=len(timel)) - 0.5))
    return frame

def testiter(method, iters, phase_random, phase_noise, adc_values, adc_noise, amplitude):
    tt = 5000
    results = []
    for i in range(iters):
        result, time = testone(method, gen_frame(phase_random, phase_noise, adc_values, adc_noise, amplitude))
        if adc_values:
            result = result / adc_resolution * adc_max
        results.append(result)
        if time < tt:
            tt = time
    _min = np.min(results)
    _max = np.max(results)
    _mean = np.mean(results)
    _std = np.std(results)

    return _min, _max, _mean, _std, tt

dist = 1

def test(method, iters, phase_random, phase_noise, adc_values, adc_noise):
    global dist

    _min, _max, _mean, _std, _tt = testiter(method, iters, phase_random, phase_noise, adc_values, adc_noise, amplitude)
    _min2, _max2, _mean2, _std2, _tt2 = testiter(method, iters, phase_random, phase_noise, adc_values, adc_noise, amplitude + 1)

    ret = '\t'
    if phase_random:
        ret += 'Random phase'
    else:
        ret += 'Fixed phase'
        if phase_noise:
            ret += f' + noise {phase_noise}'
    ret += ', '
    if adc_values:
        ret += 'ADC values'
        if adc_noise:
            ret += f' + noise {adc_noise}'
    else:
        ret += 'exact values'
    ret += ': '

    dist = _min2 - _max

    return ret + f'min={_min:.2f},{_min2:.2f}, max={_max:.2f},{_max2:.2f}, mean={_mean:.2f},{_mean2:.2f}, std={_std:.2f}, max-min={_max - _min:.2f}, error={amplitude - _mean:.2f}, time={_tt}, distinction={dist:.2f},{dist > 0}'

def iterate_phases(method, iters, adc_values, adc_noise):
    print(test(method, iters, False, 0, adc_values, adc_noise))
    print(test(method, iters, False, 1, adc_values, adc_noise))
    print(test(method, iters, False, 2, adc_values, adc_noise))
    print(test(method, iters, False, 10, adc_values, adc_noise))
    print(test(method, iters, False, 45, adc_values, adc_noise))
    print(test(method, iters, True, 0, adc_values, adc_noise))

def iterate_adc(method, iters):
    try:
        iterate_phases(method, iters, False, 0)
        for i in range(1000):
            iterate_phases(method, iters, True, i)
            if dist <= 0:
                break
    except Exception:
        print('\tException')

methods = {
    'All maxes mean': all_maxes_mean,
    'All 3 maxes mean': all_3_maxes_mean,
    'Parabola': parabola,
    'Sine-wave fitting': sine_wave_fitting,
    'Sine 2 points': sine_2_points,
    'Goertzel': goertzel,
    'Fast 4 points': estimator_fast_4_point_better_3x_sr,
    '4 points': estimator_4_point,
    'FFT': fft,
}

for name, method in methods.items():
    print(name, ':')
    iterate_adc(method, 1000)
