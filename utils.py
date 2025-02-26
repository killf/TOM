import torch
import numpy as np
import time


class Counter:
    def __init__(self):
        self.data = dict()

    def append(self, **kwargs):
        for key, value in kwargs.items():
            if key not in self.data:
                self.data[key] = []

            if isinstance(value, torch.Tensor):
                value = value.cpu().numpy()

            self.data[key].append(value)

    def __getattr__(self, key):
        if key not in self.data:
            return 0
        return np.mean(self.data[key])


class Timer:
    """ Simple timer class for measuring time consuming """

    def __init__(self, auto_start=True):
        self._start_time = 0.0
        self._end_time = 0.0
        self._elapsed_time = 0.0
        self._is_running = False
        
        if auto_start:
            self.start()

    def start(self):
        self._is_running = True
        self._start_time = time.time()

    def restart(self):
        self.start()

    def stop(self):
        self._is_running = False
        self._end_time = time.time()

    def elapsed_time(self):
        self._end_time = time.time()
        self._elapsed_time = self._end_time - self._start_time
        if not self.is_running:
            return 0.0

        return self._elapsed_time

    @property
    def is_running(self):
        return self._is_running


def calculate_eta(remaining_step, speed):
    if remaining_step < 0:
        remaining_step = 0
    remaining_time = int(remaining_step * speed)
    result = "{:0>2}:{:0>2}:{:0>2}"
    arr = []
    for i in range(2, -1, -1):
        arr.append(int(remaining_time / 60**i))
        remaining_time %= 60**i
    return result.format(*arr)