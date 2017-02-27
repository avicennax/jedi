# func_generator.py
# Random generation of functions of interest
from __future__ import division
import numpy as np

def flip_flop_generator(n=1, spikes=[[3,3]], t=1000, dt=.01, pulse_len=None):
    inputs = []
    outputs = []

    if n != len(spikes):
        raise ValueError("n must be same as len(spikes).")

    if pulse_len is None:
        pulse_len = max(((1/100)*t)/dt, 2)

    for spike_type in spikes:
        t_len = int(t/dt)
        inp = np.zeros(t_len)
        out = np.ones(t_len)

        pos_spikes = map(int, np.random.uniform(0, t/dt-1, spike_type[0]))
        neg_spikes = map(int, np.random.uniform(0, t/dt-1, spike_type[1]))

        inp[pos_spikes] = 1
        inp[neg_spikes] = -1

        inp = inp.tolist()
        out = out.tolist()

        last = 1
        pulse = 0
        pulse_counter = 0
        for i in range(t_len):
            if inp[i] == 1:
                if pulse == -1:
                    inp[i] = -1
                    pulse = -1
                    out[i] = -1
                    last = -1
                else:
                    if pulse_counter == 0:
                        pulse_counter = pulse_len
                    pulse = 1
                    out[i] = 1
                    last = 1
            if inp[i] == -1:
                if pulse == 1:
                    inp[i] = 1
                    pulse = 1
                    out[i] = 1
                    last = 1
                else:
                    if pulse_counter == 0:
                        pulse_counter = pulse_len
                    pulse = -1
                    out[i] = -1
                    last = -1
            else:
                if pulse_counter == 0:
                    pulse = 0
                    if last == 1:
                        out[i] = 1
                    else:
                        out[i] = -1
                else:
                    pulse_counter -= 1
                    if pulse == 1:
                        inp[i] = 1
                        out[i] = 1
                    else:
                        inp[i] = -1
                        out[i] = -1


        inp += [inp[-1], inp[-1]]
        out += [out[-1], out[-1]]

        inputs.append(inp)
        outputs.append(out)

    return inputs, outputs
