# Epidemiological models' library.

import numpy as np

"""
Base class for epidemiological models. Each model is meant to return a time series from 0 to t,
with each parameter set before execution.
"""
class Model:
    def __init__(self):
        # Dictionary of input parameters.
        self.input_params = {}
        # Set of output parameters.
        self.output_params = {}

    # Returns a dictionary with a time series for each parameter in output_params.
    def eval_series(self, t):
        return {}

    # Returns a dictionary with a single prediction at time t for each parameter in output_params.
    def eval_last(self, t):
        out = self.eval_series(t)
        return {k: v[-1] for k, v in out.items()}

    # Returns a list of input parameter names.
    def get_input_keys(self):
        return list(self.input_params.keys())

    # Returns a list of output parameter names.
    def get_output_keys(self):
        return list(self.output_params)

    """
    Proxy function for scipy's curve_fit method. It should compute the same values of eval_last in a vectorized fashion
    (given a vector [t0, t1, ..., tn] it should return a matrix with a row for each prediction).
    Parameters are passed as arguments after the independent variable (t).
    """
    def f(self):
        return 0

"""
SIR model.
S0 = initial susceptibles
I0 = initial infected
R0 = initial removed
beta = infection rate
gamma = recovery rate (1 / time required for recovery)
"""
class SIR(Model):
    def __init__(self, s=1, i=0, r=0, b=0, g=0):
        self.input_params = {"S0": s, "I0": i, "R0": r, "beta": b, "gamma": g}
        self.output_params = {"S", "I", "R"}

    def f(self, t, s, i, r, beta, gamma):
        self.input_params = {"S0": s, "I0": i, "R0": r, "beta": beta, "gamma": gamma}
        tmp = self.eval_series(int(t[-1]))
        out = np.zeros((len(t), 3), dtype=np.float)
        for i in t:
            out[int(i), 0] = tmp["S"][int(i)]
            out[int(i), 1] = tmp["I"][int(i)]
            out[int(i), 2] = tmp["R"][int(i)]
        return out.flatten()

    def eval_series(self, t):
        # Using shorter variable names.
        S0 = self.input_params["S0"]
        I0 = self.input_params["I0"]
        R0 = self.input_params["R0"]
        b = self.input_params["beta"]
        g = self.input_params["gamma"]

        # Some sanity checking on the inputs:
        S0 = S0 if S0 > 0 else 0
        I0 = I0 if I0 > 0 else 0
        R0 = R0 if R0 > 0 else 0
        if b < 0:
            b = 0
        elif b > 1:
            b = 1

        if g < 0:
            g = 0
        elif g > 1:
            g = 1

        S = [S0]
        I = [I0]
        R = [R0]

        n = S0 + I0 + R0

        for i in range(t):
            SI = (b * S[-1] * I[-1]) / n
            IR = g * I[-1]
            S.append(S[-1] - SI)
            I.append(I[-1] + SI - IR)
            R.append(R[-1] + IR)

        return {"S": S, "I": I, "R": R}
"""
SIRD model.
S0 = initial susceptibles
I0 = initial infected
R0 = initial recovered
D0 = initial dead
beta = infection rate
gamma = recovery rate (1 / time required for recovery)
f = fatality rate
"""
class SIRD(Model):
    def __init__(self, s=1, i=0, r=0, d=0, b=0, g=0, f=0):
        self.input_params = {"S0": s, "I0": i, "R0": r, "D0": d, "beta": b, "gamma": g, "f": f}
        self.output_params = {"S", "I", "R", "D"}

    def f(self, t, s, i, r, d, beta, gamma, f):
        self.input_params = {"S0": s, "I0": i, "R0": r, "D0": d, "beta": beta, "gamma": gamma, "f": f}
        tmp = self.eval_series(int(t[-1]))
        out = np.zeros((len(t), 4), dtype=np.float)
        for i in t:
            out[int(i), 0] = tmp["S"][int(i)]
            out[int(i), 1] = tmp["I"][int(i)]
            out[int(i), 2] = tmp["R"][int(i)]
            out[int(i), 3] = tmp["D"][int(i)]
        return out.flatten()

    def eval_series(self, t):
        # Using shorter variable names.
        S0 = self.input_params["S0"]
        I0 = self.input_params["I0"]
        R0 = self.input_params["R0"]
        D0 = self.input_params["D0"]
        b = self.input_params["beta"]
        g = self.input_params["gamma"]
        f = self.input_params["f"]

        # Some sanity checking on the inputs:
        S0 = S0 if S0 > 0 else 0
        I0 = I0 if I0 > 0 else 0
        R0 = R0 if R0 > 0 else 0
        D0 = D0 if D0 > 0 else 0
        if b < 0:
            b = 0
        elif b > 1:
            b = 1

        if g < 0:
            g = 0
        elif g > 1:
            g = 1

        if f < 0:
            f = 0
        elif f > 1:
            f = 1

        S = [S0]
        I = [I0]
        R = [R0]
        D = [D0]



        for i in range(t):
            n = S[-1] + I[-1] + R[-1]
            SI = (b * S[-1] * I[-1]) / n
            IR = g * I[-1]
            ID = f * I[-1]
            S.append(S[-1] - SI)
            I.append(I[-1] + SI - IR - ID)
            R.append(R[-1] + IR)
            D.append(D[-1] + ID)

        return {"S": S, "I": I, "R": R, "D": D}

"""
SEIR model.
S0 = initial susceptibles
E0 = initial exposed
I0 = initial infected
R0 = initial removed
beta = infection rate
gamma = recovery rate (1 / time required for recovery)
sigma = incubation rate (1 / time required to become symptomatic)
c = mutation rate (1 / time required for a removed to become infected again). Can be set to 0 if we assume reinfection is unlikely.
"""
class SEIR(Model):
    def __init__(self, s=1, e=0, i=0, r=0, b=0, g=0, sig=0, c=0):
        self.input_params = {"S0": s, "E0": e, "I0": i, "R0": r, "beta": b, "gamma": g, "sigma": sig, "c": c}
        self.output_params = {"S", "E", "I", "R"}

    def f(self, t, s, e, i, r, beta, gamma, sigma, c):
        self.input_params = {"S0": s, "E0": e, "I0": i, "R0": r, "beta": beta, "gamma": gamma, "sigma": sigma, "c": c}
        tmp = self.eval_series(int(t[-1]))
        out = np.zeros((len(t), 4), dtype=np.float)
        for i in t:
            out[int(i), 0] = tmp["S"][int(i)]
            out[int(i), 1] = tmp["E"][int(i)]
            out[int(i), 2] = tmp["I"][int(i)]
            out[int(i), 3] = tmp["R"][int(i)]
        return out.flatten()

    def eval_series(self, t):
        # Using shorter variable names.
        S0 = self.input_params["S0"]
        E0 = self.input_params["E0"]
        I0 = self.input_params["I0"]
        R0 = self.input_params["R0"]
        b = self.input_params["beta"]
        g = self.input_params["gamma"]
        s = self.input_params["sigma"]
        c = self.input_params["c"]

        # Some sanity checking on the inputs:
        S0 = S0 if S0 > 0 else 0
        E0 = E0 if E0 > 0 else 0
        I0 = I0 if I0 > 0 else 0
        R0 = R0 if R0 > 0 else 0
        if b < 0:
            b = 0
        elif b > 1:
            b = 1

        if g < 0:
            g = 0
        elif g > 1:
            g = 1

        if s < 0:
            s = 0
        elif s > 1:
            s = 1

        if c < 0:
            c = 0
        elif c > 1:
            c = 1

        S = [S0]
        E = [E0]
        I = [I0]
        R = [R0]

        n = S0 + E0 + I0 + R0

        for i in range(t):
            SI = (b * S[-1] * I[-1]) / n
            EI = s * E[-1]
            IR = g * I[-1]
            RI = c * R[-1] * I[-1] / n
            S.append(S[-1] - SI)
            E.append(E[-1] + SI - EI)
            I.append(I[-1] + EI - IR + RI)
            R.append(R[-1] + IR - RI)

        return {"S": S, "E": E, "I": I, "R": R}


"""
SEIRD model.
S0 = initial susceptibles
E0 = initial exposed
I0 = initial infected
R0 = initial recovered
D0 = initial deceased
beta = infection rate
gamma = recovery rate (1 / time required for recovery)
sigma = incubation rate (1 / time required to become symptomatic)
c = mutation rate (1 / time required for a removed to become infected again). Can be set to 0 if we assume reinfection is unlikely.
f = fatality rate
"""
class SEIRD(Model):
    def __init__(self, s=1, e=0, i=0, r=0, d=0, b=0, g=0, sig=0, c=0, f=0):
        self.input_params = {"S0": s, "E0": e, "I0": i, "R0": r, "D0": d, "beta": b, "gamma": g, "sigma": sig, "c": c,
                             "f": f}
        self.output_params = {"S", "E", "I", "R", "D"}

    def f(self, t, s, e, i, r, d, beta, gamma, sigma, c, f):
        self.input_params = {"S0": s, "E0": e, "I0": i, "R0": r, "D0": d, "beta": beta, "gamma": gamma, "sigma": sigma, "c": c, "f": f}
        tmp = self.eval_series(int(t[-1]))
        out = np.zeros((len(t), 5), dtype=np.float)
        for i in t:
            out[int(i), 0] = tmp["S"][int(i)]
            out[int(i), 1] = tmp["E"][int(i)]
            out[int(i), 2] = tmp["I"][int(i)]
            out[int(i), 3] = tmp["R"][int(i)]
            out[int(i), 4] = tmp["D"][int(i)]
        return out.flatten()

    def eval_series(self, t):
        # Using shorter variable names.
        S0 = self.input_params["S0"]
        E0 = self.input_params["E0"]
        I0 = self.input_params["I0"]
        R0 = self.input_params["R0"]
        D0 = self.input_params["D0"]
        b = self.input_params["beta"]
        g = self.input_params["gamma"]
        s = self.input_params["sigma"]
        c = self.input_params["c"]
        f = self.input_params["f"]

        # Some sanity checking on the inputs:
        S0 = S0 if S0 > 0 else 0
        E0 = E0 if E0 > 0 else 0
        I0 = I0 if I0 > 0 else 0
        R0 = R0 if R0 > 0 else 0
        D0 = D0 if D0 > 0 else 0
        if b < 0:
            b = 0
        elif b > 1:
            b = 1

        if g < 0:
            g = 0
        elif g > 1:
            g = 1

        if s < 0:
            s = 0
        elif s > 1:
            s = 1

        if c < 0:
            c = 0
        elif c > 1:
            c = 1

        if f < 0:
            f = 0
        elif f > 1:
            f = 1

        S = [S0]
        E = [E0]
        I = [I0]
        R = [R0]
        D = [D0]

        for i in range(t):
            n = S[-1] + E[-1] + I[-1] + R[-1]  # Population is of live people only in a given moment.
            SI = (b * S[-1] * I[-1]) / n
            EI = s * E[-1]
            IR = g * I[-1]
            RI = c * R[-1] * I[-1] / n
            ID = f * I[-1]
            S.append(S[-1] - SI)
            E.append(E[-1] + SI - EI)
            I.append(I[-1] + EI - IR + RI - ID)
            R.append(R[-1] + IR - RI)
            D.append(D[-1] + ID)

        return {"S": S, "E": E, "I": I, "R": R, "D": D}

# SEIRD model with hidden E compartment. Useful for fitting or melding without knowing the values of E0/Et.
class HiddenSEIRD(SEIRD):
    def __init__(self, s=1, e=0, i=0, r=0, d=0, b=0, g=0, sig=0, c=0, f=0):
        self.input_params = {"S0": s, "E0": e, "I0": i, "R0": r, "D0": d, "beta": b, "gamma": g, "sigma": sig, "c": c,
                             "f": f}
        self.output_params = {"S", "I", "R", "D"}

    def f(self, t, s, e, i, r, d, beta, gamma, sigma, c, f):
        self.input_params = {"S0": s, "E0": e, "I0": i, "R0": r, "D0": d, "beta": beta, "gamma": gamma, "sigma": sigma, "c": c, "f": f}
        tmp = self.eval_series(int(t[-1]))
        out = np.zeros((len(t), 4), dtype=np.float)
        for i in t:
            out[int(i), 0] = tmp["S"][int(i)]
            out[int(i), 1] = tmp["I"][int(i)]
            out[int(i), 2] = tmp["R"][int(i)]
            out[int(i), 3] = tmp["D"][int(i)]
        return out.flatten()

    def eval_series(self, t):
        seird = super(HiddenSEIRD, self).eval_series(t)
        seird.pop('E', None)
        return seird