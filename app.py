import numpy as np
from scipy.integrate import solve_ivp
from scipy.io import loadmat
import plotly.graph_objects as go

# ============================================================
# LOAD PARAMETERS
# ============================================================

mat = loadmat("par.mat")

p = mat["par"][0, 0]

par = {}

for name in p.dtype.names:
    par[name] = float(p[name])

# MATLAB overrides
par["I0"] = 0.2
par["k"] = 10
par["je"] = 0.5

# ============================================================
# EXPERIMENTAL SETTINGS
# ============================================================

stim = 0.5
stimB = 2.6

# ============================================================
# SIGMOID
# ============================================================

def phi(a, F, theta):

    return (
        1 / (1 + np.exp(-a * (F - theta)))
        - 1 / (1 + np.exp(a * theta))
    )

# ============================================================
# BASELINE / UCN3 MODEL
# ============================================================

def model_ucn3(t, x):

    Gl, Gi, Ge, Dyn, NKB, v = x

    def urocortin(gamma):

        return (
            gamma * par["B"]
            + par["A"] * np.sin(2 * np.pi * par["f"] * t)
        )

    GABA_stim = (
        par["GABA_B"]
        + par["GABA_A"] * np.sin(2 * np.pi * par["f"] * t)
    )

    Fl = (
        (1 - par["beta2"]) * par["cll"] * Gl
        - (1 - par["beta1"]) * par["cil"] * Gi
        - (1 - par["beta1"]) * par["cel"] * Ge
        + urocortin(par["gamma1"])
        + par["base_l"]
    )

    Fi = (
        (1 - par["beta2"]) * par["cli"] * Gl
        - (1 - par["beta1"]) * par["cii"] * Gi
        - (1 - par["beta1"]) * par["cei"] * Ge
        + urocortin(par["gamma2"])
        + GABA_stim
        + par["base_i"]
    )

    Fe = (
        (1 - par["beta2"]) * par["cle"] * Gl
        - (1 - par["beta1"]) * par["cie"] * Gi
        - (1 - par["beta1"]) * par["cee"] * Ge
        + GABA_stim
        + par["base_e"]
    )

    dGl = par["dl"] * (
        -Gl + (1 - Gl) * phi(par["al"], Fl, par["thetal"])
    )

    dGi = par["di"] * (
        -Gi + (1 - Gi) * phi(par["ai"], Fi, par["thetai"])
    )

    dGe = par["de"] * (
        -Ge + (1 - Ge) * phi(par["ae"], Fe, par["thetae"])
    )

    dDyn = (
        par["k01"]
        + par["k1"] * v**2 / (v**2 + par["Kr1"]**2)
        - Dyn * par["d1"]
    )

    dNKB = (
        par["k02"]
        + par["k2"]
        * v**2
        / (v**2 + par["Kr2"]**2)
        * par["KD"]**2
        / (Dyn**2 + par["KD"]**2)
        - par["d2"] * NKB
    )

    Im = -par["je"] * Ge

    I = (
        par["I0"]
        + Im
        + par["pr"] * NKB**2 * v / (NKB**2 + par["KN"]**2)
    )

    dv = (
        par["v0"]
        * (
            1
            / (
                1
                + np.exp(
                    par["k"] * (-I + par["theta"])
                )
            )
        )
        - par["d3"] * v
    )

    return [dGl, dGi, dGe, dDyn, dNKB, dv]

# ============================================================
# STRESS MODEL
# ============================================================

def model_stress(t, x):

    Gl, Gi, Ge, Dyn, NKB, v = x

    def stress(gamma, base):

        return (
            gamma
            * (
                par["a"] * t * np.exp(-par["r2"] * t)
                + par["b"] * (1 - np.exp(-par["r1"] * t))
            )
            + base
        )

    Fl = (
        par["cll"] * Gl
        - par["cil"] * Gi
        - par["cel"] * Ge
        + stress(par["gamma1"], par["base_l"])
    )

    Fi = (
        par["cli"] * Gl
        - par["cii"] * Gi
        - par["cei"] * Ge
        + stress(par["gamma2"], par["base_i"])
    )

    Fe = (
        par["cle"] * Gl
        - par["cie"] * Gi
        - par["cee"] * Ge
    )

    dGl = par["dl"] * (
        -Gl + (1 - Gl) * phi(par["al"], Fl, par["thetal"])
    )

    dGi = par["di"] * (
        -Gi + (1 - Gi) * phi(par["ai"], Fi, par["thetai"])
    )

    dGe = par["de"] * (
        -Ge + (1 - Ge) * phi(par["ae"], Fe, par["thetae"])
    )

    dDyn = (
        par["k01"]
        + par["k1"] * v**2 / (v**2 + par["Kr1"]**2)
        - Dyn * par["d1"]
    )

    dNKB = (
        par["k02"]
        + par["k2"]
        * v**2
        / (v**2 + par["Kr2"]**2)
        * par["KD"]**2
        / (Dyn**2 + par["KD"]**2)
        - par["d2"] * NKB
    )

    Im = -par["je"] * Ge

    I = (
        par["I0"]
        + Im
        + par["pr"] * NKB**2 * v / (NKB**2 + par["KN"]**2)
    )

    dv = (
        par["v0"]
        * (
            1
            / (
                1
                + np.exp(
                    par["k"] * (-I + par["theta"])
                )
            )
        )
        - par["d3"] * v
    )

    return [dGl, dGi, dGe, dDyn, dNKB, dv]

# ============================================================
# SIMULATION SETTINGS
# ============================================================

dt = 0.02

baseline_duration = 120
perturb_duration = 120

t_baseline = np.arange(0, baseline_duration, dt)
t_perturb = np.arange(0, perturb_duration, dt)

y0 = np.zeros(6)

# ============================================================
# BASELINE
# ============================================================

par["A"] = 0

sol_baseline = solve_ivp(
    model_ucn3,
    [0, baseline_duration],
    y0,
    t_eval=t_baseline,
    rtol=1e-6,
    atol=1e-8
)

# ============================================================
# UCN3
# ============================================================

par["A"] = stim
par["B"] = stimB

sol_ucn3 = solve_ivp(
    model_ucn3,
    [0, perturb_duration],
    sol_baseline.y[:, -1],
    t_eval=t_perturb,
    rtol=1e-6,
    atol=1e-8
)

t_ucn3 = np.concatenate([
    t_baseline,
    baseline_duration + t_perturb
])

v_ucn3 = np.concatenate([
    sol_baseline.y[5] / 60,
    sol_ucn3.y[5] / 60
])

# ============================================================
# STRESS
# ============================================================

par["a"] = 8
par["r2"] = 3
par["b"] = 3.9
par["r1"] = 15

sol_stress = solve_ivp(
    model_stress,
    [0, perturb_duration],
    sol_baseline.y[:, -1],
    t_eval=t_perturb,
    rtol=1e-6,
    atol=1e-8
)

t_stress = np.concatenate([
    t_baseline,
    baseline_duration + t_perturb
])

v_stress = np.concatenate([
    sol_baseline.y[5] / 60,
    sol_stress.y[5] / 60
])

# ============================================================
# BASELINE ONLY
# ============================================================

v_baseline = sol_baseline.y[5] / 60

# ============================================================
# DOWNSAMPLE
# ============================================================

downsample = 3

t_baseline_ds = t_baseline[::downsample]
v_baseline_ds = v_baseline[::downsample]

t_ucn3_ds = t_ucn3[::downsample]
v_ucn3_ds = v_ucn3[::downsample]

t_stress_ds = t_stress[::downsample]
v_stress_ds = v_stress[::downsample]

# ============================================================
# FIGURE
# ============================================================

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=t_baseline_ds,
        y=v_baseline_ds,
        mode="lines",
        name="Baseline",
        line=dict(width=3),
        visible=True
    )
)

fig.add_trace(
    go.Scatter(
        x=t_ucn3_ds,
        y=v_ucn3_ds,
        mode="lines",
        name="UCN3",
        line=dict(width=3),
        visible=False
    )
)

fig.add_trace(
    go.Scatter(
        x=t_stress_ds,
        y=v_stress_ds,
        mode="lines",
        name="Stress",
        line=dict(width=3),
        visible=False
    )
)

# ============================================================
# LAYOUT
# ============================================================

fig.update_layout(

    template="plotly_white",

    autosize=True,

    height=450,

    margin=dict(
        l=55,
        r=20,
        t=120,
        b=50
    ),

    title=dict(
        text="MePD KNDy Dynamics",
        x=0.5,
        y=0.97,
        font=dict(size=22)
    ),

    xaxis=dict(
        title="Time [min]",
        range=[0,240],
        title_font=dict(size=16),
        tickfont=dict(size=13),
        showgrid=True
    ),

    yaxis=dict(
        title="KNDy firing rate [Hz]",
        range=[0, 50],
        title_font=dict(size=16),
        tickfont=dict(size=13),
        showgrid=True
    ),

    font=dict(size=14),

    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5,
        font=dict(size=13)
    ),

    updatemenus=[

        dict(

            type="buttons",

            direction="right",

            showactive=True,

            x=0.5,
            y=1.30,

            xanchor="center",
            yanchor="top",

            bgcolor="white",
            bordercolor="black",
            borderwidth=1,

            font=dict(size=14),

            pad=dict(
                r=6,
                t=6
            ),

            buttons=[

                dict(
                    label="Baseline",
                    method="update",
                    args=[{
                        "visible": [True, False, False]
                    }]
                ),

                dict(
                    label="UCN3",
                    method="update",
                    args=[{
                        "visible": [False, True, False]
                    }]
                ),

                dict(
                    label="Stress",
                    method="update",
                    args=[{
                        "visible": [False, False, True]
                    }]
                )

            ]
        )
    ],

    shapes=[

        dict(
            type="line",
            x0=120,
            x1=120,
            y0=0,
            y1=50,
            line=dict(
                color="gray",
                dash="dash",
                width=2
            )
        )
    ]
)

# ============================================================
# EXPORT
# ============================================================

fig.write_html(
    "index.html",
    include_plotlyjs="cdn",
    full_html=True,
    config={
        "responsive": True,
        "displayModeBar": False
    }
)

print("\nSaved: index.html\n")