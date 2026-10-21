# Copyright 2026 D-Wave
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import plotly.graph_objects as go

from demo_configs import (
    ENERGY_GAP_COLOR,
    EXCITED_STATE_COLORS,
    MODEL_COLORS,
    SHOW_PARALLEL_SAMPLES,
)
from src.demo_enums import ModelType, ProblemType
from src.model_loading import ProblemModel
from src.qpu_resources import get_schedule_filepath
from src.values import (
    DEFAULT_PLOT_WIDTH,
    ENERGY_TIME_RESCALING,
    MAX_PLOT_WIDTH,
    SAMPLE_POINT_SIZE,
    Y_AXIS_RESCALING,
)

LINE_WIDTH = 2
MARKER_OUTLINE = dict(color="white", width=1)
ERROR_BAR = dict(thickness=3, width=8)  # Stem thickness and cap width in pixels.

PLOT_LAYOUT = dict(
    template="plotly_white",
    margin=dict(l=60, r=10, t=10, b=50),
    legend=dict(
        xanchor="left",
        x=0.01,
        yanchor="bottom",
        y=0.01,
        bgcolor="rgba(255, 255, 255, 0.7)",
    ),
)



def make_scatter(x, y, type, id, other_args=None):

    if type == "Theory":
        scatter = go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name=f"{id.label} Theory",
            line=dict(color=MODEL_COLORS[id], width=LINE_WIDTH),
        )

    elif type == "Sample": #TODO finish and apply later
        model = other_args["model"]
        y_errs = other_args["err"]
        sample_point_size = other_args["point_size"]
        point_label = other_args["label"]
        go.Scatter(
                    x=x,
                    y=y,
                    error_y=dict(type="data", array=y_errs, visible=True, **ERROR_BAR),
                    mode="markers",
                    marker=dict(
                        symbol="diamond-wide",
                        size=sample_point_size,
                        color=MODEL_COLORS[model],
                        line=MARKER_OUTLINE,
                    ),
                    name=f"{model.label} {point_label}",
                )
    elif type == "Parallel":
        pass
    elif type == "Spectrum":
        series_labels = ["1st", "2nd", "3rd", "4th", "5th"]
        scatter = go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name=f"{series_labels[id]} Excited State",
        )
    return scatter
    

def get_schedule(qpu_name: str="Advantage2_system1") -> np.ndarray:
    schedule_filepath = get_schedule_filepath(qpu_name)
    with open(schedule_filepath, "r") as f:
        schedule_data = np.loadtxt(f, delimiter=",", skiprows=1)
    return schedule_data

def rescale_spectrum(
        spectrum_data: np.ndarray,
        qpu_schedule: np.ndarray,
        reference_schedule: np.ndarray,  
        energy_rescaling: float = 1.0
    ) -> tuple[np.ndarray, np.ndarray]:
    column_length = len(spectrum_data[:, 0])
    A_ref = reference_schedule[:column_length, 1]
    B_ref = reference_schedule[:column_length, 2]
    S_ref = reference_schedule[:column_length, 0]

    A = qpu_schedule[:column_length, 1]
    B = qpu_schedule[:column_length, 2]

    s_vals = np.interp((B*energy_rescaling)/A, B_ref/A_ref, S_ref)

    # TODO: Jack, this was the calculation I originally used, based on what you
    # wrote on the whiteboard. It's the version that produced the plots I sent you.
    # Below it is a more recent attempt to directly copy the relevant part of your
    # code file. It seems to produce much worse plots, but might be closer
    # and only need minor adjustments.

    #E_rescale = np.interp((B*energy_rescaling)/A, B_ref/A_ref, A_ref)

    # I left off the factor of 2 because I'm directly pulling the annealing
    # schedule, which doesn't include it.
    E_rescale = np.interp(s_vals, S_ref, A_ref/A)

    multiplier = np.zeros((column_length, column_length))
    np.fill_diagonal(multiplier, E_rescale/A_ref)
    
    new_spectrum_data = np.matmul(multiplier, spectrum_data)
    return s_vals, new_spectrum_data


def calculate_tau(spectra: np.ndarray, c: np.ndarray) -> float:
    """Calculate the Landau-Zener excitation form under a linearized approximation

    gap(s) ~ root(e(s)^2 + Delta(s)^2) = root(Dm^2 + (s-s*)^2 v^2)
    gap^2 = (Dm^2 + s*^2 v^2) - (2 s* v^2)s + v^2 s^2
    # gap2 is quadratic model in s, fit coefficients:
    Pex = exp(- tf /t_ad)
    t_ad = 2 hbar v / ( pi * Dm^2)

    equivalent statement: v^2 = 1/2 d^2/ds^2 gap^2(s) at s=s*

    Args:
        spectra: sequential eigenvalues as row vectors, for given c.
        c: c values at which spectra is evaluated

    Returns:
        * adiabatic time scale t_ad. P(diabatic) = exp(- tf /t_ad)
    """
    gap2 = (spectra[:, 4] - spectra[:, 3]) ** 2
    idx = int(np.argmin(gap2))
    if not (idx > 0 and idx < spectra.shape[0] - 1):
        warnings.warn("The gap is minimized at the boundary, fit aborted")
        return -1, -1, -1, -1
    p = np.polyfit(x=c[idx - 1 : idx + 2], y=gap2[idx - 1 : idx + 2], deg=2)
    v = np.sqrt(p[0])
    cstar = -p[1] / p[0] / 2
    gap = np.sqrt(p[2] - p[0] * cstar**2)
    t_ad = v / (
        np.pi**2 * gap**2
    )  # gap and v have units h. Need to multiply by dt_by_ds
    return t_ad


def generate_plots(
    job_results: dict|None,
    qpu_name: str,
    model_type: ModelType,
    problem_type: ProblemType = ProblemType.DEFAULT,
    show_parallel_samples: bool = SHOW_PARALLEL_SAMPLES,
) -> tuple[go.Figure, go.Figure]:
    """
    """

    #=======================================
    # START HERE

    qpu_schedule = get_schedule(qpu_name)
    ref_schedule = get_schedule()
    
    results_fig = go.Figure()
    spectrum_fig = go.Figure()

    spectrum_to_plot = None
    spectrum_s_vals = None

    for model in ModelType:
        theory_xs, theory_ys = ProblemModel.load_theory_data_from_file(problem_type, model)
        base_spectrum = ProblemModel.load_spectrum_scatter_from_file(model, problem_type)
        s_rescaled, spectrum_rescaled = rescale_spectrum(base_spectrum, qpu_schedule, ref_schedule)

        if model == model_type:
            spectrum_to_plot = spectrum_rescaled
            spectrum_s_vals = s_rescaled
            
        tau_qpu = calculate_tau(spectrum_rescaled, s_rescaled)
        tau_ref = calculate_tau(base_spectrum, ref_schedule[:,3])
        x_multiplier =  tau_ref / tau_qpu

        print(f"Model: {model.label}, QPU{qpu_name}, tau_qpu: {tau_qpu}, tau_ref: {tau_ref}, mult: {x_multiplier}")

        theory_xs = [x * x_multiplier for x in theory_xs[:-1]]
        theory_ys = theory_ys[:-1]
        results_fig.add_trace(
            make_scatter(theory_xs, theory_ys, "Theory", model)
        )

    egs_column = spectrum_to_plot[:, 3]
    excited_columns = [spectrum_to_plot[:, i] for i in range(4,9)]

    for idx, column in enumerate(excited_columns):
        spectrum_fig.add_trace(
            make_scatter(spectrum_s_vals, column - egs_column, "Spectrum", idx)
        )

    #========================================================
    # After this point is unlikely to be relevant

    spectrum_fig.update_yaxes(type="log", range=[-1, 2])
    spectrum_fig.update_xaxes(range=[0,0.545])

    spectrum_fig.update_layout(
        xaxis_title="Anneal Progress", yaxis_title="Excitation Energy (GHz)", **PLOT_LAYOUT
    )

    x_range = DEFAULT_PLOT_WIDTH
    y_axis_min = -1.0

    for model in ModelType:
        if len(job_results[model.name]) > 0:
            anneal_times = []
            egs_probs = []
            y_errs = []
            parallel_times = []
            parallel_probs = []
            sample_point_size = SAMPLE_POINT_SIZE
            sample_point_label = "Samples"

            for ta, stat_set in job_results[model.name].items():
                anneal_times.append(ta)
                egs_probs.append(np.mean(stat_set))
                y_errs.append(np.std(stat_set))

                if show_parallel_samples:
                    parallel_times.extend([ta] * len(stat_set))
                    parallel_probs.extend(stat_set)

            if show_parallel_samples:
                sample_point_size += 4
                y_errs = []
                results_fig.add_trace(
                    go.Scatter(
                        x=parallel_times,
                        y=parallel_probs,
                        mode="markers",
                        marker=dict(
                            symbol="circle",
                            size=max(sample_point_size - 7, 1),
                            color=MODEL_COLORS[model],
                            line=MARKER_OUTLINE,
                        ),
                        name=f"{model.label} {sample_point_label}",
                    )
                )
                sample_point_label = "Sample Mean"

            results_fig.add_trace(
                go.Scatter(
                    x=anneal_times,
                    y=egs_probs,
                    error_y=dict(type="data", array=y_errs, visible=True, **ERROR_BAR),
                    mode="markers",
                    marker=dict(
                        symbol="diamond-wide",
                        size=sample_point_size,
                        color=MODEL_COLORS[model],
                        line=MARKER_OUTLINE,
                    ),
                    name=f"{model.label} {sample_point_label}",
                )
            )

            if max(anneal_times) > DEFAULT_PLOT_WIDTH - 1:
                x_range = MAX_PLOT_WIDTH

    

    results_fig.update_layout(
        xaxis_title="Annealing Time (ns)", yaxis_title="Excited State Probability", **PLOT_LAYOUT
    )

    results_fig.update_xaxes(range=[0, x_range])
    results_fig.update_yaxes(type="log", range=[y_axis_min, 0])

    return results_fig, spectrum_fig
