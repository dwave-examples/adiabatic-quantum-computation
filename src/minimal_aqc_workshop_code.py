# Copyright 2025 D-Wave
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

"""
A repository of code to demonstrate simple Kibble-Zurek and Landau-Zener experiments
compatible with https://doi.org/10.1038/s41586-023-05867-2 (Figure 2C)
and https://doi.org/10.1038/s41567-022-01741-6 (Figure 2A)
"""

from tqdm import tqdm
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pickle

import minorminer.subgraph
from minorminer.utils.parallel_embeddings import find_multiple_embeddings
from dwave.preprocessing.composites import SpinReversalTransformComposite
import dwave.system
from dwave.samplers import TreeDecompositionSolver
from dwave_networkx import draw_parallel_embeddings


def get_model(model_idx=0):
    """ """
        # TO DO, plot spectrum by brute force with published schedules.
    with open("helper_files/Andrew220713800Fig2c.pkl", "rb") as f:
        X = pickle.load(f)
    Js = list(X.values())
    J = Js[model_idx]
    if model_idx == 0:
        label = "Small Gap"
    elif model_idx == 1:
        label = "Intermediate Gap"
    else:
        label = "Large Gap"
    nodes = sorted({n for e in J for n in e})
    h = {n: 0 for n in nodes}
    ss = TreeDecompositionSolver().sample_ising(h, J)
    EGS = ss.first.energy
    print(h, J, EGS)
    return h, J, EGS, label

def get_embedding(source_graph, target_graph, num_parallel=1):
    
    if num_parallel > 1:
        max_num_emb = max(1, num_parallel)  
        embeddings = find_multiple_embeddings(
            source_graph,
            target_graph,
            max_num_emb=max_num_emb,
            timeout=30,
            embedder_kwargs=embedder_kwargs,
        )

    else:
        embedding = minorminer.subgraph.find_subgraph(
            source_graph, target_graph, **embedder_kwargs
        )

        embeddings = [embedding]

    return embeddings

def plot_figure_plotly(title, model_stats, tas_nanosec, fig_labels):
    fig = go.Figure()
    
    # Add trace for each data set
    for stat_set, label in zip(model_stats, fig_labels):
        fig.add_trace(go.Scatter(
            x=tas_nanosec,
            y=stat_set,
            mode='lines+markers',
            marker=dict(symbol='x', size=8),
            name=label
        ))
    
    # Update layout with title, axis labels, and log scale
    fig.update_layout(
        title="Landau-Zener: Exponential decay as function of the gap",
        xaxis_title="Annealing time (ns)",
        yaxis_title="1 - Ground state probability",
        yaxis_type="log",
        hovermode='x unified',
        template='plotly_white'
    )
    
    # Save as HTML file
    html_filename = title.replace('.png', '.html')
    fig.write_html(html_filename)
    print(f"A scaling plot has been saved to {html_filename}.")
    
    # Display the figure
    fig.show()

def plot_figure(title, model_stats, tas_nanosec, fig_labels):

    plt.figure()
    for stat_set, label in zip(model_stats, fig_labels):
        plt.plot(tas_nanosec, stat_set, marker="x", label=label)
    plt.ylabel("1 - Ground state probability")
    plt.title("Landau-Zener: Exponential decay as function of the gap")
    plt.legend()
    plt.yscale("log")
    plt.xlabel("Annealing time (ns)")
    plt.savefig(title, bbox_inches="tight")

    print(f"A scaling plot has been saved to {title}.")
    plt.show()

def main(
    solver=None,
    tas_nanosec=(5, 7, 10, 14, 20),
    num_models=3,
    parallelize_embedding=False,
    use_srt=False,
):
    """Experiment with customizable parameters; see main function descriptions."""
    # Source graph connectivity:
    h, J, EGS, label = get_model(model_idx=0)  # For topology only
    num_vars = len(h)
    source_graph = nx.from_edgelist(J.keys())

    # Connect to solver and infer target connectivity:
    qpu = dwave.system.DWaveSampler(solver=solver)
    solver = qpu.solver.name
    qpu_kwargs = dict(
        num_reads=100,
        answer_mode="raw",
        fast_anneal=True,
        auto_scale=False,
        label="AQC2025 Workshop",
    )
    target_graph = qpu.to_networkx_graph()

    sampler = qpu

    # Apply a spin-reversal transform
    if use_srt == True:
        sampler = SpinReversalTransformComposite(sampler)

    num_parallel = 1
    if parallelize_embedding:
        num_parallel = max(1, len(qpu.nodelist) // num_vars)  # Aim for half-full:

    embeddings = get_embedding(source_graph, target_graph, num_parallel=num_parallel)


    if parallelize_embedding:
        sampler = dwave.system.ParallelEmbeddingComposite(
            sampler,
            embeddings=[{k: (v,) for k, v in emb.items()} for emb in embeddings],
        )
    else:
        sampler = dwave.system.FixedEmbeddingComposite(
            sampler, embedding={k: (v,) for k, v in embeddings[0].items()}
        )

    model_stats = []
    fig_labels = []

    for model_idx in range(num_models):
        h, J, EGS, label = get_model(model_idx=model_idx)  # For topology
        fig_labels.append(label)
        stats = []
        print(
            f"{len(tas_nanosec)}",
            " sequential QPU programmings (varying in annealing time)",
        )
        for ta_nanosec in tqdm(tas_nanosec):
            qpu_kwargs["annealing_time"] = ta_nanosec / 1000
            ss = sampler.sample_ising(h, J, **qpu_kwargs)
            stats.append(
                1 - np.mean(ss.record.energy <= EGS + 1e-8)
            )  # 1 - Prob(Ground state)

        model_stats.append(stats)

    print("Plotting results...")
    plot_title = f"Landau-Zener Scaling Plot for {solver}"
    plot_figure_plotly(plot_title, model_stats, tas_nanosec, fig_labels)

    print("Exiting...")

if __name__ == "__main__":

    tas_nanosec = [5, 7, 10, 14, 20]  # Larger values can show thermal deviation


    main()
