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

from __future__ import annotations

from functools import lru_cache
from typing import NamedTuple

from dwave.cloud import Client

from src.demo_enums import JobState, SolverType
from src.values import SCHEDULES_PATH
from demo_configs import ANNEALING_TIME_SERIES
annealing_times = [t/1000 for t in ANNEALING_TIME_SERIES]  # Convert milliseconds to seconds


class QPUResources(NamedTuple):
    client: Client | None
    solvers: dict
    init_job_status: JobState


@lru_cache(maxsize=len(SolverType))
def get_qpus() -> QPUResources: #TODO troubleshoot: not currently working
    """Get QPU resources, including client and solvers. Results are cached."""
    try:
        client = Client.from_config(client="qpu")
        solver_opts = client.get_solvers()
        print(f"Available solvers: {[s.name for s in solver_opts]}")
        solvers = {s.name: s for s in solver_opts}
        #solvers = {
            #qpu.name: qpu 
            #for qpu in client.get_solvers(fast_anneal_time_range__covers=annealing_times)
            #if qpu.name in [slvr.label for slvr in SolverType]
        #}

        if not solvers:
            raise Exception

        return QPUResources(client=client, solvers=solvers, init_job_status=JobState.READY)

    except Exception:
        return QPUResources(client=None, solvers={}, init_job_status=JobState.NO_SOLVER)

def get_qpu_ids() -> dict:
    available_qpus = get_qpus().solvers
    return {qpu.name: qpu.graph_id for qpu in available_qpus.values()}

def get_client() -> Client | None:
    """Get the client, if available. Returns None if no client is available."""
    return get_qpus().client


def get_solvers() -> dict:
    """Get available solvers, if any. Returns an empty dictionary if no solvers are available."""
    return get_qpus().solvers


def get_init_job_status() -> str:
    """Get initial job status. "READY" if solvers are available, "NO SOLVER" otherwise."""
    return get_qpus().init_job_status

def get_schedule(qpu_name: str):
    schedule = None
    for file in SCHEDULES_PATH.glob("*_schedule.csv"):  # get schedule files
        if qpu_name in file.name:
            if schedule is not None:
                raise ValueError(f"Multiple schedule files found for QPU {qpu_name}.")
            schedule = file #TODO extract actual schedule

    if schedule is None:
        schedule = SCHEDULES_PATH / "default_schedule.csv" #TODO improve
    return schedule