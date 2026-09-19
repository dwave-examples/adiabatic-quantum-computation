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

from src.demo_enums import JobState

class QPUResources(NamedTuple):
    client: Client | None
    solvers: dict
    init_job_status: JobState


@lru_cache(maxsize=1)
def get_qpus() -> QPUResources: 
    """Get QPU resources, including client and solvers. Results are cached."""
    try:
        client = Client.from_config(client="qpu")
        solver_opts = client.get_solvers()
        print(f"Available solvers: {[s.name for s in solver_opts]}")
        solvers = {s.name: s for s in solver_opts}

        if not solvers:
            raise Exception

        return QPUResources(client=client, solvers=solvers, init_job_status=JobState.READY)

    except Exception:
        return QPUResources(client=None, solvers={}, init_job_status=JobState.NO_SOLVER)


def get_qpu_ids() -> dict:
    available_qpus = get_qpus().solvers
    return {qpu.name: qpu.graph_id for qpu in available_qpus.values()}
