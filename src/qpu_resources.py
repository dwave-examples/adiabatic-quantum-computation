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

from dwave.cloud import Client

def get_qpu_ids(print_available: bool = False) -> dict:
    """Get the available QPU names mapped to their graph ids."""
    client = Client.from_config(client="qpu")
    solver_opts = client.get_solvers()
    solver_list = sorted({qpu.name for qpu in solver_opts})
    if print_available:
        print(f"Available solvers: {solver_list}")
    return solver_list
