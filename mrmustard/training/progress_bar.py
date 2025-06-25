# Copyright 2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A module containing classes and methods for progress bars."""

from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn

from mrmustard import settings


class ProgressBar:
    "A spiffy loading bar to display the progress during an optimization."

    def __init__(self, max_steps: int):
        self.taskID = None
        if max_steps == 0:
            self.bar = Progress(
                TextColumn("Step {task.completed}/∞"),
                BarColumn(),
                TextColumn("Cost = {task.fields[loss]:.5f}"),
            )
        else:
            self.bar = Progress(
                TextColumn("Step {task.completed}/{task.total} | {task.fields[speed]:.1f} it/s"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("Cost = {task.fields[loss]:.5f} | ⏳ "),
                TimeRemainingColumn(),
            )
        self.taskID = self.bar.add_task(
            description="Optimizing...",
            start=max_steps > 0,
            speed=0.0,
            total=max_steps,
            loss=1.0,
            refresh=True,
            visible=settings.PROGRESSBAR,
        )

    def step(self, loss):
        """Update bar step and the loss information associated with it."""
        speed = self.bar.tasks[0].speed or 0.0
        self.bar.update(self.taskID, advance=1, refresh=True, speed=speed, loss=loss)

    def __enter__(self):
        return self.bar.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.bar.__exit__(exc_type, exc_val, exc_tb)
