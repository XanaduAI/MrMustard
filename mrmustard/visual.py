from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn

class Progressbar:
    def __init__(self, max_steps: int):
        self.taskID = None
        if max_steps == 0:
            self.bar = Progress(TextColumn("Step {task.completed}/∞"),
                                BarColumn(),
                                TextColumn("Loss = {task.fields[loss]:.5f}"))
        else:
            self.bar = Progress(TextColumn("Step {task.completed}/{task.total} | {task.fields[speed]:.1f} it/s"),
                                BarColumn(),
                                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                                TextColumn("Loss = {task.fields[loss]:.5f} | ⏳ "),
                                TimeRemainingColumn())
        self.taskID = self.bar.add_task(description="Optimizing...", start=max_steps > 0, speed=0.0, total=max_steps, loss=1.0, refresh=True)

    def step(self, loss):
        speed = self.bar.tasks[0].speed or 0.0
        self.bar.update(self.taskID, advance=1, refresh=True, speed = speed, loss=loss)

    def __enter__(self):
        return self.bar.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.bar.__exit__(exc_type, exc_val, exc_tb)