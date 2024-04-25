from src.workspace.abstract_sink import Sink

class WandbSink(Sink):
    def __init__(self, project, params, src_path):
        import wandb
        self.run = wandb.init(project=project, config=params)
        self.run.log_code(src_path)

    def log(self, metrics):
        self.run.log(metrics)

    def log_artifacts(self, name, type, files):
        import wandb
        artifact = wandb.Artifact(name, type=type)
        [artifact.add_file(f) for f in files]
        self.run.log_artifact(artifact)

    def stop(self):
        self.run.finish()