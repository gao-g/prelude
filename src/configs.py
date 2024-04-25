from pathlib import Path
from git import Repo, InvalidGitRepositoryError

class UserConfig:
    def __init__(self, model='gpt-4'):
        """
        model: language model to use for user. possible values: 'gpt-4', 'gpt-35-turbo-instruct', 'gpt-35-turbo', 'dummy'
        """
        self.model = model

    def __repr__(self):
        return f'{self.model}'


class AgentConfig:
    def __init__(self, model='gpt-35-turbo-instruct', agent='no-learning', **params):
        """
        model: language model to use for agent. possible values: 'gpt-4', 'gpt-35-turbo-instruct', 'gpt-35-turbo', 'dummy'
        agent: learning method to use. possible values: 'no-learning', 'continual', 'explore-exploit', 'icl', 'oracle-preference'
        params: agent-specific parameters
        """
        self.model = model
        self.agent = agent
        self.params = params

    def __repr__(self):
        return f'{self.model}.{self.agent}.{".".join([f"{k}-{v}" for k, v in self.params.items()])}'


class TaskConfig:
    def __init__(self, task='summarization', cost='L-distance', num_train_ex=10, seed=None, datasets=None):
        """
        task: task to use. possible values: 'summarization'
        cost: cost function to use. possible values: 'L-distance'
        num_train_ex: number of examples to use for training
        seed: random seed. Undeterministic if None
        """
        self.task = task
        self.cost = cost
        self.num_train_ex = num_train_ex
        self.seed = seed
        self.datasets = datasets

        self.trans_table = str.maketrans({
            '<': '.LT',
            '>': '.GT',
            ':': '.CLN',
            '"': '.QT',
            '/': '.SL',
            '\\': '.BSL',
            '|': '.PP',
            '?': '.Q',
            '*': '.WC'
        })

    def __repr__(self):
        return f'{self.task}.{"-".join(self.datasets).translate(self.trans_table) if self.datasets else "All"}.{self.cost}.{self.num_train_ex}.{self.seed}'


class WorkspaceConfig:
    def __init__(self, sink='dummy', project='test', description="", log_filename=None, log_folder='outputs', log_level='WARNING', src_path='.'): 
        """
        sink: where to save logs. possible values: 'dummy', 'wandb'
        project: name of the project
        description: description of the experiment
        log_filename: name of the log file - generated automatically based on hyperparameters if None
        log_folder: folder to save logs
        log_level: logging level. possible values: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
        src_path: path to the root of github repo
        """
        self.sink = sink
        self.project = project
        self.description = description
        self.log_filename = log_filename
        self.log_folder = Path(log_folder)
        self.log_level = log_level
        self.src_path = src_path
        repo = GitRepo(Path(src_path))
        self.is_dirty = repo.is_dirty
        self.commit_id = repo.commit_id


class ExperimentsConfig:
    def __init__(self, workspace_config, task_config, user_config, agent_configs):
        self.workspace_config = WorkspaceConfig(**workspace_config)
        self.task_config = TaskConfig(**task_config)
        self.user_config = UserConfig(**user_config)
        self.agent_configs = {k: AgentConfig(**v) for k, v in agent_configs.items()}


class GitRepo:
    def __init__(self, path):
        self.dirty = set()
        self.commit_id = None
        try:
            repo = Repo(path)
            self.commit_id = repo.head.commit.hexsha
            self.dirty = [item.a_path for item in repo.index.diff(None) if item.a_path.startswith('src')]
        except InvalidGitRepositoryError as e:
            print('Not a git repo')

    @property
    def is_dirty(self):
        return any(self.dirty)