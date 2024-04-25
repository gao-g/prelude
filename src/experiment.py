from src.agent.cipher import Cipher1Agent, CipherNAgent
from src.agent.icl_edit import IclEditAgent
from src.workspace.workspace import Workspace
from src.agent.abstract_agent import Agent
from src.agent.no_learning import NoLearningAgent
from src.agent.continual import ContinualLPIAgent
from src.agent.explore_exploit import ExploreExploitLPIAgent
from src.agent.oracle_preference import OraclePreferenceAgent
from src.user import User
from src.environment import Environment
from src.utils.logs import Logs
import logging


def create_agent(agent_config, task, workspace):
    if agent_config.agent == 'no-learning':
        return NoLearningAgent(agent_config, task, workspace, **agent_config.params)
    elif agent_config.agent == 'continual':
        return ContinualLPIAgent(agent_config, task, workspace, **agent_config.params)
    elif agent_config.agent == 'explore-exploit':
        return ExploreExploitLPIAgent(agent_config, task, workspace, **agent_config.params)
    elif agent_config.agent == 'oracle-preference':
        return OraclePreferenceAgent(agent_config, task, workspace, **agent_config.params)
    elif agent_config.agent == 'cipher-1':
        return Cipher1Agent(agent_config, task, workspace, **agent_config.params)
    elif agent_config.agent == 'cipher-n':
        return CipherNAgent(agent_config, task, workspace, **agent_config.params)
    elif agent_config.agent == 'icl-edit':
        return IclEditAgent(agent_config, task, workspace, **agent_config.params)
    else:
        raise ValueError(f'Unknown agent {agent_config.agent}')


class Experiment:
    def __init__(self, user: User, agent: Agent, workspace: Workspace):
        self.workspace = workspace
        logging.basicConfig(format='[%(levelname)s] %(asctime)s - %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            level=self.workspace.log_level)
        self.agent = agent
        self.environment = Environment(user)

    def iter(self):
        for message in self.environment:
            if self.agent.can_cheat:
                self.agent.cheat(self.environment)    # look at user preference. Only for oracle baseline
            completion = self.agent.complete(message)
            correction = self.environment.user_edit(completion)   
            metrics = self.agent.learn(message, correction)
            self.workspace.log(dict(metrics, **self.environment.debug_metrics()))
            yield 1
        self.workspace.stop()

    def run(self):
        for _ in self.iter():
            ...

    @property
    def log_path(self):
        return self.workspace.log_path
    
    @property
    def logs(self):
        return Logs.load(self.log_path)