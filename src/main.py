import argparse
from pathlib import Path
from src.experiment import Experiment, create_agent
from src.configs import UserConfig, AgentConfig, TaskConfig, WorkspaceConfig, ExperimentsConfig
from src.user import User
from src.workspace.workspace import Workspace

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--user_config', type=Path, default='user.json')
    parser.add_argument(
        '--agent_config', type=Path, default='agent.json')
    parser.add_argument(
        '--task_config', type=Path, default='task.json')
    parser.add_argument(
        '--workspace_config', type=Path, default='workspace.json')
    parser.add_argument(
        '--experiments_config', type=Path, default=None)
    return parser

def load_json(path):
    import json
    with open(path) as f:
        return json.load(f)
    
def run_single(args):
    user_config = UserConfig(**load_json(args.user_config))
    agent_config = AgentConfig(**load_json(args.agent_config))
    task_config = TaskConfig(**load_json(args.task_config))
    workspace_config = WorkspaceConfig(**load_json(args.workspace_config))
    workspace = Workspace(workspace_config, user_config, agent_config, task_config)
    user = User(user_config, task_config, workspace)
    agent = create_agent(agent_config, user.task, workspace)
    e = Experiment(user, agent, workspace)
    e.run()

def run_multi(workspace_config, task_config, user_config, agent_configs):
    exp_config = ExperimentsConfig(workspace_config, task_config, user_config, agent_configs)

    exp_iters = {}
    for k, agent_config in exp_config.agent_configs.items():
        workspace = Workspace(exp_config.workspace_config, exp_config.user_config, agent_config, exp_config.task_config)
        user = User(exp_config.user_config, exp_config.task_config, workspace)
        agent = create_agent(agent_config, user.task, workspace)
        exp_iters[k] = Experiment(user, agent, workspace).iter()
    done = set()
    while len(done) < len(exp_iters):
        for k, v in exp_iters.items():
            if k in done:
                continue
            if not next(v, None):
                print(f'Agent {k} is done')
                done.add(k)

def main(args):
    if not args.experiments_config:
        run_single(args)
    else:
        run_multi(**load_json(args.experiments_config))

if __name__ == "__main__":
    args = args_parser().parse_args()
    main(args)