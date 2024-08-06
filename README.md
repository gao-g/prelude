# PRELUDE
Code for Aligning LLM Agents by Learning Latent Preference from User Edits

## Table of Contents
- [Installation](#installation)
- [Implementation of PRELUDE Framework](#implemetaion-of-prelude-framework)
- [Reproduce Our Experiments](#reproduce-our-experiments)
- [Implement Your Own Agents](#implement-your-own-agent)

## Installation
1. This project is developed in Python 3.6. Using Conda to set up a virtual environment is recommended.

2. Install the required dependencies. 
    ```
    pip install -r requirements.txt
    ```
    
3. Install PyTorch from http://pytorch.org/.


## Implementation of PRELUDE Framework
PRELUDE implementation contains the follwoing main concepts `task`, `user`, and `agent`.
### Task
Task is the class encapsulating the following:
1) Access to dataset which is sequence of the $(x_t, f^\star_t)$ pairs of (context, true user preference pairs)
2) Main task prompt (Prompts to generate $y_t$ given $x_t$ and optionally $f_t$):
```
def get_task_prompt(self, input: str, preference: Optional[str] = None) -> str:
    ...
```
3) User evaluation prompts (Prompts to generate $y'_t$):
```
def get_edit_prompts(self, input: str, output: str, preference: str) -> Tuple[str, str]:
    ...
```
Right now two different tasks are implemented - [content summarization](https://github.com/gao-g/prelude/blob/main/src/task/summarization.py) and [email writing](https://github.com/gao-g/prelude/blob/main/src/task/email_writing.py)

Task specifics can be controlled using [TaskConfig](https://github.com/gao-g/prelude/blob/7171dd1a64fc2068133bde723ca779e74ee48766/src/configs.py#L30) which allows to:
1) Change the number of examples
2) Choose random seed
3) Specify data source

### User
User encapsulates access to task and LLM resource for simulating user responses. For initialization, TaskConfig and [UserConfig](https://github.com/gao-g/prelude/blob/7171dd1a64fc2068133bde723ca779e74ee48766/src/configs.py#L4) (allowing to specify the LLM model name) are required. 

### Agent
Classes responsible for accomplishing the tasks, encapsulating access to LLM and learning algorithm implementations.

## Reproduce Our Experiments
All agents mentioned in our paper are located in the [agent folder](https://github.com/gao-g/prelude/tree/main/src/agent). 

(INSTRUCTIONS TO BE ADDED)

## Implement Your Own Agent
Every agent should be inherited from the base [Agent](https://github.com/gao-g/prelude/tree/main/src/agent/abstract_agent.py#L10C7-L10C12) class, and have implementations of the following methods:
1) `def complete(self, text) -> LLMOutput` - task completion method returning LLMOutput object containing output text and (optionally) debug token information
2) `def learn(self, message, correction: Correction) -> Dict` - learning method taking context text and pair of (agent completion, user edits) as inputs. Return value is the dictionary of metrics required to be logged.

Please check [the notebook example](https://github.com/gao-g/prelude/blob/main/examples/Try_new_agent.ipynb) of dummy agent implementation and end-to-end experiment run here. 



