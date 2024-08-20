# Experiments
Configuration files and scripts for reproduction results of the [Aligning LLM Agents by Learning Latent Preference from User Edits](https://arxiv.org/pdf/2404.15269) paper.

# How to run
There are two experiment scripts - run_summarization.sh for "Article summarization" and run_email.sh for "Email writing". There are 3 configuration files (one per random seed) for each task.
Here is the instruction for running summarization experiment (email writing one is similar):
1. Update [lines 4 and 5](https://github.com/gao-g/prelude/blob/2568b96994121e939cc94a952efbbfe6ff51c30d/experiments/run_summarization.sh#L4) of shell script with your AzureOpenAi endpoint and key
2. Update [line 7](https://github.com/gao-g/prelude/blob/2568b96994121e939cc94a952efbbfe6ff51c30d/experiments/run_summarization.sh#L7C1-L7C25) with your python path if needed
3. `bash experiments/run_summarization.sh` from the root folder of the repository
4. After experiment is done, the "Aggregate" cells from [the analysis notebook](https://github.com/gao-g/prelude/blob/main/experiments/analysis.ipynb) will produce Table 3 from the Section 4.3 of the paper.

