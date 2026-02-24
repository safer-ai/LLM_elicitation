import marimo

__generated_with = "0.18.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import yaml
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import kendalltau, chi2, rankdata
    import scipy
    from fractions import Fraction
    return Fraction, chi2, mo, np, plt, rankdata, yaml


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Configuration + Loading
    """)
    return


@app.cell
def _(yaml):
    def load_yaml(file: str) -> dict:
        """Loads the experiment configuration from a YAML file."""
        with open(file, 'r') as f:
            return yaml.safe_load(f)
    return (load_yaml,)


@app.cell
def _(chi2, np):
    def kendalls_w(rankings):
        """
        Calculates Kendall's W (Coefficient of Concordance) for multiple rankings.

        Args:
            rankings (list of lists or 2D numpy array): Each inner list/row represents
                                                      the ranking of items by a judge.
                                                      All lists must have the same length.

        Returns:
            float: The Kendall's W coefficient.
            float: The Chi-squared statistic.
            float: The p-value for the Chi-squared statistic.
        """
        if not isinstance(rankings, (list, np.ndarray)):
            raise TypeError("Rankings must be a list of lists or a 2D numpy array.")
        if len(rankings) < 2:
            raise ValueError("At least two rankings are required for Kendall's W.")

        num_judges = len(rankings)
        num_items = len(rankings[0])

        # Ensure all rankings have the same number of items
        for r in rankings:
            if len(r) != num_items:
                raise ValueError("All rankings must have the same number of items.")

        # Convert lists to a numpy array for easier calculation
        rankings_array = np.array(rankings)

        # Calculate the sum of ranks for each item across all judges
        # If the input contains values instead of ranks, convert them to ranks first.
        # We assume the input 'rankings' are already ranks or convert them to ranks.
        # If your input are actual values, you'd apply rankdata to each row first.
        # For simplicity, we assume 'rankings' are already assigned ranks (e.g., 1st, 2nd, 3rd).
        # If not, you'd apply rankdata to each row before summing.
        # e.g., ranked_data = np.array([rankdata(row, method='average') for row in rankings])
        # sum_of_ranks_per_item = np.sum(ranked_data, axis=0)

        # Assuming input `rankings` are already actual ranks (e.g., 1, 2, 3...)
        # If they are raw scores that need to be ranked, use rankdata as commented above.
        sum_of_ranks_per_item = np.sum(rankings_array, axis=0)

        # Calculate the mean of the sum of ranks
        mean_sum_of_ranks = np.mean(sum_of_ranks_per_item)

        # Calculate the sum of squared deviations (S)
        S = np.sum((sum_of_ranks_per_item - mean_sum_of_ranks)**2)

        # Calculate the maximum possible sum of squared deviations (S_max)
        # This formula is for when there are no tied ranks within any judge's ranking.
        # If there are ties, a more complex adjustment for T is needed.
        # For simplicity, assuming no significant ties within individual rankings that affect the denominator.
        # The common denominator is: m^2 * (n^3 - n) / 12
        # m = num_judges, n = num_items
        S_max = num_judges**2 * (num_items**3 - num_items) / 12

        # Calculate Kendall's W
        W = S / S_max

        # Calculate Chi-squared statistic
        # Chi-squared = num_judges * (num_items - 1) * W
        chi_squared = num_judges * (num_items - 1) * W

        # Degrees of freedom for Chi-squared
        df = num_items - 1

        # Calculate p-value from Chi-squared distribution

        p_value = 1 - chi2.cdf(chi_squared, df)

        return W, chi_squared, p_value
    return (kendalls_w,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Scores
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Convert to Ranking
    """)
    return


@app.cell
def _(load_yaml):
    response_files = [
        "./results/swebench_verified/fst/ordered_tasks_subset.yaml",
        "./results/swebench_verified/difficulty/ordered_tasks_subset.yaml",
    ]

    responses = [{response["task_id"]:response for response in load_yaml(response_file)} for response_file in response_files]
    ordered_tasks = list(sorted(responses[0].keys()))
    return ordered_tasks, response_files, responses


@app.cell
def _(Fraction, np, ordered_tasks, rankdata, responses):
    task_scores = np.zeros((len(responses), len(ordered_tasks)))
    for _j, task in enumerate(ordered_tasks):
        for _i, response in enumerate(responses):
            est = response[task].get('estimate')
            task_scores[_i, _j] = float(Fraction(est)) if isinstance(est, str) else est
    task_rankings = rankdata(task_scores, axis=1, method='ordinal')
    print(f"task_scores shape: {task_scores.shape}")
    return task_rankings, task_scores


@app.function
def get_task_id(task: dict) -> str:
    """Returns the unique instance_id for a SWEBench-verified task."""
    return task['instance_id']


@app.cell
def _(kendalls_w, plt):
    def plot_kendall_w(task_rankings, ticks=[]):
        W, chi_squared, p_value = kendalls_w(task_rankings)
        print('Overall W:')
        print(f'\tW: {W}')
        print(f'\tχ²: {chi_squared}')
        print(f'\tp-value: {p_value}')
        results = []
        for _i in range(len(task_rankings)):
            result = []
            for _j in range(len(task_rankings)):
                result.append(kendalls_w([task_rankings[_i], task_rankings[_j]])[0])
            results.append(result)
        print('Pairwise W:')
        plt.figure(figsize=(8, 8))
        plt.imshow(results, vmin=0, vmax=1)
        plt.colorbar()
        plt.xticks(range(len(ticks)), ticks, rotation=90)
        plt.yticks(range(len(ticks)), ticks)  #ticks = [f.split('/')[-1].split('.')[0][:-7] for f in response_files]
        plt.title("Kendall's W Correlation Matrix")
        plt.ylabel('Task')
        plt.xlabel('Task')
        plt.show()
    return (plot_kendall_w,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Plot
    """)
    return


@app.cell
def _(plot_kendall_w, response_files, task_rankings):
    score_labels = [f.split('/')[-2] for f in response_files]
    plot_kendall_w(task_rankings, score_labels)
    return (score_labels,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Score details
    """)
    return


@app.cell
def _(plt, task_scores):
    plt.subplots(1, 2, figsize=(10, 4))
    for _i in range(2):
        plt.subplot(1, 2, _i + 1)
        plt.hist(task_scores[_i], bins=100, alpha=1, label=f'Difficulty {_i + 1}')
    #plt.legend()
        if _i == 0:
            plt.xlabel('FST')
        else:
            plt.xlabel('Difficulty')
        plt.ylabel('Count')
    plt.show()
    return


@app.cell
def _(np, plt, task_scores):
    plt.figure()
    plt.scatter(np.arange(len(task_scores[0])), sorted(task_scores[0]))
    plt.yscale('log')
    plt.xlabel('Task (FST Sorted)')
    plt.ylabel('FST')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Adding rank-based lists and more stats
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Getting Bounty Maps and loading Rank Data
    """)
    return


@app.cell
def _(load_yaml):
    unordered_tasks = load_yaml("benchmark_tasks/swebench_verified_subset.yaml")
    task_map = {get_task_id(t):t for t in unordered_tasks["tasks"]}
    return task_map, unordered_tasks


@app.cell
def _(load_yaml, ordered_tasks, task_map):
    def get_tasks_from_ranked_list(path):
        ranked_tasks = load_yaml(path)
        rt = [o['task_id'] for o in ranked_tasks]
        task_ranks_by_task_index = []
        for f in ordered_tasks:
            try:
                task_ranks_by_task_index.append(rt.index(f))
            except ValueError:
                continue
        rt = [task_map[f] for f in rt]
        return (task_ranks_by_task_index, rt, ranked_tasks)

    def get_tasks_by_ranking(ranking):
        ranked_task_ids = [ordered_tasks[c] for c in ranking]
        tasks_list = [task_map[t] for t in ranked_task_ids]
        return tasks_list
    return get_tasks_by_ranking, get_tasks_from_ranked_list


@app.cell
def _(get_tasks_from_ranked_list, np, task_rankings):
    task_ranks_by_task_index, ranked_tasks, rt_orig = get_tasks_from_ranked_list('./results/swebench_verified/iterative_easiest/ordered_tasks_subset.yaml')
    task_ranks_by_task_index_hard, ranked_tasks_hard, rt_orig_hard = get_tasks_from_ranked_list('./results/swebench_verified/iterative_hardest/ordered_tasks_subset.yaml')
    task_rankings_1 = np.vstack([task_rankings, task_ranks_by_task_index, task_ranks_by_task_index_hard])
    return (task_rankings_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Plot
    """)
    return


@app.cell
def _(plot_kendall_w, score_labels, task_rankings_1):
    all_labels = score_labels + ['iterative easy first', 'iterative hard first']
    plot_kendall_w(task_rankings_1, all_labels)
    return (all_labels,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Consensus
    """)
    return


@app.cell
def _(np, rankdata):
    def borda_count_consensus(scores_matrix, item_names=None):
        """
        Calculates the consensus ordering using the Borda Count method.

        Assumes:
        - scores_matrix: A NumPy array where rows are items and columns are lists/sources.
                         Higher scores indicate better preference.
        - item_names: An optional list of names for the items, corresponding to rows.
                      If provided, the output will include item names.

        Returns:
        - A list of tuples, (item_index, total_borda_score), sorted in descending
          order of Borda score (consensus ranking).
        - If item_names is provided, returns (item_name, total_borda_score).
        """

        num_items, num_lists = scores_matrix.shape

        # Step 1 & 2: Rank each list and convert to Borda points
        # We want higher scores to get more points.
        # rankdata assigns rank 1 to the lowest value.
        # So, we first take the negative of scores_matrix to rank higher scores lower (rank 1 is best)
        # Then, we convert these ranks to Borda points: (num_items - actual_rank)

        borda_points_matrix = np.zeros_like(scores_matrix, dtype=float)

        for col_idx in range(num_lists):
            # Rank the scores in the current column (list)
            # Using 'average' method to handle ties correctly
            # rankdata assigns 1 to the lowest score, N to the highest.
            # We need the inverse: higher score -> lower rank_value.
            # So, we rank (-scores) to achieve this, where higher original scores
            # will have lower negative values and thus lower ranks.
            ranks_in_list = rankdata(-scores_matrix[:, col_idx], method='average')

            # Convert ranks to Borda points: (num_items - rank)
            # For rank 1 (best), points = num_items - 1
            # For rank num_items (worst), points = num_items - num_items = 0
            borda_points_matrix[:, col_idx] = num_items - ranks_in_list

        # Step 3: Sum Borda Points for each item
        total_borda_scores = np.sum(borda_points_matrix, axis=1)

        # Step 4: Derive Consensus Order
        # Get the indices that would sort the total_borda_scores in descending order
        sorted_indices = np.argsort(total_borda_scores)[::-1]

        consensus_ranking = []
        borda_scores = []
        for rank, item_idx in enumerate(sorted_indices):
            item_identifier = item_names[item_idx] if item_names is not None else f"Item {item_idx}"
            consensus_ranking.append(int(item_idx))
            borda_scores.append(total_borda_scores[item_idx])

        return consensus_ranking, borda_scores
    return (borda_count_consensus,)


@app.cell
def _(borda_count_consensus, np, task_rankings_1):
    consensus_ranking, borda_scores = borda_count_consensus(task_rankings_1.T)
    consensus_task_ranks_by_task_index = len(consensus_ranking) - np.argsort(consensus_ranking)
    return borda_scores, consensus_task_ranks_by_task_index


@app.cell
def _(
    all_labels,
    consensus_task_ranks_by_task_index,
    np,
    plot_kendall_w,
    task_rankings_1,
):
    task_rankings_2 = np.vstack([task_rankings_1, np.array(consensus_task_ranks_by_task_index)])
    all_labels_consensus = all_labels + ['consensus_ranking']
    plot_kendall_w(task_rankings_2, all_labels_consensus)
    return (task_rankings_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Saving Results
    """)
    return


@app.cell
def _(borda_scores, get_tasks_by_ranking, task_scores, unordered_tasks, yaml):
    def save_from_ranking(order, save_path='results/swebench_verified/swebench_verified_ordered_subset.yaml', filter=[]):
        _tasks = get_tasks_by_ranking(order)
        # print(tasks[0].keys())
        task_ixs = [_i for _i, task in enumerate(_tasks) if len(filter) == 0 or get_task_id(task) in filter]
        _tasks = [_tasks[_i] for _i in task_ixs]
        order_filtered = [order[_i] for _i in task_ixs]
        borda_scores_filtered = [borda_scores[_i] for _i in task_ixs]

        # Reorder each task's keys to match desired order
        ordered_tasks = []
        for _i, (_ix, score) in enumerate(zip(order_filtered, borda_scores_filtered)):
            task = _tasks[_i]
            # Create ordered dict with desired key order
            ordered_task = {}
            ordered_task['instance_id'] = task['instance_id']
            ordered_task['problem_statement'] = task['problem_statement']
            ordered_task['patch'] = task['patch']
            ordered_task['test_patch'] = task['test_patch']

            # Create metrics dict with desired order
            metrics = {}
            metrics['borda_score'] = str(score)
            metrics['estimated_difficulty'] = str(task_scores[1, _ix])
            metrics['estimated_fst'] = str(task_scores[0, _ix])
            # Add difficulty from original if it exists
            if 'difficulty' in task.get('metrics', {}):
                metrics['difficulty'] = task['metrics']['difficulty']
            elif 'difficulty' in task:
                metrics['difficulty'] = task['difficulty']

            ordered_task['metrics'] = metrics
            ordered_tasks.append(ordered_task)

        with open(save_path, 'w') as f:
            yaml.dump({**unordered_tasks, **{'tasks': ordered_tasks}}, f, sort_keys=False)
    return (save_from_ranking,)


@app.cell
def _(ordered_tasks, save_from_ranking, task_map, task_rankings_2):
    final_tasks_sorted = task_rankings_2[-1].argsort()
    save_from_ranking(final_tasks_sorted)
    for _i, c in enumerate(final_tasks_sorted):
        print(f"{_i + 1}: {task_map[ordered_tasks[c]]['instance_id']}")
    return


if __name__ == "__main__":
    app.run()
