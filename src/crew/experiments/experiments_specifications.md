# Specifications for main experiments to be included in paper

## INTRINSIC REWARD FUNCTIONS CONSIDERED

- RND (only inventory) 

- NGU (only inventory) 

- ICM (full observation)


## EXTRINSIC REWARD FUNCTIONS CONSIDERED

- all experiments are performed without health rewards

**1 ACHIEVEMENT EXPERIMENTS**

We will pick 2-3 of the achievements from craftax classic.

**2 ACHIEVEMENT EXPERIMENTS**

We will pick pairs of 2 achievements. eg: place plant + eat plant, or make wood pickaxe + collect iron


## OTHER EXPERIMENTAL DETAILS:
- length of runs: All runs are equal length.
- We fix all hyperparameters that aren't the reward functions weights (no tuning). We have to put the final values we use on the main config.py file.


## CONSIDERATIONS ON SAVING RESULTS

- agree on a naming standard and directory structure for saves. set that as the default thing in the repo so that we can all understand each other's saves.

- each of us saves stuff locally

- We also keep on logging to weights and biases and agree on a naming standard for runs and also on what tags to add.



## EXPERIMENTS WE RUN AND PLOTS WE SHOW:

What follows describes what we do for EACH extrinsic task (for each 1 achievement or 2 achievement experiment).

All experiments are performed on craftax classic with symbolic observations.

First we select two out of the three intrinsic reward functions randomly.

### fixed weighting runs:

For the fixed weightings results we consider a 8x8 grid where each axis corresponds to one of the intrinsic reward functions. The values on each axis correspond to how much weight is put on it. The possible values on each axis are [0, 0.125, 0.250, …, 0.875]. The weight on the extrinsic reward function is 1 minus the sum of the weights selected for the intrinsic reward functions.

It is important to note that since the weights have to sum to 1, many grid cells represent invalid values. We can also have the requirement for the weight on the extrinsic reward to not be 0 (at least the minimum which is 0.125). Then every grid cell which corresponds to weightings on intrinsic reward functions that sum up to more than 0.9 are considered invalid.

We sample 9 values from the discretized simplex that are also valid positions in this grid and perform runs for each of them with three seeds. 

We also make a run (with three seeds) without any intrinsic reward function (only extrinsic). This would correspond to the grid cell that puts 0 weight on both of the intrinsic reward functions.

### curriculum runs:

We run the algorithm with three seeds. Throughout training, we evaluate on performance conditioning on every valid combination of reward weights from the grid defined above. config.evaluation_alphas=((1.0, 0.0, 0.0), (0.875, 0.125, 0.0), (0.875, 0.0, 0.125), … ),


### PLOT 1

We show two 8x8 grids as heatmaps. Color represents final performance. 
The first grid corresponds to the results from the fixed weights runs. 
- For the grid cells for which there was a run performed, it shows the color corresponding to the mean final performance across the 3 seeds. The value from each seed is itself obtained from multiple episodes.
- The invalid grid cells (the ones that correspond to weightings on intrinsic reward functions that sum up to more than 0.9) are masked out. They are black.
- The valid grid cells for which we did not perform a run, are left in white.
The second grid corresponds to the results from the curriculum method.
- Each valid grid cell is filled up with the color corresponding to the final performance of the alpha-conditioned method when conditioned on the weights corresponding to that grid cell. The mean from the 3 seeds is shown. The value from each seed is itself obtained from multiple episodes.
- The invalid grid cells (the ones that correspond to weightings on intrinsic reward functions that sum up to more than 0.9) are masked out. They are black.



### PLOT 2

We show a standard plot of extrinsic return vs num timesteps throughout training.
We plot:
- The performance from the extrinsic only run
- The performance from the best run using fixed weights. (if multiple got to the same final performance, we plot the one that learned the fastest)
- The performance from the curriculum method when conditioned on alpha that puts only weight on the extrinsic reward
- The performance from the curriculum method whit the best conditioning. There are several considerations to make for this one: 1) the results for the method conditioned on all the different alphas in the grid is logged less often, so this curve/scatter will have more sparse values than the others. 2) What is the best weighting is defined on a per-time and per-seed basis. We are plotting the best result we can get from the network if we tried different conditionings at that point in time.

All results are plotted as the average across the three seeds with standard deviations shown. The value from each seed is itself obtained from multiple episodes.


### PLOT 3

For one or two of the extrinsic tasks (maybe one with 1 achievement and one with 2), we can plot the mean weight the curriculum puts on each reward function as training progresses and pair this with the performance of the curriculum method when conditioned on alpha that puts only weight on the extrinsic reward


### PLOT 4

For one or two of the extrinsic tasks (maybe one with 1 achievement and one with 2) we make an extra run with a single seed. In this run we evaluate performance on every valid combination of reward weights from the grid more often than before (ideally on every batch). 

The plot now consists of a sequence of 2 grids as training progresses.
The first grid shows the performance for each valid alpha in the grid (similar to what we did for PLOT 2)
The second grid shows the alphas on which the network trained. The heatmap represents a density over which weight configurations the curriculum has selected.
By plotting a sequence of both this grids over time, we get a clearer view of how the curriculum behaves and the training dynamics.
