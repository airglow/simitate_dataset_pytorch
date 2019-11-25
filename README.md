# Simitate dataset class for pytorch

Simitate is a hybrid approach for benchmarking imitation learning tasks. This repository gives a pytorch dataset integration for the Simitate dataset which is part of the benchmarking approach.

Currently only trajectories are supported. The dataset class downloads the simitate trajectories 
and creates a pytorch dataset class.

As of now just trajectories are loaded. Image sequences will be added soon.

## Examples

### Loading

```python
import torch
from simitate_dataset import SimitateTrajectoriesDataset
# First param is the source/taget directory, Second parameter describes if the trajectories should be downloaded
# Subcategories can be chosen as a list of category names
simitate_data = SimitateTrajectoriesDataset(".", download=True, categories=["basic_motions"]) 
```

### Plotting


```python
for i in range(3): 
    data_sample = random.choice(range(len(simitate_data.trajectory_data)))
    simitate_data.plot(data_sample)
```

![Heart](examples/heart_plot.png "Heart")
![Rect](examples/rect_plot.png "Rectangle")
![Triangle](examples/triangle_plot.png "Triangle")


## Run

* `jupyter notebook`
