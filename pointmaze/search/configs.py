import os
from dataclasses import dataclass, field
from typing import Literal, Optional, Union, List

@dataclass
class Arguments:
    
    # data related
    data_type: Literal['traj'] = field(default='traj')
    dataset: str = field(default='pointmaze-giant-navigate-v0')
    task: List[int] = field(default_factory=lambda: [1,])
    method: str = field(default='bon')   

    # diffusion related
    train_steps: int = field(default=256)
    inference_steps: int = field(default=16)
    eta: float = field(default=1.0)
    clip_x0: bool = field(default=True)
    clip_sample_range: float = field(default=1.0)

    # maze related
    sampling_horizon: int = field(default=600) # 600 for Giant and 2800 for Ultra

    # inference related:
    seed: int = field(default=42)
    device: str = field(default='cuda')
    logging_dir: str = field(default='logs')
    per_sample_batch_size: int = field(default=1)
    num_samples: int = field(default=40)
    batch_id: int = field(default=0)    # start from the zero

    # guidance related
    guidance_name: str = field(default='no')
    recur_steps: int = field(default=1)    
    iter_steps: int = field(default=1)
    clip_scale: float = field(default=100)

    # specific for local search
    rho: float = field(default=0.04)
    mu: float = field(default=0.01)
    sigma: float = field(default=0.00)
    eps_bsz: int = field(default=1)
    rho_schedule: str = field(default='increase')
    mu_schedule: str = field(default='increase')
    sigma_schedule: str = field(default='decrease')

    # for dfs
    threshold: float = field(default=1000)
    threshold_schedule: str = field(default='increase')
    recur_depth: int = field(default=4)
    budget: int = field(default=4)

    # for bfs
    temp: float = field(default=1.0)
    temp_schedule: str = field(default='increase')

    # for eval steps
    start_step: int = field(default=4)
    step_size: int = field(default=4)

    # for OOD maze variants loaded from JSON
    maze_json_dir: str = field(default='')   # dir containing giant_task{N}.json files
    maze_variant_idx: int = field(default=0)
    model_dataset: str = field(default='')   # checkpoint dataset; defaults to dataset if empty

    # BFS distance field guidance
    use_distance_field: bool = field(default=False)
    dist_omega: float = field(default=1.0)          # scale on distance-field cost
    # NOTE: 'endpoint' is useless when apply_conditioning pins t=T-1 to goal (D=0 always).
    # Use 'sum' (mean per timestep, range 0..max_BFS_dist) as the default.
    # For DFS, tune --threshold upward when using 'sum' with omega=1.0 (effective cost ≈ 15
    # for a good trajectory on the giant maze, vs the default DFS threshold of 6).
    dist_mode: str = field(default='sum')            # endpoint | sum | weighted | monotonic
    dist_smooth_sigma: float = field(default=0.5)    # Gaussian blur on raw BFS field
    dist_connectivity: int = field(default=4)        # 4 or 8
    maze_weight: float = field(default=1.0)          # weight of MazeVerifier in composite
    dist_weight: float = field(default=1.0)          # weight of DistanceFieldVerifier in composite

