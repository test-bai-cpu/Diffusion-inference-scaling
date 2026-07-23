import torch


def distance_to_closest_non_adjacent_wall_boundary(pos, wall_pos, wall_size, adjacency):
    lower_bound = wall_pos - wall_size   # (2,)
    upper_bound = wall_pos + wall_size   # (2,)

    is_inside = torch.logical_and(
        torch.logical_and(pos[..., 0] >= lower_bound[0], pos[..., 0] <= upper_bound[0]),
        torch.logical_and(pos[..., 1] >= lower_bound[1], pos[..., 1] <= upper_bound[1])
    )
    dist_x_low = torch.where(
        is_inside,
        pos[..., 0] - lower_bound[0],
        torch.zeros_like(pos[..., 0])
    ) if adjacency[2] == 1 else torch.full_like(pos[..., 0], float('inf'))
    dist_x_high = torch.where(
        is_inside,
        upper_bound[0] - pos[..., 0],
        torch.zeros_like(pos[..., 0])
    ) if adjacency[3] == 1 else torch.full_like(pos[..., 0], float('inf'))
    dist_y_low = torch.where(
        is_inside,
        pos[..., 1] - lower_bound[1],
        torch.zeros_like(pos[..., 1])
    ) if adjacency[0] == 1 else torch.full_like(pos[..., 1], float('inf'))
    dist_y_high = torch.where(
        is_inside,
        upper_bound[1] - pos[..., 1],
        torch.zeros_like(pos[..., 1])
    ) if adjacency[1] == 1 else torch.full_like(pos[..., 1], float('inf'))
    dist = torch.minimum(
        torch.minimum(dist_x_low, dist_x_high),
        torch.minimum(dist_y_low, dist_y_high)
    )
    return dist

def batched_inside_wall_loss_non_adjacent(positions, wall_boxes, wall_adjacency):
    barch_size, timesteps, _ = positions.shape
    device = positions.device
    all_distances = torch.zeros((barch_size, timesteps, len(wall_boxes)), device=device)
    for w, (wall_pos, wall_size) in enumerate(wall_boxes):
        pos_2d = positions.reshape(barch_size * timesteps, 2)
        dist_2d = distance_to_closest_non_adjacent_wall_boundary(pos_2d, wall_pos.to(device), wall_size.to(device), wall_adjacency[w])
        dist_2d = dist_2d.view(barch_size, timesteps)
        all_distances[..., w] = dist_2d
    dist = all_distances.max(dim=-1).values
    loss = dist ** 2
    return loss


def get_forbidden_corner_points(env, device):
    """
    Return world-space corner points for diagonal wall pairs:

        wall free      free wall
        free wall  or  wall free

    A point robot can numerically slip through this zero-width diagonal gap
    while barely entering either wall box. Marking a small hard forbidden zone
    around the shared grid corner catches that corner-cut topology.
    """
    corners = []
    grid = env.maze_map
    H, W = grid.shape
    mu = env._maze_unit
    ox = env._offset_x
    oy = env._offset_y

    for i in range(H - 1):
        for j in range(W - 1):
            nw = grid[i, j] == 1
            ne = grid[i, j + 1] == 1
            sw = grid[i + 1, j] == 1
            se = grid[i + 1, j + 1] == 1
            if (nw and se and not ne and not sw) or (ne and sw and not nw and not se):
                x = (j + 0.5) * mu - ox
                y = (i + 0.5) * mu - oy
                corners.append((x, y))

    if not corners:
        return torch.empty((0, 2), dtype=torch.float32, device=device)
    return torch.tensor(corners, dtype=torch.float32, device=device)

def batched_diagonal_transition_loss(positions, maze_grid, maze_unit, offset_x, offset_y):
    """
    Hard topological penalty for crossing a diagonal zero-width corner gap.

    This catches transitions like:

        wall free      free wall
        free wall  or  wall free

    when consecutive trajectory points jump between the two free cells. It does
    not depend on penetration depth, so it catches the corner-cut cases where
    the path barely enters either wall box.
    """
    if maze_grid is None or positions.shape[1] < 2:
        return torch.zeros(positions.shape[:2], dtype=positions.dtype, device=positions.device)

    device = positions.device
    grid = maze_grid.to(device=device)
    H, W = grid.shape

    x = positions[..., 0]
    y = positions[..., 1]
    i = torch.floor((y + offset_y + 0.5 * maze_unit) / maze_unit).long()
    j = torch.floor((x + offset_x + 0.5 * maze_unit) / maze_unit).long()

    i0, j0 = i[:, :-1], j[:, :-1]
    i1, j1 = i[:, 1:], j[:, 1:]
    valid0 = (i0 >= 0) & (i0 < H) & (j0 >= 0) & (j0 < W)
    valid1 = (i1 >= 0) & (i1 < H) & (j1 >= 0) & (j1 < W)

    di = i1 - i0
    dj = j1 - j0
    diagonal_adjacent = (di.abs() == 1) & (dj.abs() == 1)

    def cell_is_wall(ii, jj):
        ii = ii.clamp(0, H - 1)
        jj = jj.clamp(0, W - 1)
        return grid[ii, jj] == 1

    endpoints_free = (~cell_is_wall(i0, j0)) & (~cell_is_wall(i1, j1))
    side_wall_a = cell_is_wall(i0, j1)
    side_wall_b = cell_is_wall(i1, j0)
    corner_cut = valid0 & valid1 & diagonal_adjacent & endpoints_free & side_wall_a & side_wall_b

    out = torch.zeros(positions.shape[:2], dtype=positions.dtype, device=device)
    out[:, 1:] = corner_cut.to(dtype=positions.dtype)
    return out


def batched_corner_zone_transition_loss(positions, corner_points, radius):
    """
    Hard geometric penalty for entering a forbidden corner zone.

    This is binary and non-smooth: it is meant for DFS accept/reject, not for
    gradient shaping. It catches dense paths that slide through a diagonal pinch
    without producing a cell-index diagonal jump.
    """
    if radius <= 0 or corner_points.numel() == 0:
        return torch.zeros(positions.shape[:2], dtype=positions.dtype, device=positions.device)

    corners = corner_points.to(device=positions.device, dtype=positions.dtype)
    delta = positions.unsqueeze(-2) - corners.view(1, 1, -1, 2)
    point_hit = (delta.square().sum(dim=-1) <= radius ** 2).any(dim=-1)

    if positions.shape[1] < 2:
        return point_hit.to(dtype=positions.dtype)

    p0 = positions[:, :-1]
    p1 = positions[:, 1:]
    seg = p1 - p0
    seg_len2 = seg.square().sum(dim=-1, keepdim=True).clamp_min(1e-8)
    rel = corners.view(1, 1, -1, 2) - p0.unsqueeze(-2)
    proj = (rel * seg.unsqueeze(-2)).sum(dim=-1, keepdim=True) / seg_len2.unsqueeze(-2)
    proj = proj.clamp(0.0, 1.0)
    closest = p0.unsqueeze(-2) + proj * seg.unsqueeze(-2)
    seg_hit = ((closest - corners.view(1, 1, -1, 2)).square().sum(dim=-1) <= radius ** 2).any(dim=-1)

    out = point_hit.clone()
    out[:, 1:] = out[:, 1:] | seg_hit
    return out.to(dtype=positions.dtype)


def get_wall_boxes_and_adjacency(env,device):
    wall_boxes = []
    wall_adjacency = []
    if env._maze_type == 'ultra':
        consider_ball = True
    else:
        consider_ball = False
    for i in range(env.maze_map.shape[0]):
        for j in range(env.maze_map.shape[1]):
            if env.maze_map[i, j] == 1:
                wall_pos = (j * env._maze_unit - env._offset_x, i * env._maze_unit - env._offset_y)
                wall_size = (env._maze_unit / 2, env._maze_unit / 2)
                if consider_ball:
                    wall_size = (wall_size[0] + 0.7, wall_size[1] + 0.7)
                assert wall_pos == env.ij_to_xy((i, j))
                adjacency = [1] * 4
                if i > 0 and env.maze_map[i-1, j] == 1 or i == 0:
                    adjacency[0] = 0
                if i < env.maze_map.shape[0]-1 and env.maze_map[i+1, j] == 1 or i == env.maze_map.shape[0]-1:
                    adjacency[1] = 0
                if j > 0 and env.maze_map[i, j-1] == 1 or j == 0:
                    adjacency[2] = 0
                if j < env.maze_map.shape[1]-1 and env.maze_map[i, j+1] == 1 or j == env.maze_map.shape[1]-1:
                    adjacency[3] = 0
                if adjacency[0] == 0 and adjacency[1] == 0 and adjacency[2] == 0 and adjacency[3] == 0:
                    if i == 0: adjacency[1] = 1
                    elif i == env.maze_map.shape[0]-1: adjacency[0] = 1
                    elif j == 0: adjacency[3] = 1
                    elif j == env.maze_map.shape[1]-1: adjacency[2] = 1
                    else: adjacency = [1, 1, 1, 1]  # interior wall fully surrounded by walls: penalize from all faces

                wall_boxes.append((wall_pos, wall_size))
                wall_adjacency.append(adjacency)
    
    # Convert wall_boxes into torch tensors
    wall_boxes = [(torch.tensor(pos[:2], dtype=torch.float32,device=device), torch.tensor(size[:2], dtype=torch.float32,device=device)) for pos, size in wall_boxes]
    return wall_boxes,wall_adjacency



from search.utils import check_grad_fn, rescale_grad, ban_requires_grad
from search.configs import Arguments

class MazeVerifier():
    def __init__(self, args: Arguments = Arguments()):
        self.device = torch.device(args.device)
        self.wall_boxes = None
        self.adjacency = None
        self.corner_points = None
        self.corner_radius_frac = float(getattr(args, "corner_radius_frac", 0.30))
        self.corner_transition_weight = float(getattr(args, "corner_transition_weight", 20.0))
        self.corner_radius = 0.0
        self.maze_grid = None
        self.maze_unit = None
        self.offset_x = None
        self.offset_y = None
        self.monitor = bool(getattr(args, "verifier_monitor", False))
        self.last_stats = {}

    def update_env(self, env):
        self.env = env
        self.wall_boxes, self.adjacency = get_wall_boxes_and_adjacency(env, device=self.device)
        self.corner_points = get_forbidden_corner_points(env, device=self.device)
        self.corner_radius = self.corner_radius_frac * float(env._maze_unit)
        self.maze_grid = torch.tensor(env.maze_map, dtype=torch.long, device=self.device)
        self.maze_unit = float(env._maze_unit)
        self.offset_x = float(env._offset_x)
        self.offset_y = float(env._offset_y)
        if self.monitor:
            print(
                "[MazeVerifier] walls=%d forbidden_corners=%d "
                "corner_radius=%.3f corner_transition_weight=%.3f" % (
                    len(self.wall_boxes),
                    int(self.corner_points.shape[0]),
                    self.corner_radius,
                    self.corner_transition_weight,
                ),
                flush=True,
            )

    def get_guidance(self, x, func=lambda x:x, post_process=lambda x:x, return_logp=False, check_grad=True, **kwargs):
        if check_grad:
            check_grad_fn(x)
        
        x = post_process(func(x))
        
        x = x[..., 2:4] # only use the x, y coordinates

        wall_loss = batched_inside_wall_loss_non_adjacent(x, self.wall_boxes, self.adjacency)
        cell_transition_loss = batched_diagonal_transition_loss(
            x, self.maze_grid, self.maze_unit, self.offset_x, self.offset_y
        )
        zone_transition_loss = batched_corner_zone_transition_loss(
            x, self.corner_points, self.corner_radius
        )
        transition_loss = torch.maximum(cell_transition_loss, zone_transition_loss)
        loss = wall_loss + self.corner_transition_weight * transition_loss
        log_probs = loss * -1
        wall_cost = wall_loss.sum(dim=tuple(range(1, wall_loss.ndim)))
        cell_transition_cost = cell_transition_loss.sum(dim=tuple(range(1, cell_transition_loss.ndim)))
        zone_transition_cost = zone_transition_loss.sum(dim=tuple(range(1, zone_transition_loss.ndim)))
        transition_cost = transition_loss.sum(dim=tuple(range(1, transition_loss.ndim)))
        total_cost = loss.sum(dim=tuple(range(1, loss.ndim)))
        # per-timestep hit mask (batch, T): True where this timestep is inside a
        # wall or on a forbidden corner transition. Used by DFS local re-noising
        # to target only the violating region instead of the whole trajectory.
        violation_mask = (wall_loss.detach() > 0) | (transition_loss.detach() > 0)
        self.last_stats = {
            "wall_cost_mean": float(wall_cost.detach().mean().cpu()),
            "wall_cost_max": float(wall_cost.detach().max().cpu()),
            "corner_cell_transition_raw_mean": float(cell_transition_cost.detach().mean().cpu()),
            "corner_cell_transition_raw_max": float(cell_transition_cost.detach().max().cpu()),
            "corner_zone_transition_raw_mean": float(zone_transition_cost.detach().mean().cpu()),
            "corner_zone_transition_raw_max": float(zone_transition_cost.detach().max().cpu()),
            "corner_transition_raw_mean": float(transition_cost.detach().mean().cpu()),
            "corner_transition_raw_max": float(transition_cost.detach().max().cpu()),
            "corner_transition_mean": float((self.corner_transition_weight * transition_cost).detach().mean().cpu()),
            "corner_transition_max": float((self.corner_transition_weight * transition_cost).detach().max().cpu()),
            "total_cost_mean": float(total_cost.detach().mean().cpu()),
            "total_cost_max": float(total_cost.detach().max().cpu()),
            "wall_hit_frac": float((wall_loss.detach() > 0).float().mean().cpu()),
            "corner_transition_hit_frac": float((transition_loss.detach() > 0).float().mean().cpu()),
            "n_forbidden_corners": int(0 if self.corner_points is None else self.corner_points.shape[0]),
            "corner_radius": float(self.corner_radius),
            "corner_transition_weight": float(self.corner_transition_weight),
            "violation_mask": violation_mask,
        }

        if return_logp:
            return log_probs.sum(dim=tuple(range(1, log_probs.ndim)))

        grad = torch.autograd.grad(log_probs.mean(), x)[0]

        return rescale_grad(grad, clip_scale=1.0, **kwargs)
