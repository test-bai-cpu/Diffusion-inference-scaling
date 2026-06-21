from search.configs import Arguments
from search.script_utils import get_pipe, get_args

def main(dataset: str="pointmaze-giant-navigate-v0", method: str='dfs', device: str="cuda:7", version: str=''):
    args = Arguments()
    args.device = device
    args.dataset = dataset
    args.method = method
    args.version = version
    args.task = [1, 2, 3, 4, 5]
    args_grid = get_args(args)

    for args in args_grid:
        pipe = get_pipe(args)
        returns = pipe.experiment()
        success_rate = returns['average']['total_reward']
        average_compute = returns['average']['compute']
        print(f"Success Rate: {success_rate}, Average Compute: {average_compute}")
        output_file = f'results_{args.method}_{args.dataset}.txt'
        version_str = f" | Version: {args.version}" if args.version else ""
        with open(output_file, 'a') as f:
            f.write(f"Maze: {args.dataset} | Method: {args.method}{version_str} | Compute: {average_compute} | Success Rate: {success_rate}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run inference scaling experiment.")
    parser.add_argument('--dataset', 
                        type=str, 
                        default='pointmaze-ultra-navigate-v0', 
                        choices=['pointmaze-giant-navigate-v0', 'pointmaze-ultra-navigate-v0'],
                        help='Maze env to use for the experiment.')
    parser.add_argument('--method', 
                        type=str, 
                        default='bon', 
                        choices=['dfs', 'bon', 'bfs-resampling', 'bfs-pruning'],
                        help='Search method to use')
    parser.add_argument('--device',
                        type=str,
                        default='cuda',
                        )
    parser.add_argument('--version',
                        type=str,
                        default='',
                        help='Optional version tag (e.g. ada) shown in results and folder names.')
    cli_args = parser.parse_args()

    main(dataset=cli_args.dataset, method=cli_args.method, device=cli_args.device, version=cli_args.version)