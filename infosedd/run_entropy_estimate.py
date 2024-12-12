import torch
import argparse

from load_model import load_model
from transformers import GPT2TokenizerFast
import torch.nn.functional as F
import sampling


def main():
    parser = argparse.ArgumentParser(description="Estimate entropy")
    parser.add_argument("--model_path", default="louaaron/sedd-medium", type=str)
    parser.add_argument("--dataset", default="wikitext103", type=str)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--mc_estimates", type=int, default=100)
    args = parser.parse_args()

    
    device = torch.device('cuda')
    model, graph, noise = load_model(args.model_path, device)

    sampling_fn = sampling.get_pc_sampler(
        graph, noise, (args.batch_size, 1024), 'analytic', args.steps, device=device
    )

    estimates = []
    for i in range(args.mc_estimates):
        entropy = sampling_fn(model)
        estimates.append(entropy)
    
    estimate = torch.mean(estimates)

    print("Entropy estimated: ", estimate)

if __name__=="__main__":
    main()