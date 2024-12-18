import datetime
import os
import os.path
import gc
from itertools import chain

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F

import infosedd.data
import infosedd.losses
import infosedd.sampling
import infosedd.graph_lib
import infosedd.noise_lib
import infosedd.utils
import json
from infosedd.model import SEDD
from infosedd.model.ema import ExponentialMovingAverage
from infosedd.model.mlp import DiffusionMLP
from infosedd.model.unetmlp import UnetMLP_simple
from transformers import GPT2TokenizerFast, GPT2LMHeadModel


torch.backends.cudnn.benchmark = True
# torch.autograd.set_detect_anomaly(True)

available_distributions = ["bernoulli", "binomial", "custom_joint", "custom_univariate","categorical", "xor"]


def setup(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    # initialize the process group
    dist.init_process_group(
        "nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(minutes=30)
    )


def cleanup():
    dist.destroy_process_group()


def run_multiprocess(rank, world_size, cfg, port):
    try:
        setup(rank, world_size, port)
        _run(rank, world_size, cfg)
    finally:
        cleanup()


def _run(rank, world_size, cfg):

    torch.cuda.set_device(rank)
    work_dir = cfg.work_dir

    # Create directories for experimental logs
    sample_dir = os.path.join(work_dir, "samples")
    checkpoint_dir = os.path.join(work_dir, "checkpoints")
    checkpoint_meta_dir = os.path.join(work_dir, "checkpoints-meta", "checkpoint.pth")
    if rank == 0:
        utils.makedirs(sample_dir)
        utils.makedirs(checkpoint_dir)
        utils.makedirs(os.path.dirname(checkpoint_meta_dir))

    # logging
    if rank == 0:
        logger = utils.get_logger(os.path.join(work_dir, "logs"))
    def mprint(msg):
        if rank == 0:
            logger.info(msg)

    mprint(work_dir)
    mprint(cfg)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    results = {}
    if device.type == "cuda":
        mprint("Found {} CUDA devices.".format(torch.cuda.device_count()))
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mprint(
                "{} \t Memory: {:.2f}GB".format(
                    props.name, props.total_memory / (1024 ** 3)
                )
            )
    else:
        mprint("WARNING: Using device {}".format(device))
    mprint(f"Found {os.cpu_count()} total number of CPUs.")

    # build token graph
    graph = graph_lib.get_graph(cfg, device)
    
    # build score model
    if cfg.model.name == "mlp":
        score_model = DiffusionMLP(cfg).to(device)
    elif cfg.model.name == "unetmlp":
        score_model = UnetMLP_simple(cfg).to(device)
    else:
        score_model = SEDD(cfg).to(device)
    score_model = DDP(score_model, device_ids=[rank], static_graph=False, find_unused_parameters=True)

    num_parameters = sum(p.numel() for p in score_model.parameters())
    mprint(f"Number of parameters in the model: {num_parameters}")

    ema = ExponentialMovingAverage(
        score_model.parameters(), decay=cfg.training.ema)
    mprint(score_model)
    mprint(f"EMA: {ema}")

    # build noise
    noise = noise_lib.get_noise(cfg).to(device)
    noise = DDP(noise, device_ids=[rank], static_graph=False)
    sampling_eps = 1e-5


    # build optimization state
    optimizer = losses.get_optimizer(cfg, chain(score_model.parameters(), noise.parameters()))
    mprint(f"Optimizer: {optimizer}")
    scaler = torch.cuda.amp.GradScaler()
    mprint(f"Scaler: {scaler}")
    state = dict(optimizer=optimizer, scaler=scaler, model=score_model, noise=noise, ema=ema, step=0) 


    # load in state
    state = utils.restore_checkpoint(checkpoint_meta_dir, state, device)
    initial_step = int(state['step'])

    
    # load in tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    # Build data iterators
    train_ds, eval_ds = data.get_dataloaders(cfg)
    p = data.get_distribution(cfg.data)

    if isinstance(p, np.ndarray):
        p = torch.tensor(p, device=device).float()

    # mprint(f"Length of datasets: {len(train_ds)}, {len(eval_ds)}")

    train_iter = iter(train_ds)
    eval_iter = iter(eval_ds)

    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(cfg)
    if "mutinfo_config" in cfg:
        mutinfo_config = cfg.mutinfo_config
    else:
        mutinfo_config = None

    if cfg.cond is not None:
        input_ids = torch.tensor(cfg.cond.input_ids, device=device).long()
        input_locs = torch.tensor(cfg.cond.input_locs, device=device).long()
        
        def proj_fun(x):
            x[:, input_locs] = input_ids
            return x
        
        dim = input_ids.item()
        index = input_locs
        # Marginalize p(x) over the conditioned indices
        print(f"p before: {p}")
        print(f"Entropy of p before: {-(p * torch.log(p)).sum()}")

        p_cond = torch.index_select(p, dim, index)
        p_cond /= p_cond.sum()
        print(f"p after conditioning: {p_cond}")
        print(f"Entropy of p after conditioning: {-(p_cond * torch.log(p_cond)).sum()}")
        indeces_to_discard = list(input_locs)
        indeces_to_keep = [i for i in range(cfg.model.length) if i not in indeces_to_discard]
        # indeces_to_keep = None
        # p = p_cond.expand(*p.shape)
        print(f"New p: {p}")
    else:
        proj_fun = lambda x: x
        if p is not None:
            p_joint = np.squeeze(p)

            if len(p_joint.shape) > 1:
            
                px = p_joint.sum(dim=1)
                py = p_joint.sum(dim=0)
                p_marg = px.unsqueeze(1) * py
                print(f"Joint p: {np.squeeze(p)}")
                print(f"Mutual information of p: {torch.sum(p_joint * torch.log(p_joint / p_marg))}")
                print(f"Marginalized p: {p_marg}")

            print(f"Entropy of p: {-(p * torch.log(p)).sum()}")
            # p = p/p.sum(dim=1, keepdim=True)
        indeces_to_keep = None
    
    if p is not None:
        p_joint = p.clone()
        joint_score_fn = lambda x, sigma: graph.get_analytic_score(x, p_joint, sigma)
        px = torch.sum(p, axis=1)
        py = torch.sum(p, axis=0)
        pxy_margin = px * py.T
        pxy_margin = pxy_margin.unsqueeze(-1)
        marginal_score_fn = lambda x, sigma: graph.get_analytic_score(x, pxy_margin, sigma)
    else:
        joint_score_fn = None
        marginal_score_fn = None

    if not cfg.use_analytic_score:
        p = None
    
    train_step_fn = losses.get_step_fn(noise, graph, True, optimize_fn, cfg.training.accum, mutinfo_config, marginal_score_fn=marginal_score_fn, joint_score_fn=joint_score_fn, debug=cfg.debug)
    eval_step_fn = losses.get_step_fn(noise, graph, False, optimize_fn, cfg.training.accum, mutinfo_config, marginal_score_fn=marginal_score_fn, joint_score_fn=joint_score_fn, debug=cfg.debug)
    
    if cfg.training.snapshot_sampling:
        sampling_shape = (cfg.training.batch_size // (cfg.ngpus * cfg.training.accum), cfg.model.length)
        sampling_fn = sampling.get_sampling_fn(cfg, graph, noise, sampling_shape, sampling_eps, device, p)
        entropy_estimate_fn = sampling.get_entropy_estimate_fn(cfg, graph, noise, sampling_shape, sampling_eps, device, p, proj_fun, indeces_to_keep)
        entropy_estimate_montecarlo_fn = sampling.get_entropy_montecarlo_estimate_fn(cfg, graph, noise, sampling_shape, sampling_eps, device, p)
        entropy_estimate_dynkin_fn = sampling.get_entropy_dynkin_estimate_fn(cfg, graph, noise, sampling_shape, sampling_eps, device, p, proj_fun, indeces_to_keep)
        mutinfo_estimate_fn = sampling.get_mutinfo_estimate_fn(cfg, graph, noise, sampling_shape, sampling_eps, device, p, proj_fun, indeces_to_keep)
        mutinfo_estimate_dynkin_fn = sampling.get_mutinfo_dynkin_estimate_fn(cfg, graph, noise, sampling_shape, sampling_eps, device, p, proj_fun, indeces_to_keep)

    num_train_steps = cfg.training.n_iters
    mprint(f"Starting training loop at step {initial_step}.")

    if cfg.use_analytic_score:
        if cfg.estimate_entropy:
            if cfg.montecarlo:
                entropy_estimate = entropy_estimate_montecarlo_fn(score_model, train_ds)
            elif cfg.dynkin:
                entropy_estimate = entropy_estimate_dynkin_fn(score_model, train_ds)
                print("Dynkin estimate: ", entropy_estimate)
            else:
                entropy_estimate = entropy_estimate_fn(score_model)
        if cfg.estimate_mutinfo:
            if cfg.dynkin:
                mutinfo_estimate = mutinfo_estimate_dynkin_fn(score_model, train_ds)
                print("Dynkin estimate: ", mutinfo_estimate)
            else:
                mutinfo_estimate = mutinfo_estimate_fn(score_model)
                print("Mutual Information estimate: ", mutinfo_estimate)
        sample = sampling_fn(score_model)
        hist = np.zeros((torch.max(sample)+1, torch.max(sample)+1))
        for s in sample:
            hist[s[0], s[1]] += 1
        hist = hist / hist.sum()
        print(f"Generated samples: {sample}, with shape: {sample.shape}")
        print(f"Histogram of samples: {hist}")

    while state['step'] < num_train_steps + 1 and not cfg.use_analytic_score:
        step = state['step']
        if cfg.data.train != "text8" and cfg.data.train not in available_distributions:
            batch = next(train_iter)['input_ids'].to(device)
        else:
            if cfg.data.train in available_distributions:
                batch = next(train_iter)["feature"].to(device)
            else:
                batch = next(train_iter).to(device)
        # raise UserWarning(f"Batch shape: {batch.shape}")
        # Batch shape is (batch_size, seq_len)
        # print(f"Batch shape: {batch.shape}")

        loss = train_step_fn(state, batch)

        # flag to see if there was movement ie a full batch got computed
        if step != state['step']:
            if step % cfg.training.log_freq == 0:
                dist.all_reduce(loss)
                loss /= world_size

                mprint("step: %d, training_loss: %.5e" % (step, loss.item()))
            
            if step % cfg.training.snapshot_freq_for_preemption == 0 and rank == 0:
                utils.save_checkpoint(checkpoint_meta_dir, state)

            if step % cfg.training.eval_freq == 0:
                if cfg.data.train != "text8" and cfg.data.train not in available_distributions:
                    eval_batch = next(eval_iter)['input_ids'].to(device)
                else:
                    if cfg.data.train in available_distributions:
                        eval_batch = next(train_iter)["feature"].to(device)
                    else:
                        eval_batch = next(train_iter).to(device)
                eval_loss = eval_step_fn(state, eval_batch)

                dist.all_reduce(eval_loss)
                eval_loss /= world_size

                mprint("step: %d, evaluation_loss: %.5e" % (step, eval_loss.item()))

            if step > 0 and step % cfg.training.snapshot_freq == 0 or step == num_train_steps:
                # Save the checkpoint.
                save_step = step // cfg.training.snapshot_freq
                if rank == 0:
                    utils.save_checkpoint(os.path.join(
                        checkpoint_dir, f'checkpoint_{save_step}.pth'), state)

                # Generate and save samples
                if cfg.training.snapshot_sampling:
                    mprint(f"Eval at step: {step}")

                    this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
                    utils.makedirs(this_sample_dir)

                    ema.store(score_model.parameters())
                    ema.copy_to(score_model.parameters())
                    if cfg.estimate_entropy:
                        if cfg.montecarlo:
                            entropy_estimate = entropy_estimate_montecarlo_fn(score_model, train_ds)
                        elif cfg.dynkin:
                            entropy_estimate = entropy_estimate_dynkin_fn(score_model, train_ds)
                        else:
                            entropy_estimate = entropy_estimate_fn(score_model)
                    if cfg.estimate_mutinfo:
                        if cfg.dynkin:
                            mutinfo_estimate = mutinfo_estimate_dynkin_fn(score_model, train_ds)
                            print("Dynkin estimate: ", mutinfo_estimate)
                        else:
                            mutinfo_estimate = mutinfo_estimate_fn(score_model)
                            print("Mutual Information estimate: ", mutinfo_estimate)
                    sample = sampling_fn(score_model)
                    hist = np.zeros((torch.max(sample)+1, torch.max(sample)+1))
                    for s in sample:
                        hist[s[0], s[1]] += 1
                    hist = hist / hist.sum()
                    print(f"Generated samples: {sample}, with shape: {sample.shape}")
                    print(f"Histogram of samples: {hist}")
                    ema.restore(score_model.parameters())

                    if cfg.estimate_entropy:
                        mprint(f"Entropy estimated: {entropy_estimate.item()}")
                        results["entropy_estimate"] = entropy_estimate.item()
                    if cfg.estimate_mutinfo:
                        mprint(f"Mutual Information estimated: {mutinfo_estimate.item()}")
                        results["mutinfo_estimate"] = mutinfo_estimate.item()

                    dist.barrier()
    
    json.dump(results, open(os.path.join(work_dir, "results.json"), "w"))

