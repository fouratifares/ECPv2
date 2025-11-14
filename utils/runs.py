import argparse
import csv
import os
import numpy as np
import time
from optimizers.AdaLIPO import AdaLIPO
from optimizers.AdaLIPO_P import AdaLIPO_P
from optimizers.PRS import PRS
# from optimizers.BayesOpt import BayesOpt
from optimizers.CMAES import CMAES
from optimizers.BayesOpt import BayesOpt
from optimizers.Direct import Direct
from optimizers.ECP import ECP
from optimizers.ECP_P import ECP_P
from optimizers.NeuralUCB import NeuralUCB
from optimizers.SHGO import SHGO
from optimizers.Brute import Brute
from optimizers.Dual_Annealing import Dual_Annealing
from optimizers.BayesOpt_botorch import botorch
from optimizers.Differential_Evolution import Differential_Evolution
from optimizers.ECP_plus import ECP_plus
from optimizers.ECPv2 import ECPv2
from optimizers.SMAC3 import SMAC3
from optimizers.berkenkamp import berkenkamp


# from utils.config import all_results


def convert_to_function(name):
    """
    Converts a list of function names (as strings) into their actual function references.

    Args:
    - func_names: List of strings representing function names.

    Returns:
    - List of corresponding function references.
    """
    # Debugging: Print the input to see what is being passed
    # print("Input func_names:", name)

    functions = []

    if name in globals():
        functions.append(globals()[name])
    else:
        raise ValueError(f"Function '{name}' is not defined or imported.")

    return functions


def cli():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--function", "-f", type=str, help="Function class to maximize", required=True
    )
    args.add_argument(
        "--n_eval", "-n", type=int, help="Number of function evaluations", required=True
    )
    args.add_argument("--n_run", "-r", type=int, help="Number of runs", required=True)
    args.add_argument("--optimizers_fct", "-o", type=str, help="list of optimizers", required=False, default=None)
    # args.add_argument(
    #     "--delta",
    #     "-delta",
    #     type=float,
    #     help="With proba 1-delta, the bounds are made",
    #     default=0.05,
    # )
    return args.parse_args()


def runs(
        n_run: int,
        n_eval: int,
        f,
        optimizer,
        method: str,
        args,
):
    """
    Run the optimizer several times and return the points and values of the last run.
    n_run: number of runs (int)
    n_eval: number of function evaluations (int)
    f: function class to maximize (class)
    optimizer: optimizer function (function)
    method: name of the optimizer (str)
    delta: with proba 1-delta, the bounds holds (float)
    p: probability of success (float)
    fig_path: path to save the statistics figures (str)
    """
    print(f"Method: {method}")

    # print chunk results
    chunk = 1
    dic_chunk = {}
    for i in range(chunk, n_eval + 1, chunk):
        dic_chunk[i] = []

    times = []
    vs = []
    nb_evals = []
    max_epsilon = []
    min_epsilon = []
    average_epsilon = []
    std_epsilon = []
    all_epsilons = []

    all_values = []
    for i in range(n_run):
        # print(f"This is rep number {i}")
        start_time = time.time()
        if method == "ECP" or method == "AdaLIPO" or method == "AdaLIPO_P":
            points, values, epsilons = optimizer(f, n=n_eval)
            max_epsilon.append(np.max(epsilons))
            min_epsilon.append(np.min(epsilons))
            average_epsilon.append(np.mean(epsilons))
            std_epsilon.append(np.std(epsilons))
            all_epsilons.append(epsilons)
            all_values.append(values)

        else:
            points, values = optimizer(f, n=n_eval)
            all_values.append(values)

        times.append(time.time() - start_time)

        for i in range(chunk, n_eval + 1, chunk):
            chunk_values = values[:i]
            if chunk_values.size > 0:
                dic_chunk[i].append(np.max(chunk_values))
            # else:
            #     # Handle the case where chunk_values is empty
            #     dic_chunk[i].append(None)  # or some other default value like -np.inf

        nb_evals.append(len(values))
        if len(values) > 0:
            vs.append(np.max(values))

    # for i in range(chunk, n_eval+1, chunk):
    #     print(f"Average of best maximum until {i:.2f}: {np.mean(dic_chunk[i]):.2f}, std: {np.std(dic_chunk[i]):.2f}")
    #
    # print(f"Number of f evaluations: {np.mean(nb_evals):.2f} +- {np.std(nb_evals):.2f}")
    # print(f"Average of best maximum: {np.mean(vs):.2f}, std: {np.std(vs):.2f}")
    # print(f"Average time in  seconds: {np.mean(times):.2f}")

    with open(f"results/results_{method}.txt", "a") as file:  # Open the file in append mode
        file.write(f"Method: {method}\n")
        for i in range(chunk, n_eval + 1, chunk):
            file.write(
                f"Average of best maximum until {i:.2f}: {np.mean(dic_chunk[i]):.5f}, std: {np.std(dic_chunk[i]):.5f}\n")
            if i == 50:
                file.write(
                    f"Average of best maximum until {i:.2f}: {np.mean(dic_chunk[i]):.2f} ({np.std(dic_chunk[i]):.2f})\n")
        file.write(f"Number of final function evaluations: {np.mean(nb_evals):.2f}\n")
        file.write(f"Average of best maximum: {np.mean(vs):.2f} ({np.std(vs):.2f})\n")

        best_r = f"{np.mean(vs):.2f} ({np.std(vs):.2f}) &"

        if method == "ECP" or method == "AdaLIPO" or method == "AdaLIPO_P":
            file.write(
                f"Average of maximum epsilons: {np.mean(np.array(max_epsilon)):.2f} ({np.std(np.array(max_epsilon)):.2f})\n")
            file.write(f"Max of maximum epsilons: {np.max(np.array(max_epsilon)):.2f} \n")
            file.write(f"Min of maximum epsilons: {np.min(np.array(max_epsilon)):.2f} \n")

        file.write(f"Average time in  seconds: {np.mean(times):.5f}")
        file.write("\n\n")

    if method == "ECP" or method == "AdaLIPO" or method == "AdaLIPO_P":
        # Convert the list to a NumPy array
        epsilons_array = np.array(all_epsilons)

        # Compute the average and standard deviation over dimension 0
        epsilons_avg = np.mean(epsilons_array, axis=0)
        epsilons_std = np.std(epsilons_array, axis=0)

        with open(f"results/results_{method}_epsilons.txt", "a") as file:  # Open the file in append mode\
            file.write(f"Method: {args.function}\n")
            file.write(
                f"Average of maximum epsilons: {np.mean(np.array(max_epsilon)):.2f} ({np.std(np.array(max_epsilon)):.2f})\n")
            file.write(f"Max of maximum epsilons: {np.max(np.array(max_epsilon)):.2f} \n")
            file.write(f"Min of maximum epsilons: {np.min(np.array(max_epsilon)):.2f} \n")
            file.write(f"Averages over runs: {np.array(epsilons_avg)} and std {np.array(epsilons_std)} \n")

    # Save results to a CSV file
    name_file = "figures/functions/" + args.function + '/' + method + "_results.csv"
    with open(name_file, mode="w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["i", "Mean", "StdDev"])  # Header row

        for i in range(chunk, n_eval + 1, chunk):
            mean_value = np.mean(dic_chunk[i])
            std_dev_value = np.std(dic_chunk[i])
            csv_writer.writerow([i, mean_value, std_dev_value])  # Write data rows

    if method == "ECP" or method == "AdaLIPO" or method == "AdaLIPO_P":
        return points, values, epsilons, best_r
    else:
        return points, values, best_r


def directories(args):
    # Add this for newly added algorithms
    # if not os.path.exists(f"figures/functions/{args.function}/ECP_P"):
    #     os.mkdir(f"figures/functions/{args.function}/ECP_P")
    # if not os.path.exists(f"figures/functions/{args.function}/SHGO"):
    #     os.mkdir(f"figures/functions/{args.function}/SHGO")

    # Always
    if not os.path.exists("results/"):
        os.mkdir(f"results/")

    if not os.path.exists("figures/"):
        os.mkdir(f"figures/")

    if not os.path.exists("figures/functions/"):
        os.mkdir(f"figures/functions/")

    if not os.path.exists(f"figures/functions/{args.function}"):
        os.mkdir(f"figures/functions/{args.function}")
        os.mkdir(f"figures/functions/{args.function}/PRS")
        os.mkdir(f"figures/functions/{args.function}/Brute")
        os.mkdir(f"figures/functions/{args.function}/BayesOpt")
        os.mkdir(f"figures/functions/{args.function}/AdaLIPO")
        os.mkdir(f"figures/functions/{args.function}/AdaLIPO_P")
        os.mkdir(f"figures/functions/{args.function}/ECP")
        os.mkdir(f"figures/functions/{args.function}/CMAES")
        os.mkdir(f"figures/functions/{args.function}/Direct")
        os.mkdir(f"figures/functions/{args.function}/NeuralUCB")
        os.mkdir(f"figures/functions/{args.function}/ECP_P")
        os.mkdir(f"figures/functions/{args.function}/SHGO")
        os.mkdir(f"figures/functions/{args.function}/BayesOpt_botorch")
        os.mkdir(f"figures/functions/{args.function}/Dual_Annealing")
        os.mkdir(f"figures/functions/{args.function}/Differential_Evolution")
        os.mkdir(f"figures/functions/{args.function}/ECP_plus")
        os.mkdir(f"figures/functions/{args.function}/ECPv2")

        os.mkdir(f"figures/functions/{args.function}/ablation_tau/")
        os.mkdir(f"figures/functions/{args.function}/ablation_c/")
