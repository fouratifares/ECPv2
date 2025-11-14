import argparse
import csv
import os
import numpy as np
import time
from tqdm import tqdm


def ablation(
        n_run: int,
        n_eval: int,
        f,
        args,
        constant_ablation=True,
        tau_ablation=True,
        epsilon_ablation= True,
):
    """
    For ablation study of ECP
    n_run: number of runs (int)
    n_eval: number of function evaluations (int)
    f: function class to maximize (class)
    optimizer: optimizer function (function)
    method: name of the optimizer (str)
    delta: with proba 1-delta, the bounds holds (float)
    p: probability of success (float)
    fig_path: path to save the statistics figures (str)
    """
    method = "ECP"
    print(f"Method: {method}")

    # print chunk results
    chunk = 1
    dic_chunk = {}
    for i in range(chunk, n_eval + 1, chunk):
        dic_chunk[i] = []

    times = []
    vs = []
    nb_evals = []

    # constant C ablation
    print('Running the Constant C ablation')
    if constant_ablation:
        constants = [1, 10, 100, 1000, 2000, 4000, 6000, 8000]
        for i in tqdm(range(len(constants))):
            constant = constants[i]
            # print(constant)
            for _ in range(n_run):
                start_time = time.time()
                points, values, epsilons = ECP(f, n=n_eval, max_slope=constant)
                times.append(time.time() - start_time)

                for i in range(chunk, n_eval + 1, chunk):
                    chunk_values = values[:i + 1]
                    if chunk_values.size > 0:
                        dic_chunk[i].append(np.max(chunk_values))
                    # else:
                    #     # Handle the case where chunk_values is empty
                    #     dic_chunk[i].append(None)  # or some other default value like -np.inf

                vs.append(np.max(values))
                nb_evals.append(len(values))

            with open(f"results/results_{method}.txt", "a") as file:  # Open the file in append mode
                file.write(f"Method: {method}\n with C = {constant}")
                for i in range(chunk, n_eval + 1, chunk):
                    file.write(
                        f"Average of best maximum until {i:.2f}: {np.mean(dic_chunk[i]):.5f}, std: {np.std(dic_chunk[i]):.5f}\n")
                # file.write(f"Number of final function evaluations: {np.mean(nb_evals):.2f}\n")
                # file.write(f"Average of best maximum: {np.mean(vs):.2f} ({np.std(vs):.2f})\n")
                file.write(f"Average time in  seconds: {np.mean(times):.5f}")
                file.write("\n\n")

            # Save results to a CSV file
            # Create the filename and directory
            directory = "figures/functions/" + args.function + '/ablation_c/'
            name_file = directory + method + "(C=" + str(constant) + ")_results.csv"

            # Create the directory if it doesn't exist
            os.makedirs(directory, exist_ok=True)
            # name_file = "figures/functions/" + args.function + '/ablation_c/' + method + "(C =" + str(
            #     constant) + ")_results.csv"
            with open(name_file, mode="w", newline="") as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(["i", "Mean", "StdDev"])  # Header row

                for i in range(chunk, n_eval + 1, chunk):
                    mean_value = np.mean(dic_chunk[i])
                    std_dev_value = np.std(dic_chunk[i])
                    csv_writer.writerow([i, mean_value, std_dev_value])  # Write data rows

    # constant tau ablation
    print('Running the tau ablation')
    if tau_ablation:
        taus = [1.001, 1.01, 1.1, 1.2, 1.4, 1.6, 1.8, 2]
        for i in tqdm(range(len(taus))):
            tau = taus[i]
            for _ in range(n_run):
                start_time = time.time()
                points, values, epsilons = ECP(f, n=n_eval, tau_=tau)
                times.append(time.time() - start_time)

                for i in range(chunk, n_eval + 1, chunk):
                    chunk_values = values[:i + 1]
                    if chunk_values.size > 0:
                        dic_chunk[i].append(np.max(chunk_values))
                    # else:
                    #     # Handle the case where chunk_values is empty
                    #     dic_chunk[i].append(None)  # or some other default value like -np.inf

                vs.append(np.max(values))
                nb_evals.append(len(values))

            with open(f"results/results_{method}.txt", "a") as file:  # Open the file in append mode
                file.write(f"Method: {method}\n with tau = {tau} and C = 500")
                for i in range(chunk, n_eval + 1, chunk):
                    file.write(
                        f"Average of best maximum until {i:.2f}: {np.mean(dic_chunk[i]):.5f}, std: {np.std(dic_chunk[i]):.5f}\n")
                # file.write(f"Number of final function evaluations: {np.mean(nb_evals):.2f}\n")
                # file.write(f"Average of best maximum: {np.mean(vs):.2f} ({np.std(vs):.2f})\n")
                file.write(f"Average time in  seconds: {np.mean(times):.5f}")
                file.write("\n\n")

            # Save results to a CSV file
            directory = "figures/functions/" + args.function + '/ablation_tau/'
            name_file = directory + method + "(tau=" + str(
                tau) + ")_results.csv"

            # Create the directory if it doesn't exist
            os.makedirs(directory, exist_ok=True)
            # name_file = "figures/functions/" + args.function + '/ablation_tau/' + method + "(tau=" + str(
            # tau) + ")_results.csv"
            with open(name_file, mode="w", newline="") as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(["i", "Mean", "StdDev"])  # Header row

                for i in range(chunk, n_eval + 1, chunk):
                    mean_value = np.mean(dic_chunk[i])
                    std_dev_value = np.std(dic_chunk[i])
                    csv_writer.writerow([i, mean_value, std_dev_value])  # Write data rows

        # constant epsilon ablation
    print('Running the epsilon ablation')
    if epsilon_ablation:
        epsilon = [1.0001, 1.001, 1.01, 1.1, 1.5]
        for i in tqdm(range(len(epsilon))):
            ep = epsilon[i]
            for _ in range(n_run):
                start_time = time.time()
                points, values, epsilons = ECP(f, n=n_eval, epsilon=ep)
                times.append(time.time() - start_time)

                for i in range(chunk, n_eval + 1, chunk):
                    chunk_values = values[:i + 1]
                    if chunk_values.size > 0:
                        dic_chunk[i].append(np.max(chunk_values))
                    # else:
                    #     # Handle the case where chunk_values is empty
                    #     dic_chunk[i].append(None)  # or some other default value like -np.inf

                vs.append(np.max(values))
                nb_evals.append(len(values))

            with open(f"results/results_{method}.txt", "a") as file:  # Open the file in append mode
                file.write(f"Method: {method}\n with epsilon = {ep} and C = 1000")
                for i in range(chunk, n_eval + 1, chunk):
                    file.write(
                        f"Average of best maximum until {i:.2f}: {np.mean(dic_chunk[i]):.5f}, std: {np.std(dic_chunk[i]):.5f}\n")
                # file.write(f"Number of final function evaluations: {np.mean(nb_evals):.2f}\n")
                # file.write(f"Average of best maximum: {np.mean(vs):.2f} ({np.std(vs):.2f})\n")
                file.write(f"Average time in  seconds: {np.mean(times):.5f}")
                file.write("\n\n")

            # Save results to a CSV file
            directory = "figures/functions/" + args.function + '/ablation_epsilon/'
            name_file = directory + method + "(epsilon=" + str(ep) + ")_results.csv"

            # Create the directory if it doesn't exist
            os.makedirs(directory, exist_ok=True)
            # name_file = "figures/functions/" + args.function + '/ablation_tau/' + method + "(tau=" + str(
            # tau) + ")_results.csv"
            with open(name_file, mode="w", newline="") as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(["i", "Mean", "StdDev"])  # Header row

                for i in range(chunk, n_eval + 1, chunk):
                    mean_value = np.mean(dic_chunk[i])
                    std_dev_value = np.std(dic_chunk[i])
                    csv_writer.writerow([i, mean_value, std_dev_value])  # Write data rows

    return points, values
