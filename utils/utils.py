import subprocess
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

from utils.config import all_results


def Uniform(X: np.array):
    """
    This function generates a random point in the feasible region X. We assume that X is a subset of R^n
    described by the inequalities X = {x in R^n | a_i <= x_i <= b_i, i = 0, ..., m-1} where a_i, b_i are given
    such that X[i,j] = [a_i, b_i] for i = 0, ..., m-1 and j = 0, 1.
    For simplicity, we assume that X C Rectangle given by an infinite norm (i.e. X = {x in R^n | -M <= x_i <= M, i = 1, ..., n}).
    X: feasible region (numpy array)
    """

    theta = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        theta[i] = np.random.uniform(X[i, 0], X[i, 1])
    return theta


def Bernoulli(p: float):
    """
    This function generates a random variable following a Bernoulli distribution.
    p: probability of success (float)
    """
    a = np.random.uniform(0, 1)
    if a <= p:
        return 1
    else:
        return 0


def growth_condition(last_nb_samples, max_slope):
    """
    Check if the slope of the last points of the nb_samples vs nb_evaluations curve
    is greater than max_slope.
    """
    slope = (last_nb_samples[-1] - last_nb_samples[0]) / (len(last_nb_samples) - 1)
    return slope > max_slope

def slope_stop_condition(last_nb_samples, max_slope):
    """
    Check if the slope of the last points of the nb_samples vs nb_evaluations curve
    is greater than max_slope.
    """
    slope = (last_nb_samples[-1] - last_nb_samples[0]) / (len(last_nb_samples) - 1)
    return slope > max_slope




def LIPO_condition(x, values, k, points, strict=False):
    """
    Subfunction to check the condition in the loop, depending on the set of values we already have.
    values: set of values of the function we explored (numpy array)
    x: point to check (numpy array)
    k: Lipschitz constant (float)
    points: set of points we have explored (numpy array)
    """
    max_val = np.max(values)
    left_min = np.min(
        values.reshape(-1) + k * np.linalg.norm(x - points, ord=2, axis=1)
    )

    if strict:
        # print('left_min', left_min)
        # print('max_val', max_val)
        # print(left_min > max_val)
        return left_min > max_val
    else:
        return left_min >= max_val


def compute_epsilon(values, bound, t):
    """
    Subfunction to check the condition in the loop, depending on the set of values we already have.
    values: set of values of the function we explored (numpy array)
    x: point to check (numpy array)
    k: Lipschitz constant (float)
    points: set of points we have explored (numpy array)
    """
    max_val = np.max(values)
    # val = [v for v in values if v != max_val]
    # min_val = np.max(val)
    #
    min_val = np.min(values)

    pairwise_distances = pdist(bound)

    diam = np.max(pairwise_distances)
    d = 2
    print("the diameter is ", diam)
    # (t ** (1 / d))
    # 2 * ((max_val - min_val) / diam)
    return ((max_val - min_val) / diam)


def compute_sup_epsilon(values, points, bounds, prev_epsilon):
    a, b = bounds
    a1 = a[0]
    a2 = a[1]
    b1 = b[0]
    b2 = b[1]

    print(a1, a2, b1, b2)

    max_f_value = np.max(values)
    # print("max_f_value ", max_f_value)
    # poi = [points[i] for i in range(len(points)) if values[i] != max_f_value]

    sup_distances = np.array([
        np.max([
            np.linalg.norm(np.array([a1, a2]) - point),
            np.linalg.norm(np.array([a1, b2]) - point),
            np.linalg.norm(np.array([b1, a2]) - point),
            np.linalg.norm(np.array([b1, b2]) - point)
        ]) for point in points
    ])

    print("sup_distances ", sup_distances)
    sup_distances = sup_distances.flatten()
    expression_values = np.array([
        (max_f_value - values[i]) / (0.2*sup_distances[i])
        for i in range(len(values))
    ])


    min_index = np.argmax(expression_values)
    print("max_f_value: ", (max_f_value))
    print("decided_value_above: ", (values[min_index]))
    print("decided_point: ", (points[min_index]))
    print("decided_distance_below: ", (sup_distances[min_index]))
    # print("max points: ", [points[i] for i in range(len(points)) if values[i] == max_f_value], "max value: ", max_f_value)
    # print("min points: ", [points[i] for i in range(len(points)) if values[i] == np.min(values)], "min value: ", np.min(values))

    # print("epsilon is ", expression_values[min_index])

    return max(expression_values[min_index], prev_epsilon)


def run_main_for_functions(function_names, evals=50, reps=1000, optimizers=['ECP']):
    for optimizer in optimizers:
        for function_name in function_names:
            try:
                # Construct the command to run main.py with the desired function name
                command = f"python main.py --function {function_name} -n {evals} -r {reps} -o {optimizer}"

                # Execute the command and merge stdout and stderr
                print(f"Running: {command}")
                process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True)

                # Initialize an empty string to capture the output
                output = ""

                # Print all output line-by-line
                for line in process.stdout:
                    print(line, end='')  # Print the output line by line
                    output += line  # Capture the output

                # Wait for the process to complete
                process.wait()

                # Check the return code
                if process.returncode != 0:
                    print(f"Error running {command}: Return code {process.returncode}")
                else:
                    # Here you can parse the output to extract the results
                    # Assuming the result is stored in the output in a specific format
                    # For example, let's say the output contains a line like:
                    # "Result for rosenbrock: 2.00 (1.00)"
                    # You will need to adapt this part based on your actual output format.
                    # result_line = f"Result for {function_name}:"
                    result_line = f"Best value found:"

                    if result_line in output:
                        # Extract the result (adjust the logic according to your output)
                        start_index = output.index(result_line) + len(result_line)
                        end_index = output.index("\n", start_index)
                        result_value = output[start_index:end_index].strip()

                        # Store the result in the all_results dictionary
                        all_results[function_name][optimizer] = result_value



            except Exception as e:
                print(f"Exception occurred: {e}")

    # Convert the nested dictionary to a DataFrame
    df = pd.DataFrame.from_dict(all_results, orient='index')

    # Save the DataFrame to a CSV file
    df.to_csv(f'results/output_{evals}_{reps}_perm_20.csv')