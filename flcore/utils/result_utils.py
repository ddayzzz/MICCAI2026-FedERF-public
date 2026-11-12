from collections import defaultdict

import h5py
import numpy as np
import os


def average_data(metric_names, algorithm="", dataset="", goal="", times=10, experiment_sub_dirname=""):
    test_acc = get_all_results_for_one_algo(metric_names, algorithm, dataset, goal, times, experiment_sub_dirname)

    for metric_name, metric_values in test_acc.items():
        max_accuracy = []
        latest_values = []
        for i in range(times):
            if 'loss' in metric_name:
                max_accuracy.append(test_acc[metric_name][i].min())
            elif 'acc' in metric_name:
                max_accuracy.append(test_acc[metric_name][i].max())
            elif 'auroc' in metric_name:
                max_accuracy.append(test_acc[metric_name][i].max())
            elif 'c_index' in metric_name:
                max_accuracy.append(test_acc[metric_name][i].max())
            else:
                max_accuracy.append(test_acc[metric_name][i].max())
            latest_values.append(test_acc[metric_name][i][-1])

        print(f"std for best {metric_name}:", np.std(max_accuracy))
        print(f"mean for best {metric_name}:", np.mean(max_accuracy))

        print(f"std for latest {metric_name}:", np.std(latest_values))
        print(f"mean for latest {metric_name}:", np.mean(latest_values))


def get_all_results_for_one_algo(metric_names, algorithm="", dataset="", goal="", times=10, experiment_sub_dirname=""):
    test_acc = defaultdict(list)
    algorithms_list = [algorithm] * times
    for i in range(times):
        file_name = algorithms_list[i] + "_" + goal + "_fold" + str(i)
        # test_acc.append(np.array())
        metrics = read_data_then_delete(file_name, metric_names=metric_names, experiment_sub_dirname=experiment_sub_dirname, delete=False)
        for k, v in metrics.items():
            test_acc[k].append(v)
    return test_acc


def read_data_then_delete(file_name, metric_names, experiment_sub_dirname="", delete=False):
    file_path = f"./fl_results/{experiment_sub_dirname}" + file_name + ".h5"
    rs_tests = dict()
    with h5py.File(file_path, 'r') as hf:
        # rs_test_acc = np.array(hf.get('rs_test_acc'))
        for k in metric_names:
            rs_tests[k] = np.array(hf.get(f'rs_test_{k}'))

    if delete:
        os.remove(file_path)
    print("Length: ", len(rs_tests[metric_names[0]]))

    return rs_tests