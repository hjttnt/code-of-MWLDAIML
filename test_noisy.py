import numpy as np
import heapq
import pandas as pd
import random
import time
import MWLDAIML
from Params import Params

def load_csv(path):
    data_read = pd.read_csv(path)
    list = data_read.values.tolist()
    data = np.array(list)
    return data

def add_noise_to_X(X, noise_type, noise_level=0.1, **kwargs):
    """
    noise_type: "uniform"
    kwargs:
        low in [-0.1,-0.2,-0.3], high in [0.1,0.2,0.3]
    """
    X_noisy = X.copy().astype(float)
    n, d = X.shape
    rng = np.random.default_rng(kwargs.get("seed", None))

    if noise_type == "uniform":
        r = noise_level
        low = kwargs.get("low", -r)
        high = kwargs.get("high", r)
        noise = rng.uniform(low, high, size=(n, d))
        X_noisy += noise

    return X_noisy


class Label:
    def __init__(self, idx=0, rest_number=0, target_number=0):
        self.idx = idx
        self.rest_number = rest_number
        self.target_number = target_number

    def __lt__(self, other):
        # Custom priority for heapq
        if (self.rest_number - self.target_number) != (other.rest_number - other.target_number):
            return (self.rest_number - self.target_number) < (other.rest_number - other.target_number)
        elif self.target_number != other.target_number:
            return self.target_number < other.target_number
        else:
            return id(self) < id(other)

times = 5
Data_name = ["birds", "CAL500", "Emotions", "flags", "genbase", "IMA", "medical", "enron", "scene", "bibtex", "Corel5k",
             "corel16k001", "corel16k002", "corel16k003", "corel16k004", "corel16k005", "corel16k006", "corel16k007",
             "corel16k008", "corel16k009", "corel16k010"]
param_name = ['alpha', 'beta', 'lamda1', 'lamda2', 'lamda3', 'rou', 'rou1', 'rou2', 'rou3', 'miu', 'miu1',
              'miu2', 'miu3', 'yuzhi']
noise_types = ["uniform"]
noise_rates = [0.1,0.2,0.3]

for data in Data_name:
    file_name = data
    is_fix = True
    if data in ["CAL500", "IMA"]:
        is_fix = False

    print(data)
    param = {}
    for paramName in param_name:
        param[paramName] = Params[data][paramName]
    for noise_rate in noise_rates:
        print(noise_rate)
        for noise_type in noise_types:
            print(noise_type)

            MWLDAIML_hamming_loss = list()
            MWLDAIML_coverage = list()
            MWLDAIML_ranking_loss = list()
            MWLDAIML_one_error = list()
            MWLDAIML_average_precision = list()
            MWLDAIML_subset_accuracy = list()
            MWLDAIML_accuracy = list()
            MWLDAIML_precision = list()
            MWLDAIML_recall = list()
            MWLDAIML_f1 = list()
            MWLDAIML_auc = list()
            MWLDAIML_macro_averaging_accuracy = list()
            MWLDAIML_macro_averaging_precision = list()
            MWLDAIML_macro_averaging_recall = list()
            MWLDAIML_macro_averaging_f1 = list()
            MWLDAIML_macro_averaging_auc = list()
            MWLDAIML_micro_averaging_accuracy = list()
            MWLDAIML_micro_averaging_precision = list()
            MWLDAIML_micro_averaging_recall = list()
            MWLDAIML_micro_averaging_f1 = list()
            MWLDAIML_micro_averaging_auc = list()
            MWLDAIML_time = list()

            if is_fix:
                X_origin = load_csv(r"D:\Datasets\Mulan_datasets\{}\{}_train_features.csv".format(file_name, file_name))
                X = add_noise_to_X(X_origin, noise_type, noise_level=noise_rate)
                Y = load_csv(r"D:\Datasets\Mulan_datasets\{}\{}_train_labels.csv".format(file_name, file_name))
                X_test = load_csv(r"D:\Datasets\Mulan_datasets\{}\{}_test_features.csv".format(file_name, file_name))
                Y_true = load_csv(r"D:\Datasets\Mulan_datasets\{}\{}_test_labels.csv".format(file_name, file_name))
            else:
                Features = load_csv(r"D:\Datasets\Dataset_csv\{}\{}_features.csv".format(file_name, file_name))
                Labels = load_csv(r"D:\Datasets\Dataset_csv\{}\{}_labels.csv".format(file_name, file_name))
            for t in range(times):
                start_time = time.time()
                print(t)
                if not is_fix:
                    Labels[Labels == -1] = 0
                    (n0, l) = Labels.shape
                    k0 = 0.8
                    #
                    help = [row[:] for row in Labels.tolist()]
                    rest_number_list = list(Labels.sum(axis=0))
                    target_number_list = []
                    lab = {}

                    l = []  # Use a list for heapq

                    for i in range(len(rest_number_list)):
                        target_number = int((rest_number_list[i] + 1) * k0)
                        target_number_list.append(target_number)
                        lab[i] = Label(i, rest_number_list[i], target_number)
                        heapq.heappush(l, lab[i])

                    sample_idx_set = set()
                    while l:
                        heapq.heapify(l)
                        log = []
                        top_label = l[0]
                        id_label = top_label.idx

                        for i in range(len(help)):
                            if help[i][id_label] == 1:
                                log.append(i)

                        while top_label.target_number > 0:
                            if len(log) != top_label.rest_number:
                                print("wrong")

                            index = random.randint(0, top_label.rest_number - 1)
                            get_sample_idx = log[index]
                            sample_idx_set.add(get_sample_idx)

                            for j in range(len(help[get_sample_idx])):
                                if help[get_sample_idx][j] == 1:
                                    lab[j].rest_number -= 1
                                    lab[j].target_number -= 1
                                    help[get_sample_idx][j] = 0

                            log.pop(index)

                        if log:
                            for leave_sample_idx in log:
                                for j in range(len(help[leave_sample_idx])):
                                    if help[leave_sample_idx][j] == 1:
                                        lab[j].rest_number -= 1
                                        help[leave_sample_idx][j] = 0
                        heapq.heappop(l)

                    f = list(sample_idx_set)
                    X_origin = Features[f, :]
                    X = add_noise_to_X(X_origin, noise_type, noise_level=noise_rate)
                    Y = Labels[f, :]
                    c = set(range(n0))
                    g = list(c - sample_idx_set)
                    X_test = Features[g, :]
                    Y_true = Labels[g, :]

                time_start_MWLDAIML = time.time()
                MWLDAIML_scorce = MWLDAIML.MWLDAIML(X, Y, X_test, Y_true, True, miu_max=1e1000, miu1_max=1e1000,
                                                    miu2_max=1e1000, miu3_max=1e1000, **param)
                MWLDAIML_time.append(time.time() - time_start_MWLDAIML)
                print("time:", t, " finishi MWLDAIML")
                MWLDAIML_hamming_loss.append(MWLDAIML_scorce["hamming_loss"])
                MWLDAIML_coverage.append(MWLDAIML_scorce["coverage"])
                MWLDAIML_ranking_loss.append(MWLDAIML_scorce["ranking_loss"])
                MWLDAIML_one_error.append(MWLDAIML_scorce["one_error"])
                MWLDAIML_average_precision.append(MWLDAIML_scorce["average_precision"])
                MWLDAIML_subset_accuracy.append(MWLDAIML_scorce["subset_accuracy"])
                MWLDAIML_accuracy.append(MWLDAIML_scorce["accuracy"])
                MWLDAIML_precision.append(MWLDAIML_scorce["precision"])
                MWLDAIML_recall.append(MWLDAIML_scorce["recall"])
                MWLDAIML_f1.append(MWLDAIML_scorce["f1"])
                MWLDAIML_auc.append(MWLDAIML_scorce["auc"])
                MWLDAIML_macro_averaging_accuracy.append(MWLDAIML_scorce["macro_averaging_accuracy"])
                MWLDAIML_macro_averaging_precision.append(MWLDAIML_scorce["macro_averaging_precision"])
                MWLDAIML_macro_averaging_recall.append(MWLDAIML_scorce["macro_averaging_recall"])
                MWLDAIML_macro_averaging_f1.append(MWLDAIML_scorce["macro_averaging_f1"])
                MWLDAIML_macro_averaging_auc.append(MWLDAIML_scorce["macro_averaging_auc"])
                MWLDAIML_micro_averaging_accuracy.append(MWLDAIML_scorce["micro_averaging_accuracy"])
                MWLDAIML_micro_averaging_precision.append(MWLDAIML_scorce["micro_averaging_precision"])
                MWLDAIML_micro_averaging_recall.append(MWLDAIML_scorce["micro_averaging_recall"])
                MWLDAIML_micro_averaging_f1.append(MWLDAIML_scorce["micro_averaging_f1"])
                MWLDAIML_micro_averaging_auc.append(MWLDAIML_scorce["micro_averaging_auc"])

            file = open("noisy_result\MWLDAIML_{}_{}_{}_noisy_result.txt".format(file_name, noise_type, noise_rate), "a")
            file.write("\n")

            file.write(" time: " + str(np.mean(np.array(MWLDAIML_time)).item()) + " +- ")
            file.write(str(np.std(np.array(MWLDAIML_time), ddof=1).item()) + "\n")

            file.write(" hamming_loss: " + str(np.mean(np.array(MWLDAIML_hamming_loss)).item()) + " +- ")
            file.write(str(np.std(np.array(MWLDAIML_hamming_loss), ddof=1).item()) + "\n")

            file.write(" one_error: " + str(np.mean(np.array(MWLDAIML_one_error)).item()) + " +- ")
            file.write(str(np.std(np.array(MWLDAIML_one_error), ddof=1).item()) + "\n")

            file.write(" coverage: " + str(np.mean(np.array(MWLDAIML_coverage)).item()) + " +- ")
            file.write(str(np.std(np.array(MWLDAIML_coverage), ddof=1).item()) + "\n")

            file.write(" average_precision: " + str(np.mean(np.array(MWLDAIML_average_precision)).item()) + " +- ")
            file.write(str(np.std(np.array(MWLDAIML_average_precision), ddof=1).item()) + "\n")

            file.write(" ranking_loss: " + str(np.mean(np.array(MWLDAIML_ranking_loss)).item()) + " +- ")
            file.write(str(np.std(np.array(MWLDAIML_ranking_loss), ddof=1).item()) + "\n")

            file.write(" subset_accuracy: " + str(np.mean(np.array(MWLDAIML_subset_accuracy)).item()) + " +- ")
            file.write(str(np.std(np.array(MWLDAIML_subset_accuracy), ddof=1).item()) + "\n")

            file.write(" accuracy: " + str(np.mean(np.array(MWLDAIML_accuracy)).item()) + " +- ")
            file.write(str(np.std(np.array(MWLDAIML_accuracy), ddof=1).item()) + "\n")

            file.write(" precision: " + str(np.mean(np.array(MWLDAIML_precision)).item()) + " +- ")
            file.write(str(np.std(np.array(MWLDAIML_precision), ddof=1).item()) + "\n")

            file.write(" recall: " + str(np.mean(np.array(MWLDAIML_recall)).item()) + " +- ")
            file.write(str(np.std(np.array(MWLDAIML_recall), ddof=1).item()) + "\n")

            file.write(" f1: " + str(np.mean(np.array(MWLDAIML_f1)).item()) + " +- ")
            file.write(str(np.std(np.array(MWLDAIML_f1), ddof=1).item()) + "\n")

            file.write(" auc: " + str(np.mean(np.array(MWLDAIML_auc)).item()) + " +- ")
            file.write(str(np.std(np.array(MWLDAIML_auc), ddof=1).item()) + "\n")

            file.write(
                " macro_averaging_accuracy: " + str(
                    np.mean(np.array(MWLDAIML_macro_averaging_accuracy)).item()) + " +- ")
            file.write(str(np.std(np.array(MWLDAIML_macro_averaging_accuracy), ddof=1).item()) + "\n")

            file.write(
                " macro_averaging_precision: " + str(
                    np.mean(np.array(MWLDAIML_macro_averaging_precision)).item()) + " +- ")
            file.write(str(np.std(np.array(MWLDAIML_macro_averaging_precision), ddof=1).item()) + "\n")

            file.write(
                " macro_averaging_recall: " + str(np.mean(np.array(MWLDAIML_macro_averaging_recall)).item()) + " +- ")
            file.write(str(np.std(np.array(MWLDAIML_macro_averaging_recall), ddof=1).item()) + "\n")

            file.write(" macro_averaging_f1: " + str(np.mean(np.array(MWLDAIML_macro_averaging_f1)).item()) + " +- ")
            file.write(str(np.std(np.array(MWLDAIML_macro_averaging_f1), ddof=1).item()) + "\n")

            file.write(" macro_averaging_auc: " + str(np.mean(np.array(MWLDAIML_macro_averaging_auc)).item()) + " +- ")
            file.write(str(np.std(np.array(MWLDAIML_macro_averaging_auc), ddof=1).item()) + "\n")

            file.write(
                " micro_averaging_accuracy: " + str(
                    np.mean(np.array(MWLDAIML_micro_averaging_accuracy)).item()) + " +- ")
            file.write(str(np.std(np.array(MWLDAIML_micro_averaging_accuracy), ddof=1).item()) + "\n")

            file.write(
                " micro_averaging_precision: " + str(
                    np.mean(np.array(MWLDAIML_micro_averaging_precision)).item()) + " +- ")
            file.write(str(np.std(np.array(MWLDAIML_micro_averaging_precision), ddof=1).item()) + "\n")

            file.write(
                " micro_averaging_recall: " + str(np.mean(np.array(MWLDAIML_micro_averaging_recall)).item()) + " +- ")
            file.write(str(np.std(np.array(MWLDAIML_micro_averaging_recall), ddof=1).item()) + "\n")

            file.write(" micro_averaging_f1: " + str(np.mean(np.array(MWLDAIML_micro_averaging_f1)).item()) + " +- ")
            file.write(str(np.std(np.array(MWLDAIML_micro_averaging_f1), ddof=1).item()) + "\n")

            file.write(" micro_averaging_auc: " + str(np.mean(np.array(MWLDAIML_micro_averaging_auc)).item()) + " +- ")
            file.write(str(np.std(np.array(MWLDAIML_micro_averaging_auc), ddof=1).item()) + "\n")

            file.close()
