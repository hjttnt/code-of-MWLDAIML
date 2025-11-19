import numpy as np
import heapq
import pandas as pd
import random
import time
import SSMWLDAIML
from SSParams import Params

def load_csv(path):
    data_read = pd.read_csv(path)
    list = data_read.values.tolist()
    data = np.array(list)
    return data


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
Data_name = ["birds", "genbase",  "medical", "enron", "scene", "bibtex", "corel16k001"]
param_name = ['alpha', 'beta', 'lamda1', 'lamda2', 'lamda3', 'rou', 'rou1', 'rou2', 'rou3', 'rou4', 'miu', 'miu1',
              'miu2', 'miu3', 'miu4', 'yuzhi']

for data in Data_name:
    file_name = data
    is_fix = True
    if data in ["CAL500", "IMA"]:
        is_fix = False

    print(data)
    param = {}
    for paramName in param_name:
        param[paramName] = Params[data][paramName]

    SSMWLDAIML_hamming_loss = list()
    SSMWLDAIML_coverage = list()
    SSMWLDAIML_ranking_loss = list()
    SSMWLDAIML_one_error = list()
    SSMWLDAIML_average_precision = list()
    SSMWLDAIML_subset_accuracy = list()
    SSMWLDAIML_accuracy = list()
    SSMWLDAIML_precision = list()
    SSMWLDAIML_recall = list()
    SSMWLDAIML_f1 = list()
    SSMWLDAIML_auc = list()
    SSMWLDAIML_macro_averaging_accuracy = list()
    SSMWLDAIML_macro_averaging_precision = list()
    SSMWLDAIML_macro_averaging_recall = list()
    SSMWLDAIML_macro_averaging_f1 = list()
    SSMWLDAIML_macro_averaging_auc = list()
    SSMWLDAIML_micro_averaging_accuracy = list()
    SSMWLDAIML_micro_averaging_precision = list()
    SSMWLDAIML_micro_averaging_recall = list()
    SSMWLDAIML_micro_averaging_f1 = list()
    SSMWLDAIML_micro_averaging_auc = list()
    SSMWLDAIML_time = list()

    if is_fix:
        X = load_csv(r"D:\Datasets\Mulan_datasets\{}\{}_train_features.csv".format(file_name, file_name))
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
            X = Features[f, :]
            Y = Labels[f, :]
            c = set(range(n0))
            g = list(c - sample_idx_set)
            X_test = Features[g, :]
            Y_true = Labels[g, :]

        time_start_SSMWLDAIML = time.time()
        (_,SSMWLDAIML_scorce) = SSMWLDAIML.SSMWLDAIML(X, Y, X_test, Y_true, miu_max=1e1000, miu1_max=1e1000,
                                            miu2_max=1e1000, miu3_max=1e1000, miu4_max=1e1000, **param)
        SSMWLDAIML_time.append(time.time() - time_start_SSMWLDAIML)
        print("time:", t, " finishi SSMWLDAIML")
        SSMWLDAIML_hamming_loss.append(SSMWLDAIML_scorce["hamming_loss"])
        SSMWLDAIML_coverage.append(SSMWLDAIML_scorce["coverage"])
        SSMWLDAIML_ranking_loss.append(SSMWLDAIML_scorce["ranking_loss"])
        SSMWLDAIML_one_error.append(SSMWLDAIML_scorce["one_error"])
        SSMWLDAIML_average_precision.append(SSMWLDAIML_scorce["average_precision"])
        SSMWLDAIML_subset_accuracy.append(SSMWLDAIML_scorce["subset_accuracy"])
        SSMWLDAIML_accuracy.append(SSMWLDAIML_scorce["accuracy"])
        SSMWLDAIML_precision.append(SSMWLDAIML_scorce["precision"])
        SSMWLDAIML_recall.append(SSMWLDAIML_scorce["recall"])
        SSMWLDAIML_f1.append(SSMWLDAIML_scorce["f1"])
        SSMWLDAIML_auc.append(SSMWLDAIML_scorce["auc"])
        SSMWLDAIML_macro_averaging_accuracy.append(SSMWLDAIML_scorce["macro_averaging_accuracy"])
        SSMWLDAIML_macro_averaging_precision.append(SSMWLDAIML_scorce["macro_averaging_precision"])
        SSMWLDAIML_macro_averaging_recall.append(SSMWLDAIML_scorce["macro_averaging_recall"])
        SSMWLDAIML_macro_averaging_f1.append(SSMWLDAIML_scorce["macro_averaging_f1"])
        SSMWLDAIML_macro_averaging_auc.append(SSMWLDAIML_scorce["macro_averaging_auc"])
        SSMWLDAIML_micro_averaging_accuracy.append(SSMWLDAIML_scorce["micro_averaging_accuracy"])
        SSMWLDAIML_micro_averaging_precision.append(SSMWLDAIML_scorce["micro_averaging_precision"])
        SSMWLDAIML_micro_averaging_recall.append(SSMWLDAIML_scorce["micro_averaging_recall"])
        SSMWLDAIML_micro_averaging_f1.append(SSMWLDAIML_scorce["micro_averaging_f1"])
        SSMWLDAIML_micro_averaging_auc.append(SSMWLDAIML_scorce["micro_averaging_auc"])

    file = open("semi_result\SSMWLDAIML_{}_result.txt".format(file_name), "a")
    file.write("\n")

    file.write(" time: " + str(np.mean(np.array(SSMWLDAIML_time)).item()) + " +- ")
    file.write(str(np.std(np.array(SSMWLDAIML_time), ddof=1).item()) + "\n")

    file.write(" hamming_loss: " + str(np.mean(np.array(SSMWLDAIML_hamming_loss)).item()) + " +- ")
    file.write(str(np.std(np.array(SSMWLDAIML_hamming_loss), ddof=1).item()) + "\n")

    file.write(" one_error: " + str(np.mean(np.array(SSMWLDAIML_one_error)).item()) + " +- ")
    file.write(str(np.std(np.array(SSMWLDAIML_one_error), ddof=1).item()) + "\n")

    file.write(" coverage: " + str(np.mean(np.array(SSMWLDAIML_coverage)).item()) + " +- ")
    file.write(str(np.std(np.array(SSMWLDAIML_coverage), ddof=1).item()) + "\n")

    file.write(" average_precision: " + str(np.mean(np.array(SSMWLDAIML_average_precision)).item()) + " +- ")
    file.write(str(np.std(np.array(SSMWLDAIML_average_precision), ddof=1).item()) + "\n")

    file.write(" ranking_loss: " + str(np.mean(np.array(SSMWLDAIML_ranking_loss)).item()) + " +- ")
    file.write(str(np.std(np.array(SSMWLDAIML_ranking_loss), ddof=1).item()) + "\n")

    file.write(" subset_accuracy: " + str(np.mean(np.array(SSMWLDAIML_subset_accuracy)).item()) + " +- ")
    file.write(str(np.std(np.array(SSMWLDAIML_subset_accuracy), ddof=1).item()) + "\n")

    file.write(" accuracy: " + str(np.mean(np.array(SSMWLDAIML_accuracy)).item()) + " +- ")
    file.write(str(np.std(np.array(SSMWLDAIML_accuracy), ddof=1).item()) + "\n")

    file.write(" precision: " + str(np.mean(np.array(SSMWLDAIML_precision)).item()) + " +- ")
    file.write(str(np.std(np.array(SSMWLDAIML_precision), ddof=1).item()) + "\n")

    file.write(" recall: " + str(np.mean(np.array(SSMWLDAIML_recall)).item()) + " +- ")
    file.write(str(np.std(np.array(SSMWLDAIML_recall), ddof=1).item()) + "\n")

    file.write(" f1: " + str(np.mean(np.array(SSMWLDAIML_f1)).item()) + " +- ")
    file.write(str(np.std(np.array(SSMWLDAIML_f1), ddof=1).item()) + "\n")

    file.write(" auc: " + str(np.mean(np.array(SSMWLDAIML_auc)).item()) + " +- ")
    file.write(str(np.std(np.array(SSMWLDAIML_auc), ddof=1).item()) + "\n")

    file.write(
        " macro_averaging_accuracy: " + str(
            np.mean(np.array(SSMWLDAIML_macro_averaging_accuracy)).item()) + " +- ")
    file.write(str(np.std(np.array(SSMWLDAIML_macro_averaging_accuracy), ddof=1).item()) + "\n")

    file.write(
        " macro_averaging_precision: " + str(
            np.mean(np.array(SSMWLDAIML_macro_averaging_precision)).item()) + " +- ")
    file.write(str(np.std(np.array(SSMWLDAIML_macro_averaging_precision), ddof=1).item()) + "\n")

    file.write(
        " macro_averaging_recall: " + str(np.mean(np.array(SSMWLDAIML_macro_averaging_recall)).item()) + " +- ")
    file.write(str(np.std(np.array(SSMWLDAIML_macro_averaging_recall), ddof=1).item()) + "\n")

    file.write(" macro_averaging_f1: " + str(np.mean(np.array(SSMWLDAIML_macro_averaging_f1)).item()) + " +- ")
    file.write(str(np.std(np.array(SSMWLDAIML_macro_averaging_f1), ddof=1).item()) + "\n")

    file.write(" macro_averaging_auc: " + str(np.mean(np.array(SSMWLDAIML_macro_averaging_auc)).item()) + " +- ")
    file.write(str(np.std(np.array(SSMWLDAIML_macro_averaging_auc), ddof=1).item()) + "\n")

    file.write(
        " micro_averaging_accuracy: " + str(
            np.mean(np.array(SSMWLDAIML_micro_averaging_accuracy)).item()) + " +- ")
    file.write(str(np.std(np.array(SSMWLDAIML_micro_averaging_accuracy), ddof=1).item()) + "\n")

    file.write(
        " micro_averaging_precision: " + str(
            np.mean(np.array(SSMWLDAIML_micro_averaging_precision)).item()) + " +- ")
    file.write(str(np.std(np.array(SSMWLDAIML_micro_averaging_precision), ddof=1).item()) + "\n")

    file.write(
        " micro_averaging_recall: " + str(np.mean(np.array(SSMWLDAIML_micro_averaging_recall)).item()) + " +- ")
    file.write(str(np.std(np.array(SSMWLDAIML_micro_averaging_recall), ddof=1).item()) + "\n")

    file.write(" micro_averaging_f1: " + str(np.mean(np.array(SSMWLDAIML_micro_averaging_f1)).item()) + " +- ")
    file.write(str(np.std(np.array(SSMWLDAIML_micro_averaging_f1), ddof=1).item()) + "\n")

    file.write(" micro_averaging_auc: " + str(np.mean(np.array(SSMWLDAIML_micro_averaging_auc)).item()) + " +- ")
    file.write(str(np.std(np.array(SSMWLDAIML_micro_averaging_auc), ddof=1).item()) + "\n")

    file.close()
