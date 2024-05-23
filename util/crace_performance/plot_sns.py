"""
This module provides a set of plotting functions for the race results.
"""

import argparse
import glob
import json
import logging
import math
import os
import re
import sys
import statistics
import csv
import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns
import scikit_posthocs as sp
import statsmodels.api as sm
from statannotations.Annotator import Annotator
from statsmodels.formula.api import ols
from collections import Counter

sns.color_palette("vlag", as_cmap=True)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.width', 1000)

def check_for_dependencies():
    """
    Check and import all dependent libraries.

    If any required dependency is not installed, stop the script.
    """
    dependency_failed = False
    try:
        global plt  # pylint: disable=global-statement, invalid-name
        import matplotlib.pyplot as plt  # pylint: disable=import-error, import-outside-toplevel
        plt.style.use('ggplot')
    except ModuleNotFoundError:
        print("Matplotlib is a required dependency. You can install it with: ")
        print("$ pip install matplotlib")
        dependency_failed = True
    if dependency_failed:
        print("At least one required dependency was not found. Please install all required dependencies.")
        sys.exit(0)


def expand_folder_arg(folder_arg):
    """
    Runs a glob expansion on the folder_arg elements.
    Also sorts the returned folders.

    :param folder_arg:
    :return:
    """
    folders = []
    for folder in folder_arg:
        folders.extend(glob.glob(folder))
    return sorted(folders)

def csv_2_log(directory):
    """
    Covert the file named "exps_irace.log" obtained from "R2CSV" 
    to the log format in python crace
    
    :param direvtory
    :return
    """
    path = "./race_log/test/"
    # print("\n==================== Folder arg ====================\n", directory)
    folders = sorted([os.path.join(directory, folder) for folder in os.listdir(directory)
                        if os.path.isdir(os.path.join(directory, folder))])
    for folder_name in folders:
        # print("\n==================== Folder name ====================\n", folder_name)
        log_path = os.path.join(folder_name, path)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        log_data = {}
        i=0
        with open(folder_name + "/race_log/test/exps_fin.log",'w') as f2:
            with open(folder_name + "/exps_irace.log",'r') as f1:
                for row in csv.reader(f1):
                    if i >= 1:
                        log_data = {"experiment_id": int(row[0]), "configuration_id": int(row[1]), "quality": float(row[2])}
                        # print(log_data)
                        json.dump(log_data,f2)
                        f2.write("\n")
                    i += 1
        # with open(folder_name + "/race_log/test/exps_fin.log",'w') as f2:
            # f2.write(json.dumps(log_data))
        f1.close
        f2.close

        best_data = {}
        i=0
        with open(folder_name + "/race_log/race.log", 'w') as f4:
            with open(folder_name + "/race_irace.log",'r') as f3:
                for row in csv.reader(f3):
                    if i >= 1:
                        # print("row[0]: {}".format(row[0]))
                        best_data = {"best_id": row[0]}
                        json.dump(best_data, f4)
                    i += 1
                    # best_data = {"best_id": row[0]}
                    # print(best_data)
                # json.dump(best_data,f4)
        # with open(folder_name + "/race_log/race.log", 'w') as f4:
            # f4.write(json.dumps(best_data))
            # json.dump(best_data,f4)
        f3.close
        f4.close

def compute_rpd(race_results, minimum_value):
    """
    :param race_results:
    :param minimum_value:
    :return:
    """
    if minimum_value is not None:
        if minimum_value == math.inf:
            print(race_results)
            minimum_value = min([min(x) for _, x in race_results.items()])
        race_results = {x: [(y / minimum_value) - 1 for y in race_results[x]] for x in race_results}
    print(race_results)
    return race_results

def load_race_results(directory, repetitions=0):
    """
    Load the results from a list of race folders

    :param directory: a directory that contains one or multiple runs of race.
    :param repetitions: the number of repetitions to include. If set to 0, include all repetitions.
    :return:
    """

    ## flag = False: ...acotsp/numC/para128_numC1/exp-*
    ##               return result for singel experiment named para128_numC1
    ## flag = True:  ...acotsp/numC/para*
    ##               return result for all experiments in numC named with para*

    flag = False

    current_folder=(os.path.split(directory))[-2]+'/'+(os.path.split(directory))[-1]

    if "irace" in current_folder \
        and os.path.exists((glob.glob(directory + r"/*/exps_irace.log"))[0]):
        race_folders = sorted([os.path.join(directory, folder) for folder in os.listdir(directory)
                    if os.path.isdir(os.path.join(directory, folder))])
        if not os.path.exists(os.path.join(directory, "race_log/test/exps_fin.log"))  or \
            not os.path.exists(os.path.join(directory, "race_log/race.log")):
            csv_2_log(directory)
            flag = True

    elif os.path.exists((glob.glob(directory + r"/*/crace.test"))[0]):
        race_folders = sorted([os.path.join(directory, folder) for folder in os.listdir(directory)
                                if os.path.isdir(os.path.join(directory, folder))])
        flag = True

    # results from crace and not for 10 repetitions
    elif os.path.exists(os.path.join(directory, "crace.test")):
        race_folders = sorted([os.path.join(directory, folder) for folder in os.listdir(directory)
                        if os.path.isdir(os.path.join(directory, folder))])
        flag = False
    
    else:
        print("# Error: crace.train cannot be found")
        print("#        input file should be ended with para* or exp*")
        exit()

    # print("\n# curret folder: ", current_folder)
    # print("# race_folders: ", race_folders)

    test_results = pd.DataFrame()
    best_id = 0
    for i, folder_name in enumerate(race_folders):
        if 0 < repetitions <= i:
            print(f"Breaking loop at repetition limit {i}")
            break
        if not flag:
            folder_name = os.path.dirname(folder_name)
        test_file_name = folder_name + "/race_log/test/exps_fin.log"
        if (os.path.exists(os.path.join(folder_name, "crace.train")) or 
            os.path.exists(os.path.join(folder_name, "irace.train"))):
            with open(folder_name + "/race_log/race.log", "r") as log_file:
                # print("# folder name: ", folder_name)
                race_log = json.load(log_file)
                best_id = int(race_log["best_id"])
            log_file.close()
        else:
            best_id = 1
        name = folder_name.split("/")[-1]

        chunk_size = 5000
        tmp = []
        with open(test_file_name, "r") as f:
            line = f.readline()
            while line:
                chunk_size -=1
                line_results = json.loads(line)
                current_id = int(line_results["configuration_id"])
                current_quality = float(line_results["quality"])
                if current_id == best_id:
                    quality_dict = {'current_folder': current_folder, 'exp_name': name, 'config_id': current_id, 'quality': current_quality}
                    tmp.append(quality_dict)
                line = f.readline()
                if not line or chunk_size == 0:
                    tmp = pd.DataFrame(tmp)
                    test_results = pd.concat([test_results, tmp], ignore_index=True)
                    tmp = []
                    chunk_size = 5000
        f.close()

    return test_results

def plot_performance_variance(folders, repetitions=0, title="", output_filename="plot"):
    """
    Plot the variance of independent runs of race.

    :param folders: a list of folders to be plotted. Each folder should contain one or multiple race execution folders.
    :param repetitions: the number of repetitions to consider for plotting
    :param title: an optional title for the plot
    :param output_filename: the name for the file to be written to
    :output_filename: the name for the output file. Defaults to "plot.png".
    """
    directory = ""
    if len(folders) == 1:
        directory = folders[0]
    else:
        directory = os.path.dirname(folders[0])
    for folder in folders:
        if os.path.isdir(folder):
            race_results = load_race_results(folder, repetitions)
            fig, axis = plt.subplots()  # pylint: disable=undefined-variable
            axis.boxplot(race_results.values())
            axis.set_xticklabels(race_results.keys(), rotation=0)
            axis.set_title(title)
            plt.show()  # pylint: disable=undefined-variable
            # print(os.path.dirname(folders[0]))
            if title: output_filename = title
            fig.savefig(directory + "/" + output_filename + '.png', bbox_inches='tight')

def extract_number(folder):
    match = re.findall(r'(\d+)', folder)
    # l = [int(x) if x else float('inf') for x in match]
    return tuple(int(num) for num in match)

def plot_final_elite_results(folders, rpd, repetitions=0, title="", output_filename="output", show: bool=True, stest: bool=False, annot: bool=False, ori: bool=False, column: int=2):
    """
    You can either call this method directly or use the command line script (further down).

    :param folders:
    :param repetitions:
    :param title:
    :param output_filename:
    :return:
    """
    directory = ""
    if title not in (None, 'null', ' ', ''): output_filename = title
    all_results = pd.DataFrame()
    all_metrics = pd.DataFrame()

    # load race results from provided folders
    # only one folder
    if len(folders) == 1:
        directory = folders[0]
        results = load_race_results(folders[0], repetitions)
        if folders[0].split("/")[-1]:
            all_results[folders[0].split("/")[-1]] = [statistics.mean(results[x]) for x in results]
        else:
            all_results[folders[0].split("/")[-2]] = [statistics.mean(results[x]) for x in results]
    # more than one folders
    else:
        filtered_paths = [path for path in folders if not path.endswith("irace")]
        # crace is included in provided folders
        if filtered_paths:
            directory = os.path.commonpath(filtered_paths)
        # only irace in provided folders
        else:
            directory = os.path.commonpath(folders)
        basenames = []
        names = []
        parent_names = []
        new_folders = []
        for folder in folders:
            # load results from folders whose sub-folders are exp-
            if glob.glob(folder + r"/*/crace.test") or glob.glob(folder + r"/*/irace.test"):
                new_folders.append(folder)
                basenames.append(os.path.basename(folder))
                names.append(os.path.basename(os.path.abspath(os.path.dirname(folder) + os.path.sep + '.')))
                parent_names.append(os.path.basename(os.path.abspath(os.path.dirname(os.path.dirname(folder)) + os.path.sep + '.')))
        basenames_count = Counter(basenames)
        names_count = Counter(names)
        parent_names_count = Counter(parent_names)

        # load results
        for i, folder in enumerate(new_folders):
            if os.path.isdir(folder):
                results = load_race_results(folder, repetitions)
                avg = results['quality'].replace([np.inf, -np.inf], np.nan).groupby(results['exp_name']).mean().to_dict()
                med = results['quality'].replace([np.inf, -np.inf], np.nan).groupby(results['exp_name']).median().to_dict()
                std = results['quality'].replace([np.inf, -np.inf], np.nan).groupby(results['exp_name']).std().to_dict()
                # /path/to/parent_name/name/base_name/exp-XX
                if len(names_count.keys()) > 1:
                    folder_name = names[i] + '/' + os.path.basename(folder)
                    if len(basenames_count.keys()) == 1:
                        folder_name = names[i]
                elif len(parent_names_count.keys()) == 1:
                    folder_name = folder.split("/")[-1]
                else:
                    folder_name = parent_names[i]
                for exp, quality in avg.items():
                    tmp = pd.DataFrame([[folder_name, exp, quality]], columns=['folder', 'exp_name', 'quality'])
                    all_results = pd.concat([all_results, tmp], ignore_index=True)
                    tmp = pd.DataFrame([[folder_name, exp, quality, med[exp], std[exp]]], columns=['folder', 'exp_name', 'avg', 'med', 'std'])
                    all_metrics = pd.concat([all_metrics, tmp], ignore_index=True)

    if rpd is not None:
        all_results = compute_rpd(all_results, rpd)

    # sort results by [folder, exp_name] based on the name/number
    all_results.sort_values(by=['folder', 'exp_name'], key=lambda x: x.map(extract_number), inplace=True, ignore_index=True)

    # 单栏: plot used in one column
    if column == 1:
        fig_width = 11.69 
        fig_height = 8.27

    # 双栏, 三栏: plots used in double or trible columns
    if column in (2,3):
        fig_width = 3.16
        fig_height = 2.24

    margin = 0 
    left_margin = margin / fig_width
    right_margin = 1 - (margin / fig_width)
    top_margin = 1 - (margin / fig_height)
    bottom_margin = margin / fig_height

    # 单栏: plot used in one column
    if column == 1:
        sns.set_theme(rc={'figure.figsize':(fig_width,fig_height)}, font_scale=1.5) 
        fontSize = 17
        fliersize = 2

    # 双栏: plots used in double columns
    if column == 2:
        sns.set_theme(rc={'figure.figsize':(fig_width,fig_height)}, font_scale=0.7) 
        fontSize = 7 
        fliersize = .5
    
    # 三栏: plots used in trible columns
    if column == 3:
        sns.set_theme(rc={'figure.figsize':(fig_width,fig_height)}, font_scale=0.9) 
        fontSize = 12
        fliersize = .5

    fig, axis = plt.subplots()  # pylint: disable=undefined-variable
    fig.subplots_adjust(left=left_margin, right=right_margin, top=top_margin, bottom=bottom_margin)

    if not show:
        fig = sns.boxplot(x='folder', y='quality', data=all_results, 
                          whis=0.8, showfliers=False, 
                          showmeans=True,
                          meanprops={"marker": "+", 
                                     "markeredgecolor": "black",
                                     "markersize": "8"},
                          width=0.5, linewidth=.5, palette='vlag')
    else:
        fig = sns.boxplot(x='folder', y='quality', data=all_results, 
                          whis=0.8, showfliers=True, fliersize=fliersize, 
                          showmeans=True,
                          meanprops={"marker": "+", 
                                     "markeredgecolor": "black",
                                     "markersize": "8"},
                          width=0.5, linewidth=.5, palette='vlag')

    # show original points
    if ori:
        fig = sns.stripplot(x='folder', y='quality', data=all_results,
                            color='red', size=2, jitter=True)

    # if stest == 'True':
    #     order = ['irace']
    #     pairs = []
    #     for x in all_results['folder'].unique():
    #         if 'irace' in x:
    #             order[0] = x
    #         else:
    #             order.append(x)
    #     for x in order[1:]:
    #         pairs.append((order[0],x))

    l = logging.getLogger('st_log')
    filehandler = logging.FileHandler(directory + "/" + output_filename + '.log', mode='w')
    filehandler.setLevel(0)
    streamhandler = logging.StreamHandler()
    l.setLevel(logging.DEBUG)
    l.addHandler(filehandler)
    l.addHandler(streamhandler)

     ###############################################################################
    #   SSSSS  TTTTT   AAA   TTTTT  IIIII  SSSSS  TTTTT  IIIII  CCCCC   AAA   L     #
    #   S        T    A   A    T      I    S        T      I   C       A   A  L     #
    #   SSSSS    T    AAAAA    T      I    SSSSS    T      I   C       AAAAA  L     #
    #       S    T    A   A    T      I        S    T      I   C       A   A  L     #
    #   SSSSS    T    A   A    T    IIIII  SSSSS    T    IIIII  CCCCC  A   A  LLLL  #
    #                                                                               #
    #                        TTTTT  EEEEE  SSSSS  TTTTT                             #
    #                          T    E      S        T                               #
    #                          T    EEEE   SSSSS    T                               #
    #                          T    E          S    T                               #
    #                          T    EEEEE  SSSSS    T                               #
     ###############################################################################

    if stest in (True, "True", "ture"):

        l.debug(f"All metrics:\n{all_metrics}")

        l.debug(f"\nAll medians: \n{all_metrics['med'].groupby(all_metrics['folder']).median()}")


        ############################# CHECK RESULTS #############################
        #                           Shapiro-Wilk Test                           #
        #                                 LEVENE                                #
        #                                 ANOVA                                 #
        #                         Kruskal-Wallis H Test                         #
        #########################################################################

        # avg for each folder
        data_groups = [all_results['quality'][all_results['folder'] == folder] for folder in all_results['folder'].unique()]

        # Shapiro-Wilk Test
        # H0 hypothesis: normality (正态分布)
        shapiro_string = ''
        stat_s = p_s = []
        for folder in all_results['folder'].unique():
            data_group = all_results[all_results['folder'] == folder]['quality']
            ss, ps = stats.shapiro(data_group)
            stat_s.append(ss)
            p_s.append(ps)
            # print('Shapiro-Wilk Test for folder {}, Statistic: {:.4f}, p-value: {:.4f}'.format(folder, stat_s, p_s))
            shapiro_string += 'Shapiro-Wilk Test for folder {}, Statistic: {:.4f}, p-value: {:.4f}\n'.format(folder, ss, ps)
        l.debug(f'\nShapiro-Wilk Test - H0 hypothesis: normality (0.05)\n{shapiro_string}')

        # do levene
        # H0 hypothesis: homogeneity of variance (方差齐性)
        stat_l, p_l = stats.levene(*data_groups)
        l.debug('\nLevene’s Test - H0 hypothesis: homogeneity of variance (0.05)\n' \
                'stat_l: {:.4f}, p-value: {:.4f}\n'.format(stat_l, p_l))

        # check the results from Shapiro-Wilk Test and levene
        KW_test = ANOVA_test = False
        if p_l < 0.05 or any(x<0.05 for x in p_s):
            KW_test = True
        else:
            ANOVA_test = True

        if ANOVA_test:
            # simulate ANOVA 
            # H0 hypothesis: same mean values
            model = ols('quality ~ C(folder)', data=all_results).fit()
            anova_results = sm.stats.anova_lm(model, typ=2)  # Type 2 ANOVA DataFrame
            l.debug(f'\nANOVA_results - H0 hypothesis: all folders have the same mean values\n{anova_results}')

        if KW_test:
            # do Kruskal-Wallis H
            # H0 hypothesis: same medians
            stat_k, p_k = stats.kruskal(*data_groups)
            l.debug('Kruskal-Wallis Test - H0 hypothesis: all folders have the same medians (0.05)\n' \
                    'Statistic: {:.4f}, p-value: {:.4f}'.format(stat_k, p_k))

            ############################# POSTHOC TEST ##############################
            #                             posthoc_dunn                              #
            #                          posthoc_mannwhitney                          #
            #########################################################################

            # # # Dunn:  
            # p1 = sp.posthoc_dunn(all_results, val_col='quality', group_col='folder')
            # # p_values after multiple test correction
            # p2 = sp.posthoc_dunn(all_results, val_col='quality', group_col='folder',
            #                         p_adjust='fdr_bh')
            # p2_4 = p2.round(4)

            # l.debug(f"\nOriginal p_values caculated by 'dunn':\n{p1}")
            # l.debug(f"\nNew p_values corrected by 'fdr_bh':\n{p2}")
            # l.debug(f"\nNew rounded p_values:\n{p2_4}")

            # Wilcoxon rank-sum test
            p1 = sp.posthoc_mannwhitney(all_results, val_col='quality', group_col='folder')
            # p_values after multiple test correction
            p2 = sp.posthoc_mannwhitney(all_results, val_col='quality', group_col='folder',
                                    p_adjust='fdr_bh')
            p2_4 = p2.round(4)

            l.debug(f"\nOriginal p_values caculated by 'mannwhitney (Wilcoxon rank-sum test)':\n{p1}")
            l.debug(f"\nNew p_values corrected by 'fdr_bh':\n{p2}")
            l.debug(f"\nNew rounded p_values:\n{p2_4}")

    ############################# POSTHOC TEST ##############################
    #                           posthoc_wilcoxon                            #
    #########################################################################

    # Wilcoxon signed-rank test
    p1 = sp.posthoc_wilcoxon(all_results, val_col='quality', group_col='folder')
    # p_values after multiple test correction
    p2 = sp.posthoc_wilcoxon(all_results, val_col='quality', group_col='folder',
                            p_adjust='fdr_bh')
    p2_4 = p2.round(4)
    l.debug(f"\n############################# Wilcoxon Signed-rank Test ##############################")
    l.debug(f"\nOriginal p_values caculated by 'Wilcoxon':\n{p1}")
    l.debug(f"\nNew p_values corrected by 'fdr_bh':\n{p2}")
    l.debug(f"\nNew rounded p_values:\n{p2_4}")

    ############################# ANNOATE ##############################
    if annot:
        order = []
        pairs = []
        p_values = []
        for x in all_results['folder'].unique():
            order.append(x)
        i = 0
        for x in order[:-1]:
            i += 1
            for y in order[i:]:
                pairs.append((x,y))
                p_values.append(p2.loc[x, y])

        annotator = Annotator(fig, pairs=pairs, order=order,
                            data=all_results, x='folder', y='quality')
        annotator.configure(test='Wilcoxon', text_format='star', comparisons_correction='fdr_bh',
                            line_width=.3, fontsize=fontSize-2)
        # annotator.apply_and_annotate()

        with open(directory + "/" + output_filename + '.log', 'a') as f1:
            original_stdout = sys.stdout
            sys.stdout = f1

            try:
                annotator.apply_and_annotate()
            finally:
                sys.stdout = original_stdout

    if title not in (None, 'None' 'null'): fig.set_xlabel('\n'+title, size=fontSize)
    scenarios = ['tsp', 'TSP', 'qap', 'QAP', 'pso', 'PSO', 'sat', 'SAT', 'emili', 'EMILI', 'pfsp', 'PFSP']
    if any(k in folder for folder in new_folders for k in scenarios):
        fig.set_ylabel('mean quality', size=fontSize)
    else:
        fig.set_ylabel('mean runtime', size=fontSize)
    xlables = fig.get_xticklabels()
    l_xlables = sum(len(str(x)) for x in xlables)
    print(l_xlables)
    if l_xlables > 25:
        plt.xticks(rotation=90, size=fontSize)
    else:
        plt.xticks(rotation=0, size=fontSize)
    plt.yticks(rotation=0, size=fontSize)

    plot = fig.get_figure()
    plot.savefig(directory + "/" + output_filename + '.png', bbox_inches='tight', dpi=500)

    print("\n# Plot is saved in folder: ", directory)

def parse_arguments():
    """
    Parse the command line arguments and return them.
    """
    parser = argparse.ArgumentParser(description="eg: python plot_sns.py ~/race/experiments/crace-2.11/acotspqap/qap/numC/para*")
    parser.add_argument('--title', '-t', default="", help="The title of the plot.")
    parser.add_argument('--repetitions', '-r', default=0, type=int, help="The number of repetitions of the experiment")
    parser.add_argument("--statistical-test", "-st", type=bool, default=False, help="Do statistical test or not", dest="st")
    parser.add_argument("--show-annotation", "-a", type=bool, default=False, help="Show annotation or not", dest="annot")
    parser.add_argument("--column", "-c", default=2, type=int, help="How many columns is Plot is used in how many columns", dest="column")
    parser.add_argument("--show-original", "-sa", type=bool, default=False, help="Show original points or not", dest="original")
    parser.add_argument("--folder", "-e", nargs='+', help="A list of folders to include, supports glob expansion")
    parser.add_argument("--output", "-o", default="plot", help="The name of the output file(.png)")
    parser.add_argument("--showfliers", "-s", type=bool, default=True, help="show fliers or not")
    parser.add_argument("--relative-difference", "-rpd", nargs='?', type=int, default=None, const=math.inf, dest="rdp",
                        help="The best known quality. If only the flag -rpd is set, "
                             "then the minimum value from the test set will be used.")
    args = parser.parse_args()
    return args

def execute_from_args():
    """
    Parse the command line arguments and plot according to them.
    """
    args = parse_arguments()
    print("# Provided parameters: ")
    for k,v in vars(args).items():
        if v:
            print(f"#  {k}: '{v}'")
    check_for_dependencies()
    folders = expand_folder_arg(args.folder)

    plot_final_elite_results(folders, args.rdp, args.repetitions, args.title, 
                             args.output, args.showfliers, args.st, args.annot, args.original, args.column)


if __name__ == "__main__":
    execute_from_args()
