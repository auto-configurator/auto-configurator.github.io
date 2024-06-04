import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import scikit_posthocs as sp
from statannotations.Annotator import Annotator

from plot.containers.read_options import ReadOptions
from plot.draw.data import ReadResults

sns.set_theme(rc={'figure.figsize':(11.7,8.27)})
sns.color_palette("vlag", as_cmap=True)
fontSize=13.5
sns.set_theme(font_scale=1.2)

class Configurations:
    """
    Class defines a list of methods used to compare the performance of sampled configurations from Crace
    :ivar exec_dir: the directory of Crace results
    :ivar num_repetitions: the number of repetitions of the provided Crace results
    :ivar title: the title of output plot
    :ivar file_name: the full address and name of the output plot file
    :ivar elitist: analyse the elitist configuration or not
    :ivar all: analyse all configurations or not
    """
    def __init__(self, options: ReadOptions):

        self.exec_dir = options.execDir.value
        self.out_dir = options.outDir.value
        self.num_repetitions = options.numRepetitions.value
        
        self.title = options.title.value
        self.file_name = options.fileName.value

        self.save_name = os.path.join(self.out_dir, self.file_name)

        self.dpi = options.dpi.value
        self.showfliers = options.showfliers.value
        self.showmeans = options.showmeans.value

        if options.numConfigurations.value == "elites":
            self.num_config = 5
        elif options.numConfigurations.value == "elitist":
            self.num_config = 1
        elif options.numConfigurations.value == "all":
            self.num_config = -5
        elif options.numConfigurations.value == 'allelites':
            self.num_config = -1
        elif options.numConfigurations.value == "else":
            self.num_config = options.elseNumConfigs.value

        self.results_from = options.resultsFrom.value

        self.draw_method = options.drawMethod.value = "boxplot"

        exp_folders = sorted([subdir for subdir, dirs, files in os.walk(self.exec_dir) \
                      for dir in dirs if dir == 'race_log' ])
        print("# Loading Crace results..")
        self.load = ReadResults(exp_folders, options)
        self.all_results, self.exp_names, self.elite_ids = self.load.load_for_configurations()

    def boxplot(self):
        """
        call the function to draw boxplot
        """
        self.draw_boxplot(self.all_results, self.exp_names, self.elite_ids)

    def draw_boxplot(self, elite_results, exp_names, elite_ids):
        """
        Use data to draw a boxplot for the top5 elite configurations
        """

        print("#\n# The Crace results:")
        print(elite_results)
        print("#\n# Elite configurations from the Crace results you provided that will be analysed here :")
        # print("#   ", elite_ids)

        # when drawing plot for the process, num_config is -5 (allelites) or 5 (elites)
        results1 = elite_results.groupby(['config_id']).mean().T.to_dict()
        results2 = elite_results.groupby(['config_id']).count().T.to_dict()
        results_mean = {}
        results_count = {}

        # mean based on quality
        # for id, item in results1.items():
        #     results_mean[int(id)] = None
        #     results_mean[int(id)] = item['quality']
        #     if item['quality'] < min_quality:
        #         min_quality = item['quality']
        #         best_id = id

        for id, item in results2.items():
            results_count[int(id)] = None
            results_count[int(id)] = item['instance_id']

        # mean based on Nins, when there is a tie on quality
        for id, item in results1.items():
            results_mean[int(id)] = None
            results_mean[int(id)] = item['quality']

        # Sort elite_ids based on results_count values
        #   1. num of instances, increase
        #   2. mean quality, decrease
        results_count_sorted = {k:v for k, v in sorted(results_count.items(), key=lambda x: (x[1], -results_mean[x[0]]), reverse=False)}
        elite_ids_sorted = [str(x) for x in results_count_sorted.keys()]

        min_quality = float("inf")
        best_id = 0
        pairs = [ k for k,v in results_count_sorted.items() if v==max(results_count_sorted.values())]
        for id in pairs:
            if results_mean[int(id)] <= min_quality:
                min_quality = results_mean[int(id)]
                best_id = id

        final_best = [x for x in elite_ids if isinstance(x, str)][0]

        elite_labels = []
        key_name = "-%(num)s-" % {"num": int(best_id)}
        for x in elite_ids_sorted:
            if int(x) == int(re.search(r'\d+', final_best).group()):
                x = final_best
                if int(re.search(r'\d+', x).group()) == int(best_id):
                    x = "-%(num)s-" % {"num": x}
            elif int(x) == int(best_id):
                x = key_name
            elite_labels.append(x)

        # print("#   ", elite_labels)
        print("#  (number with *: final elite configuration from crace)")
        print("#  (number with -: configuration having the minimal average value on the most instances)")
        for x in elite_labels:
            int_x = int(re.search(r'\d+', x).group())
            print(f"#{x:>12}: ({results_count_sorted[int_x]:>2}, {results_mean[int_x]})", end="\n")

        # draw the plot
        fig, axis = plt.subplots()  # pylint: disable=undefined-variable
        sns.set_theme(font_scale=1.5)
        fig = sns.boxplot(x='config_id', y='quality', data=elite_results, \
                            order=elite_ids_sorted, \
                            width=0.5, showfliers=self.showfliers, fliersize=2,\
                            showmeans=self.showmeans,
                            meanprops={"marker": "+",
                                        "markeredgecolor": "black",
                                        "markersize": "8"},
                            linewidth=0.5, palette="vlag")
        if max(results_count_sorted.values()) <= 30:
            fig = sns.stripplot(x='config_id', y='quality', data=elite_results,
                                color='darkred', size=2, jitter=True)

        # add p-value for the configurations who have the most instances
        if len(pairs) > 1:
            if (int(re.search(r'\d+', final_best).group()) != best_id and 
                int(re.search(r'\d+', final_best).group()) in pairs):
                pair = (str(best_id), str(re.search(r'\d+', final_best).group()))
            else:
                pair = (elite_ids_sorted[-2], elite_ids_sorted[-1])
            pairs_results = elite_results.loc[elite_results['config_id'].isin(pair)].copy()

            pairs_results.loc[:, 'config_id'] = pairs_results['config_id'].astype(int)
            pairs_results.loc[:, 'instance_id'] = pairs_results['instance_id'].astype(int)
            pairs_results = pairs_results.sort_values(by=['config_id', 'instance_id'], ascending=True)
            pairs_results.loc[:, 'config_id'] = pairs_results['config_id'].astype(str)
            pairs_results.loc[:, 'instance_id'] = pairs_results['instance_id'].astype(str)

            # p1 = sp.posthoc_wilcoxon(pairs_results, val_col='quality', group_col='config_id')
            # p_values after multiple test correction
            p2 = sp.posthoc_wilcoxon(pairs_results, val_col='quality', group_col='config_id',
                                    p_adjust='fdr_bh')
            p2_4 = p2.round(4)

            print("#\n# Adjusted p-values of the last two elite configurations:")
            print(p2_4)

            annotator = Annotator(fig, pairs=[pair], data=elite_results, order=elite_ids_sorted, x='config_id', y='quality')
            annotator.configure(test='Wilcoxon', text_format='simple', comparisons_correction='fdr_bh',
                                show_test_name=False, line_width=1, fontsize=fontSize)
            annotator.apply_and_annotate()

        if len(elite_labels) > 12:
            fig.set_xticklabels(elite_labels, rotation=90)
        else:
            fig.set_xticklabels(elite_labels, rotation=0)
        if self.title in (None, 'none', 'None', '', ' '):
            fig.set_xlabel(None)
        elif self.title:
            fig.set_xlabel(self.title, size=fontSize)
        else:
            fig.set_xlabel(exp_names)

        # fig.set_ylabel('quality', size=fontSize)
        fig.set_ylabel('runtime', size=fontSize)
        plt.xticks(size=fontSize)
        plt.yticks(size=fontSize)

        results_count = dict(sorted(results_count.items(), key=lambda x:x[0]))

        # if self.results_from in (None, "None"):
        # add instance numbers
        ymin, ymax = plt.ylim()
        _, xmax = plt.xlim()
        ylength = ymax - ymin
        ymax_n = ylength * 1.05 + ymin
        ymax_n1 = ylength * 1.06 + ymin
        ymax_n2 = ylength * 1.02 + ymin
        plt.ylim(ymin, ymax_n)
        plt.text(x=xmax-0.3,y=ymax_n1,s="ins_num",ha='right',size=fontSize,color='blue')
        i=0
        for x in results_count_sorted.values():
            plt.text(x=i,y=ymax_n2,s=x,ha='center',size=fontSize,color='blue')
            i+=1

        plt.tight_layout()

        plot = fig.get_figure()
        plot.savefig(self.save_name, dpi=self.dpi)
        print("#\n# {} has been saved.".format(self.file_name))

        # plt.show()
