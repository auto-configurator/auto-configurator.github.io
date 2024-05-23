import os
import re
import sys

import matplotlib.pyplot as plt
import seaborn as sns
import scikit_posthocs as sp
from statannotations.Annotator import Annotator

from plot.containers.read_options import ReadOptions
from plot.draw.data import ReadResults

sns.set_theme(rc={'figure.figsize':(11.7,8.27)})
sns.color_palette("vlag", as_cmap=True)

class Performance:
    """
    Class defines a list of methods used to compare configurations from Crace
    :ivar exec_dir: the directory of Crace results
    :ivar num_repetitions: the number of repetitions of the provided Crace results
    :ivar title: the title of output plot
    :ivar file_name: the full address and name of the output plot file
    :ivar elitist: analyse the elitist configuration or not
    :ivar all: analyse all configurations or not
    """
    def __init__(self, options: ReadOptions ):

        self.exec_dir = options.execDir.value
        self.out_dir = options.outDir.value
        self.num_repetitions = options.numRepetitions.value
        
        self.title = options.title.value
        self.file_name = options.fileName.value

        self.save_name = os.path.join(self.out_dir, self.file_name)

        self.dpi = options.dpi.value
        self.showfliers = options.showfliers.value
        self.stest = options.statisticalTest.value

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

        # select exact method to draw plot via :param{drawMethod}
        # e.g.: boxplot, violinplot
        self.draw_method = options.drawMethod.value

        exp_folders = sorted([subdir for subdir, dirs, files in os.walk(self.exec_dir) \
                      for dir in dirs if dir == 'race_log' ])
        print("# Loading Crace results..")
        self.load = ReadResults(exp_folders, options)

        self.all_results, self.exp_names, self.elite_ids = self.load.load_for_perfomance()
    
    def boxplot(self):
        """
        call the function to draw boxplot
        """
        self.draw_boxplot(self.all_results, self.exp_names, self.elite_ids)

    def violinplot(self):
        """
        call the function to draw violinplot
        """
        self.draw_violinplot(self.all_results, self.exp_names, self.elite_ids)   

    def draw_boxplot(self, data, exp_names, elite_ids):
        """
        Use data to draw a boxplot for the top5 elite configurations
        """

        num = self.num_repetitions

        print("#\n# The Crace results:")
        print(data)
        print("\n# The experiment name(s) of the Crace results you provided:")
        print("# ", re.sub('\'','',str(exp_names)))
        print("#\n# Elite configurations from the Crace results you provided that will be analysed here :")

        if self.num_config != 1:

            for i in range(0, num):
                print("#   {}: {}".format(exp_names[i], elite_ids[exp_names[i]]))

            i = 1
            column = 1
            row = 1
            while i < num:
                i += 1
                if num%i == 0 and num/i >= column:
                    row = int(num/i)
                    column = i
            
            fig, axis = plt.subplots(row, column, sharey=True)
            plt.subplots_adjust(hspace=0.3)

            n = 0
            if row > 1:
                for i in range(0, row):
                    for j in range(0, column):
                        data_exp = data.loc[data['exp_name']==exp_names[n]]
                        if self.showfliers == True:
                            fig = sns.boxplot(
                                x='config_id', y='quality', data=data_exp, width=0.5, fliersize=1, linewidth=0.5, palette="vlag", ax=axis[i,j])
                        else:
                            fig = sns.boxplot(
                                x='config_id', y='quality', data=data_exp, width=0.5, showfliers=False, linewidth=0.5, palette="vlag", ax=axis[i,j])
                        print(elite_ids[exp_names[n]])
                        fig.set_xticklabels(elite_ids[exp_names[n]], rotation=0, fontsize=10)
                        fig.set_xlabel(exp_names[n])
                        if j != 0:
                            fig.set_ylabel('')
                        n += 1  
            elif num > 1:
                for i in range(0, column):
                    data_exp = data.loc[data['exp_name']==exp_names[n]]
                    if self.showfliers == True:
                        fig = sns.boxplot(
                            x='config_id', y='quality', data=data_exp, width=0.5, fliersize=1, linewidth=0.5, palette="vlag", ax=axis[i])
                    else:
                        fig = sns.boxplot(
                            x='config_id', y='quality', data=data_exp, width=0.5, showfliers=False, linewidth=0.5, palette="vlag", ax=axis[i])
                    fig.set_xticklabels(elite_ids[exp_names[n]], rotation=0)
                    fig.set_xlabel(exp_names[n])
                    if i != 0:
                            fig.set_ylabel('')
                    n += 1 
            else:
                if self.showfliers == True:
                    fig = sns.boxplot(x='config_id', y='quality', data=data, width=0.5, fliersize=2, linewidth=0.5, palette="vlag") 

                else:
                    fig = sns.boxplot(x='config_id', y='quality', data=data, width=0.5, showfliers=False, linewidth=0.5, palette="vlag") 
                fig.set_xticklabels(elite_ids[exp_names[0]], rotation=0)
                fig.set_xlabel(exp_names[0])

        elif self.num_config == 1:

            ids = re.sub('}','',re.sub('{','',re.sub('\'','',str(elite_ids))))
            print("#   {}".format(ids))

            if self.showfliers == True:
                fig = sns.boxplot(x='exp_name', y='quality', data=data, width=0.5, fliersize=2, linewidth=0.5, palette="vlag")
            else:
                fig = sns.boxplot(x='exp_name', y='quality', data=data, width=0.5, showfliers=False, linewidth=0.5, palette="vlag")

            fig.set_xlabel('\n{}'.format(ids))

        if self.stest in (True, "True", "ture"):
            plog = self.file_name

            # p_values
            p1 = sp.posthoc_wilcoxon(data, val_col='quality', group_col='exp_name')
            # p_values after multiple test correction
            p2 = sp.posthoc_wilcoxon(data, val_col='quality', group_col='exp_name',
                                    p_adjust='fdr_bh')
            print("Original p_values caculated by 'Wilcoxon':\n", p1)
            print("New p_values corrected by 'fdr_bh':\n", p2)

            with open(self.out_dir + "/" + plog + '.log', 'w') as f1:
                print("Original p_values caculated by 'Wilcoxon':\n", p1, file=f1)
                print("\nNew p_values corrected by 'fdr_bh':\n", p2, file=f1)
                print("\n", file=f1)

            order = []
            pairs = []
            p_values = []
            for x in data['exp_name'].unique():
                order.append(x)
            i = 0
            for x in order[:-1]:
                i += 1
                for y in order[i:]:
                    pairs.append((x,y))
                    p_values.append(p2.loc[x, y])

            annotator = Annotator(fig, pairs=pairs, order=order,
                                data=data, x='exp_name', y='quality')
            annotator.configure(test='Wilcoxon', text_format='star', comparisons_correction='fdr_bh',
                                line_width=0.5, fontsize=8)
                                # line_width=1, fontsize=12)
            # annotator.apply_and_annotate()

            with open(self.out_dir + "/" + plog + '.log', 'a') as f1:
                original_stdout = sys.stdout
                sys.stdout = f1

                try:
                    annotator.apply_and_annotate()
                finally:
                    sys.stdout = original_stdout

        scenarios = ['tsp', 'TSP', 'qap', 'QAP', 'pso', 'PSO', 'sat', 'SAT', 'emili', 'EMILI', 'pfsp', 'PFSP']
        if any(k in self.exec_dir for k in scenarios):
            fig.set_ylabel('mean quality')
        else:
            fig.set_ylabel('mean runtime')
        plot = fig.get_figure()
        plt.suptitle(self.title, size=15)
        plot.savefig(self.save_name, dpi=self.dpi)
        print("#\n# {} has been saved.".format(self.file_name))

    def draw_violinplot(self, data, exp_names, elite_ids):
        """
        Use data to draw a violin for the top5 elite configurations
        """

        num = self.num_repetitions

        print("#\n# The Crace results:")
        print(data)
        print("\n# The experiment name(s) of the Crace results you provided:")
        print("# ", re.sub('\'','',str(exp_names)))
        print("#\n# Elite configurations from the Crace results you provided that will be analysed here :")

        if self.num_config != 1:

            for i in range(0, num):
                print("#   {}: {}".format(exp_names[i], elite_ids[exp_names[i]]))

            i = 1
            column = 1
            row = 1
            while i < num:
                i += 1
                if num%i == 0 and num/i >= column:
                    row = int(num/i)
                    column = i
            
            fig, axis = plt.subplots(row, column, sharey=True)
            plt.subplots_adjust(hspace=0.3)

            n=0
            if row > 1:
                for i in range(0, row):
                    for j in range(0, column):
                        data_exp = data.loc[data['exp_name']==exp_names[n]]
                        fig = sns.violinplot(
                            x='config_id', y='quality', data=data_exp, inner="point", width=0.5, linewidth=0.5, palette="vlag", ax=axis[i,j])
                        fig.set_xticklabels(elite_ids[exp_names[n]], rotation=0, fontsize=10)
                        fig.set_xlabel(exp_names[n])
                        if j != 0:
                            fig.set_ylabel('')
                        n += 1  
            elif num > 1:
                for i in range(0, column):
                    data_exp = data.loc[data['exp_name']==exp_names[n]]
                    fig = sns.violinplot(
                        x='config_id', y='quality', data=data_exp, inner="point", width=0.5, linewidth=0.5, palette="vlag", ax=axis[i])
                    fig.set_xticklabels(elite_ids[exp_names[n]], rotation=0)
                    fig.set_xlabel(exp_names[n])
                    if i != 0:
                        fig.set_ylabel('')
                    n += 1
            else:
                fig = sns.violinplot(x='config_id', y='quality', data=data, inner="point", width=0.5, linewidth=0.5, palette="vlag")
                fig = sns.swarmplot(x='config_id', y='quality', data=data, size=3) 
                fig.set_xticklabels(elite_ids[exp_names[0]], rotation=0)
                fig.set_xlabel(exp_names[0])

        elif self.num_config == 1:

            ids = re.sub('}','',re.sub('{','',re.sub('\'','',str(elite_ids))))
            print("#   {}".format(ids))

            fig = sns.violinplot(x='exp_name', y='quality', data=data, inner=None, width=0.5, linewidth=0.5, palette="vlag")
            fig = sns.swarmplot(x='exp_name', y='quality', data=data, size=3, palette='pink')
            fig.set_xlabel('\n{}'.format(ids))
        
        plot = fig.get_figure()
        plt.suptitle(self.title, size=15)
        plot.savefig(self.save_name, dpi=self.dpi)
        print("# {} has been saved.".format(self.file_name))

