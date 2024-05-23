import os
import re
import copy
import math
import sys
from matplotlib.ticker import FixedLocator
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_theme(rc={'figure.figsize':(11.7,8.27)})
sns.set_style('white')
sns.color_palette("vlag", as_cmap=True)
fontSize=12

# import plotly as py
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt

from plot.containers.read_options import ReadOptions
from plot.draw.data import ReadResults
from plot.errors import OptionError, ParameterValueError

class Parameters:
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

        self.key_param = options.keyParameter.value
        self.multi_params = options.multiParameters.value
        if self.multi_params is not None and options.drawMethod.value == 'parallelcat':
            self.catx = options.multiParameters.value.split(',')[0]
            self.caty = options.multiParameters.value.split(',')[1]

        self.slice = options.slice.value
        self.dpi = options.dpi.value

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

        self.all_results, self.exp_names, self.elite_ids, self.parameters = self.load.load_for_parameters()
        self.all_results = self.all_results.replace(['null'], np.nan)

        print("#\n# The Crace results:")
        print(self.all_results,)
        print("#\n# The experiment name(s) of the Crace results you provided:")
        print("# ", re.sub('\'','',str(self.exp_names)))
        print("#")

    def parallelcoord(self):
        """
        call the function to draw parallel coord plot
        """
        self.draw_parallelcoord(self.all_results, self.exp_names, self.elite_ids, self.parameters)

    def parallelcat(self):
        """
        call the function to draw parallel categorical plot
        """
        self.draw_parallelcat(self.all_results, self.exp_names, self.elite_ids, self.parameters)

    def sunburst(self):
        """
        call the function to draw sunburst chart
        """
        self.draw_sunburst(self.all_results, self.parameters)

    def pairplot(self):
        """
        call the function to draw scatter matrix plot
        """
        self.draw_pairplot(self.all_results, self.exp_names, self.parameters)

    def histplot(self):
        """
        call the function to draw distplot
        """
        self.draw_histplot(self.all_results, self.parameters)

    def boxplot(self):
        """
        call the function to draw distplot
        """
        self.draw_boxplot(self.all_results, self.parameters)

    def violinplot(self):
        """
        call the function to draw distplot
        """
        self.draw_violinplot(self.all_results, self.parameters)

    def heatmap(self):
        """
        call the function to draw distplot
        """
        self.draw_heatmap(self.all_results, self.parameters)

    def jointplot(self):
        """
        call the function to draw jointplot
        """
        self.draw_jointplot(self.all_results, self.exp_names, self.parameters)

    def draw_parallelcoord(self, data, exp_names, elite_ids, parameters):
        """
        Use data to draw a parallel coord plot for the provided Crace results
        """

        num = len(data)

        data_new = data.copy()
        keyParam = ""

        if self.num_repetitions == 1:
            parameters['config_id'] = {}
            parameters['config_id']['type'] = 'i'
            parameters['config_id']['domain'] = [int(data['config_id'].min()), int(data['config_id'].max())]
            parameters['n_ins'] = {}
            parameters['n_ins']['type'] = 'i'
            parameters['n_ins']['domain'] = [int(data['n_ins'].min()), int(data['n_ins'].max())]
            keyParam = 'n_ins'
            if self.num_config == -5:
                parameters['n_slice'] = {}
                parameters['n_slice']['type'] = 'i'
                parameters['n_slice']['domain'] = [int(data['n_slice'].min()), int(data['n_slice'].max())]
        else:
            parameters['exp_name'] = {}
            parameters['exp_name']['type'] = 'c'
            parameters['exp_name']['domain'] = exp_names
            keyParam = 'exp_name'

        if self.num_config == -5 and self.slice in (True, "true", "True"):
            keyParam = 'n_slice'

        if self.key_param not in ('null', None):
            keyParam = self.key_param
        else:
            keyParam = keyParam 

        new_parameters = {}
        if self.multi_params is not None:
            for x in parameters.keys():
                if x in self.multi_params.split(','):
                    new_parameters[x] = {}
                    new_parameters[x] = copy.deepcopy(parameters[x])
            new_parameters[keyParam] = {}
            new_parameters[keyParam] = copy.deepcopy(parameters[keyParam])
        else:
            new_parameters = copy.deepcopy(parameters)

        map_dic = {}
        for name in new_parameters.keys():
            if new_parameters[name]['type'] == 'c':
                i = 0
                map_dic[name] = {}
                new_parameters[name]['domain'] = []
                for a in parameters[name]['domain']:
                    map_dic[name][a] = i
                    new_parameters[name]['domain'].append(i)
                    i += 1
                data_new[name] = data[name].map(map_dic[name])

        col1 = data_new.count() == 0
        for i in range(len(col1)):
            name = col1.index[i]
            if name in new_parameters.keys():
                per = list(data[name]).count('null')/float(num)
                if per > 0:
                    if new_parameters[name]['type'] != 'c':
                        old = copy.deepcopy(new_parameters[name]['domain'])
                        new_parameters[name]['domain'][0] = old[0]-1
                        print("! WARNING: 'null' values in {} has been replaced by {}.".format(name, old[0]-1))
                        data_new[name].replace('null', old[0]-1, inplace=True)
                    else:
                        old = copy.deepcopy(parameters[name]['domain'])
                        new_parameters[name]['domain'].append(len(old))
                        parameters[name]['domain'].append('null')
                        print("! WARNING: {} has 'null' values.".format(name))
                if col1[i]:
                    print("! WARNING: all values in {} are NaN, it will be deleted for the plot!".format(name) )
                    new_parameters.pop(col1.index[i])
            else:
                continue

        print("#\n# The new parameters:")
        print(new_parameters.keys())
        print("#\n# The selected Crace results:")
        print(data_new)

        print("#\n# Elite configurations from the Crace results you provided that will be analysed here :")
        for a in elite_ids.keys():
            print("#   {}: {}".format(a, elite_ids[a]))

        plot_dict = []
        each_dict = {}
        for name in new_parameters.keys():
            if new_parameters[name]['type'] != 'c':
                each_dict = dict(
                    range = new_parameters[name]['domain'],
                    label = name, 
                    values = data_new[name]
                )
            else:
                each_dict = dict(
                    tickvals = new_parameters[name]['domain'],
                    ticktext = parameters[name]['domain'],
                    label = name, 
                    values = data_new[name]
                )
            plot_dict.append(each_dict)
        
        cmin = 0
        cmax = 0

        if keyParam == 'n_slice':
            cmin = data_new[keyParam].min()
            cmax = data_new[keyParam].max()
            data_key = pd.to_numeric(data_new[keyParam])
        else:
            if parameters[keyParam]['type'] != 'c':
                cmin = parameters[keyParam]['domain'][0]
                cmax = parameters[keyParam]['domain'][1]
                data_key = pd.to_numeric(data_new[keyParam])
            else:
                cmin = 0
                cmax = len(parameters[keyParam]['domain'])
                data_key = data_new[keyParam]

        fig = go.Figure(data=
        go.Parcoords(
            line = dict(color = data_key,
                        colorscale = 'Tealrose',
                        showscale = True,
                        cmin = cmin,
                        cmax = cmax),
            dimensions = list([line for line in plot_dict])
        ))
        fig.show()

    def draw_parallelcat(self, cat_data, exp_names, elite_ids, parameters):
        """
        Use data to draw a parallel categorical plot for the provided Crace results
        """

        num = len(cat_data)

        if self.num_repetitions == 1:
            parameters['config_id'] = {}
            parameters['config_id']['type'] = 'i'
            parameters['config_id']['domain'] = [int(cat_data['config_id'].min()), int(cat_data['config_id'].max())]
            keyParam = 'config_id'
        else:
            parameters['exp_name'] = {}
            parameters['exp_name']['type'] = 'c'
            parameters['exp_name']['domain'] = exp_names
            keyParam = 'exp_name'

        new_parameters = {}
        if self.multi_params is not None:
            for x in parameters.keys():
                if x in self.multi_params.split(','):
                    new_parameters[x] = {}
                    new_parameters[x] = copy.deepcopy(parameters[x])
        else:
            new_parameters = copy.deepcopy(parameters)

        col1 = cat_data.count() == 0
        for i in range(len(col1)):
            name = col1.index[i]
            per = list(cat_data[name]).count('null')/float(num)
            if col1[i]:
                print("! WARNING: all values in {} are NaN, it will be deleted for the plot!".format(name) )
                new_parameters.pop(col1.index[i])
            if per > 0:
                if new_parameters[name]['type'] != 'c':
                    old = copy.deepcopy(new_parameters[name]['domain'])
                    new_parameters[name]['domain'][0] = old[0]-1
                    print("! WARNING: 'null' values in {} has been replaced by {}.".format(name, old[0]-1))
                    cat_data[name].replace('null', old[0]-1, inplace=True)
                else:
                    old = copy.deepcopy(parameters[name]['domain'])
                    new_parameters[name]['domain'].append(len(old))
                    parameters[name]['domain'].append('null')
                    print("! WARNING: {} has 'null' values.".format(name))


        print("#\n# The new parameters:")
        print(new_parameters.keys())

        print("#\n# Elite configurations from the Crace results you provided that will be analysed here :")
        for a in elite_ids.keys():
            print("#   {}: {}".format(a, elite_ids[a]))

        plot_dict = []
        each_dict = {}
        for name in new_parameters.keys():
            if new_parameters[name]['type'] == 'c':
                each_dict = dict(
                    label = name, 
                    values = cat_data[name]
                )
                plot_dict.append(each_dict)
        
        color = np.zeros(len(plot_dict), dtype='uint8')
        colorscale = [[0, 'gray'], [1, 'firebrick']]
        
        # Build figure as FigureWidget
        fig = go.FigureWidget(data=[
        go.Scatter(
            x = cat_data[self.catx],
            y = cat_data[self.caty],
            marker={'color': 'gray'},
            mode='markers',
            selected={'marker': {'color': 'firebrick'}},
            unselected={'marker': {'opacity': 0.5}}
        ), 
        go.Parcats(
                domain={'y': [0, 0.4]}, 
                dimensions=list([line for line in plot_dict]),
                line={'colorscale': colorscale, 'cmin': 0, 'cmax': 1, 'color': color, 'shape': 'hspline'}),
        ])

        fig.add_trace(fig.data[0])
        fig.add_trace(fig.data[1])

        fig.update_layout(
            height=1000, 
            xaxis={'title': self.catx},
            yaxis={'title': self.caty, 'domain': [0.6, 1]},
            dragmode='lasso', 
            hovermode='closest',
            overwrite=True)

        # Update color callback
        # def update_color(trace, points, state):
        #     # Update scatter selection
        #     fig.data[0].selectedpoints = points.point_inds

        #     # Update parcats colors
        #     new_color = color
        #     new_color[points.point_inds] = 1
        #     fig.data[1].line.color = new_color
        def update_color(trace, points, state):
            global color, color_numeric
            print("Event triggered", points.point_inds)

            for i in range(len(color)):
                color[i] = 'gray'
            for ind in points.point_inds:
                color[ind] = 'firebrick'

            color_numeric = [0 if c == 'gray' else 1 for c in color]
            fig.data[1].line.color = color_numeric
            fig.update_traces()

        # Register callback on scatter selection...
        fig.data[0].on_selection(update_color)
        # and parcats click
        fig.data[1].on_click(update_color)

        fig.show()

    def draw_sunburst(self, sun_data, parameters):
        """
        Use sun_data to draw a sunburst plot for the provided Crace results
        """

        num = len(sun_data)

        data = copy.deepcopy(sun_data)
        new_parameters = {}
        if self.multi_params is not None:
            for x in parameters.keys():
                if x in self.multi_params.split(','):
                    new_parameters[x] = {}
                    new_parameters[x] = copy.deepcopy(parameters[x])
        else:
            new_parameters = copy.deepcopy(parameters)

        col1 = data.count() == 0
        for i in range(len(col1)):
            name = col1.index[i]
            per = list(data[name]).count('null')/float(num)
            if col1[i]:
                print("! WARNING: all values in {} are NaN, it will be deleted for the plot!".format(name) )
                new_parameters.pop(col1.index[i])
            if per > 0:
                if new_parameters[name]['type'] != 'c':
                    old = copy.deepcopy(new_parameters[name]['domain'])
                    new_parameters[name]['domain'][0] = old[0]-1
                    print("! WARNING: 'null' values in {} has been replaced by {}.".format(name, old[0]-1))
                    data[name].replace('null', old[0]-1, inplace=True)
                else:
                    old = copy.deepcopy(parameters[name]['domain'])
                    new_parameters[name]['domain'].append(len(old))
                    parameters[name]['domain'].append('null')
                    print("! WARNING: {} has 'null' values.".format(name))


        subgroup = {}
        for name in new_parameters.keys():
            subgroup[name] = {}
            subgroup[name]['names'] = []
            subgroup[name]['props'] = []
            if new_parameters[name]['type'] != 'c':
                data[name] = pd.to_numeric(data[name])
                left = parameters[name]['domain'][0]
                right = parameters[name]['domain'][1]
                tmp = (left+right)/3
                sub1 = '[{:.2f}, {:.2f}]'.format(left, tmp)
                sub2 = '[{:.2f}, {:.2f}]'.format(tmp, 2*tmp)
                sub3 = '[{:.2f}, {:.2f}]'.format(2*tmp, right)

                subgroup[name]['names'].append(sub1)
                s_bool = ((data[name] >= left) & (data[name] < tmp))
                subgroup[name]['props'].append(s_bool.sum())

                subgroup[name]['names'].append(sub2)
                s_bool = ((data[name] >= tmp) & (data[name] < 2*tmp))
                subgroup[name]['props'].append(s_bool.sum())

                subgroup[name]['names'].append(sub3)
                s_bool = ((data[name] >= 2*tmp) & (data[name] <= right))
                subgroup[name]['props'].append(s_bool.sum())

                if sum(subgroup[name]['props']) < len(data[name]):
                    subgroup[name]['names'].append('null')
                    s_bool = (data[name] < left)
                    subgroup[name]['props'].append(s_bool.sum())
                
            else:
                for x in parameters[name]['domain']:
                    subgroup[name]['names'].append(x)
                    s_bool = data[name] == x
                    subgroup[name]['props'].append(s_bool.sum())

        num = len(new_parameters.keys())
        labels = [self.title]
        values = [0]
        parents = [""]
        for name in new_parameters.keys():
            i = 0
            labels.append(name)
            # values.append(self.count_values(sun_data[name]))
            values.append(sum(subgroup[name]['props']))
            values[0] += sum(subgroup[name]['props'])
            parents.append(self.title)
            for label in subgroup[name]['names']:
                labels.append(label)
                values.append(subgroup[name]['props'][i])
                parents.append(name)
                i += 1
        print(f'\nvalues: {values}')
        print(f'\nlabels: {labels}')
        print(f'\nparents: {parents}')

        vlag_palette = sns.color_palette("vlag", len(labels)).as_hex()
        fig = go.Figure()
        fig.add_trace(go.Sunburst(
            labels = labels,
            parents = parents,
            values = values,
            branchvalues='total',
            maxdepth=3,
            count='branches',
            insidetextorientation='horizontal',
            insidetextfont=dict(size=17,color='black'),
            outsidetextfont=dict(size=17),
            textinfo="label+value",
            marker=dict(colors=vlag_palette),
        ))
        fig.update_layout(margin = dict(t=0, l=0, r=0, b=0))

        fig.show()

    def draw_pairplot(self, data, exp_names, parameters):
        """
        Use data to draw scatter matrix plot
        """

        # sns.set(style="ticks", color_codes=True)

        if parameters[self.key_param]['type'] != 'c':
            raise OptionError("!! keyParam {} must be categorical.".format(self.key_param))

        dimensions = []
        if self.multi_params is not None:
            for x in self.multi_params.split(','):
                if parameters[x]['type'] == 'c':
                    raise OptionError("!! keyParam {} must be numeric.".format(x))
                else:
                    dimensions.append(x)
        else:
            dimensions = list(parameters.keys())

        data_new = data.copy()

        for name in parameters.keys():
            per = list(data[name]).count('null')/len(data)
            if per == float(1):
                print("! WARNING: all values in {} are 'null', it will be deleted for the plot!".format(name))
                dimensions.pop(dimensions.index(name))
            elif per > 0:
                data_new[name].replace('null', np.nan, inplace=True)

        col = data_new.count() == 0
        for i in range(len(col)):
            if col[i] and col.index[i] in dimensions:
                print("! WARNING: all values in {} are NaN, it will be deleted for the plot!".format(col.index[i]) )
                dimensions.pop(dimensions.index((col.index[i])))

        print("#")

        # fig = px.scatter_matrix(data,
        #     dimensions = dimensions,
        #     color = keyParam,
        #     symbol= keyParam,
        #     title = self.title,
        #     labels={col:col.replace('_', ' ') for col in data.columns})
        # fig.update_traces(diagonal_visible=False)
        # fig.show()

        sns.set_theme(font_scale=2) 
        fig = sns.pairplot(data=data_new,
                           hue=self.key_param,
                           vars=dimensions,
                           palette="Set2",
                           diag_kind='kde',
                           height=5)

        # plt.suptitle(self.title, size=fontSize)
        fig.savefig(self.save_name, dpi=self.dpi)
        print("# {} has been saved.".format(self.file_name))

    def draw_histplot(self, data, parameters):
        """
        Use provided data to draw distplot
        """

        sns.set_style('whitegrid')
        sns.set_theme(rc={'figure.figsize':(11.7,8.27)}, font_scale=1.2)
        fontSize=15

        param_names = []
        param_types = []
        param_len = {}
        if self.multi_params is not None:
            for x in self.multi_params.split(','):
                param_names.append(x)
        else:
            for x in parameters.keys():
                # if parameters[x]['type'] != 'c':
                param_names.append(x)

        for x in param_names:
            param_types.append(parameters[x]['type'])
            param_len[x] = 0
            if parameters[x]['type'] == 'c':
                param_len[x] = sum(len(str(x)) for x in data[x].unique() if x not in (np.nan, 'null'))
        
        print("\n# The parameters selected to be visualised: \n#  {}".format(param_names))
        print("\n# The type of selected parameters:  \n#  {}".format(param_types))

        data_new = data.copy()

        for name in parameters.keys():
            per = list(data[name]).count('null')/len(data)
            if per == float(1):
                print("! WARNING: all values in {} are 'null', it will be deleted for the plot!".format(name))
                param_names.pop(param_names.index(name))
            elif per > 0:
                data_new[name].replace('null', np.nan, inplace=True)

        col = data_new.count() == 0
        for i in range(len(col)):
            if col[i] and col.index[i] in param_names:
                print("! WARNING: all values in {} are NaN, it will be deleted for the plot!".format(col.index[i]) )
                param_names.pop(param_names.index((col.index[i])))

        print("#")

        page_break = True
        page_num = 1
        max_num = 0
        max_row = 0
        max_col = 0
        while page_break:
            max_num = int(math.ceil(len(param_names)/page_num))
            if max_num <= 2:
                max_row = 1
                max_col = max_num
                page_break = False
            elif max_num <= 8:
                if max_num in (3, 5, 7): max_num += 1
                max_row = 2
                max_col = int(math.ceil(max_num/max_row))
                page_break = False
            else:
                page_num += 1

        if self.slice not in (True, "true", "True"):
            params = locals()
            file_names = locals()
            start = 0
            for i in range(0, page_num):
                fig, axis = plt.subplots(max_row, max_col, sharey=True, sharex=False)
                plt.subplots_adjust(hspace=0.5, wspace=0.4, bottom=0.2)
                title = '\nPage ' + str(i+1) + ' of ' + str(page_num)
                
                if start+max_num-1 <= len(param_names):
                    params['params%s' % i] = param_names[start:start+max_num]
                else:
                    params['params%s' % i] = param_names[start:]              
                start = start+max_num
                file_names['plot%s' % i] = self.file_name.split('.')[0]+str(i)   

                page_plots = min(max_num, len(params['params%s' % i]))

                row = column = idx = 0
                for idx in range(max_num):
                    if max_row == 2:
                        ax = axis[row, column]
                    elif max_row == max_col == 1:
                        ax = None
                    else:
                        ax = axis[column]
                    if idx < page_plots:
                        name = params['params%s' % i][idx]
                        if parameters[name]['type'] != 'c':
                            fig = sns.histplot(data=data_new.sort_values(by=name, na_position='last', ascending=True),
                                            x=name, stat='percent', kde=True, 
                                            ax=ax)
                            fig = sns.rugplot(data=data_new.sort_values(by=name, na_position='last', ascending=True),
                                            x=name, ax=ax)
                        else:
                            fig = sns.histplot(data=data_new.sort_values(by=name, na_position='last', ascending=True),
                                                x=name, stat='percent', kde=False, 
                                                ax=ax)
                            if param_len[name] > 15:
                                labels = ax.get_xticklabels()
                                locations = ax.get_xticks()
                                ax.xaxis.set_major_locator(FixedLocator(locations))
                                ax.set_xticklabels(labels, rotation=90)
                        if len(name) > 25:
                            name = re.sub(r"(.{25})", "\\1\n", name)
                        if column != 0:
                            fig.set_ylabel('')
                        ax.xaxis.set_label_position('top')
                        fig.set_xlabel(name, rotation=0, size=fontSize)
                    else:
                        ax.axis('off')

                    if column < max_col-1:
                        column += 1
                    else:
                        column = 0
                        row += 1 

                plot = fig.get_figure()
                file_name = file_names['plot%s' % i] 
                save_name = os.path.join(self.out_dir, file_name)
                
                plt.tight_layout()

                if self.title not in (None, 'None', 'none'):
                    plt.suptitle(self.title, size=fontSize)
                plot.savefig(save_name, dpi=self.dpi)
                print("# {} has been saved in {}.".format(file_name, self.out_dir))

        else:
            file_names = locals()
            for i in range(0, page_num):
                file_names['plot%s' % i] = self.file_name.split('.')[0]+str(i)
                name = param_names[i]

                max_slice = max(data_new['n_slice'])

                fig, axis = plt.subplots(max_slice, 1, sharey=True, sharex=True)

                for j in range(1, max_slice+1):
                    ax = axis[j-1]
                    if parameters[name]['type'] != 'c':

                    # statstr
                    # Aggregate statistic to compute in each bin.
                        # count: show the number of observations in each bin
                        # frequency: show the number of observations divided by the bin width
                        # probability or proportion: normalize such that bar heights sum to 1
                        # percent: normalize such that bar heights sum to 100
                        # density: normalize such that the total area of the histogram equals 1

                        fig = sns.histplot(data=data_new[data_new['n_slice'] == j].sort_values(by=name, na_position='last', ascending=True),
                                           x=name, stat='probability', kde=True, ax=ax)
                        fig = sns.rugplot(data=data_new[data_new['n_slice'] == j].sort_values(by=name, na_position='last', ascending=True), 
                                          x=name, ax=ax)
                    else:
                        fig = sns.histplot(data=data_new[data_new['n_slice'] == j].sort_values(by=name, na_position='last', ascending=True),
                                            x=name, stat='probability', kde=False, ax=ax)
                    ax.set_ylabel(f'  {j}', rotation=0, size=fontSize)
                    ax.yaxis.set_label_position('right')

                fig.text(-0.1, 5, 'Probability', horizontalalignment='left', verticalalignment='baseline', rotation='vertical', transform=ax.transAxes)
                plot = fig.get_figure()
                file_name = file_names['plot%s' % i]
                save_name = os.path.join(self.out_dir, file_name)
                
                plt.suptitle(self.title, size=fontSize)
                plot.savefig(save_name, dpi=self.dpi)
                print("# {} has been saved in {}.".format(file_name, self.out_dir))


    def draw_boxplot(self, data, parameters):
        """
        Use provided data to draw distplot
        """

        sns.set_style('whitegrid')
        sns.set_theme(font_scale=0.9)

        param_names = []
        param_types = []
        if self.multi_params is not None:
            for x in self.multi_params.split(','):
                if parameters[x]['type'] != 'c':
                    param_names.append(x)
            if len(param_names) == 0:
                raise OptionError("!   Required parameters must not categorical for boxplot.")
        else:
            for k, i in parameters.items():
                if i['type'] != 'c':
                    param_names.append(k)

        for x in param_names:
            param_types.append(parameters[x]['type'])
        
        print("\n# The parameters selected to be visualised: \n#  {}".format(param_names))
        print("\n# The type of selected parameters:  \n#  {}".format(param_types))

        data_new = data.copy()

        for name in parameters.keys():
            per = list(data[name]).count('null')/len(data)
            if per == float(1):
                print("! WARNING: all values in {} are 'null', it will be deleted for the plot!".format(name))
                param_names.pop(param_names.index(name))
            elif per > 0:
                data_new[name].replace('null', np.nan, inplace=True)

        col = data_new.count() == 0
        for i in range(len(col)):
            if col[i] and col.index[i] in param_names:
                print("! WARNING: all values in {} are NaN, it will be deleted for the plot!".format(col.index[i]) )
                param_names.pop(param_names.index((col.index[i])))

        print("#")

        if self.multi_params is None:
            num = 6
            page_num = math.ceil(len(param_names)/num)
            params = locals()
            file_names = locals()
            start = 0
            for i in range(0, page_num):
                fig, axis = plt.subplots(2, 3, sharey=False, sharex=False)
                plt.subplots_adjust(hspace=0.5, wspace=0.4, bottom=0.2)
                title = '\nPage ' + str(i+1) + ' of ' + str(page_num)
                
                if start+num-1 <= len(param_names):
                    params['params%s' % i] = param_names[start:start+num]
                else:
                    params['params%s' % i] = param_names[start:]              
                start = start+num
                file_names['plot%s' % i] = self.file_name.split('.')[0]+str(i)   

                page_plots = min(num, len(params['params%s' % i]))

                row = column = idx = 0
                for idx in range(num):
                    ax = axis[row, column]
                    if idx < page_plots:
                        name = params['params%s' % i][idx]
                        fig = sns.boxplot(data=data_new,
                            x=name, y='n_slice', width=0.5, fliersize=1, linewidth=0.5,
                            orient='h', palette="vlag", ax=ax)
                        if len(name) > 25:
                            name = re.sub(r"(.{25})", "\\1\n", name)
                        if column != 0:
                            fig.set_ylabel('')
                        fig.set_xlabel('\n'+name, rotation=0, size=fontSize)
                    else:
                        ax.axis('off')

                    if column < 2:
                        column += 1
                    else:
                        column = 0
                        row += 1 

                plot = fig.get_figure()
                file_name = file_names['plot%s' % i]
                save_name = os.path.join(self.out_dir, file_name)

                plt.suptitle(self.title, size=fontSize)
                plot.savefig(save_name, dpi=self.dpi)
                print("# {} has been saved in {}.".format(file_name, self.out_dir))
        else:
            page_num = len(param_names)
            file_names = locals()
            start = 0
            for i in range(0, page_num):
                plt.clf()
                title = '\nPage ' + str(i+1) + ' of ' + str(page_num)

                file_names['plot%s' % i] = self.file_name.split('.')[0]+'-'+param_names[i]

                name = param_names[i]
                fig = sns.boxplot(data=data_new,
                    x=name, y='n_slice', width=0.5, fliersize=1, linewidth=0.5,
                    orient='h', palette="vlag")
                if len(name) > 25:
                    name = re.sub(r"(.{25})", "\\1\n", name)
                fig.set_xlabel('\n'+name, rotation=0, size=fontSize)

                plot = fig.get_figure()
                file_name = file_names['plot%s' % i]
                save_name = os.path.join(self.out_dir, file_name)

                plt.suptitle(self.title, size=fontSize)
                plot.savefig(save_name, dpi=self.dpi)
                print("# {} has been saved.".format(file_name))

    def draw_violinplot(self, data, parameters):
        """
        Use provided data to draw distplot
        """

        sns.set_style('whitegrid')
        sns.set_theme(font_scale=0.9)

        param_names = []
        param_types = []
        if self.multi_params is not None:
            for x in self.multi_params.split(','):
                if parameters[x]['type'] != 'c':
                    param_names.append(x)
            if len(param_names) == 0:
                raise OptionError("!   Required parameters must not categorical for violinplot.")
        else:
            for k, i in parameters.items():
                if i['type'] != 'c':
                    param_names.append(k)

        for x in param_names:
            param_types.append(parameters[x]['type'])
        
        print("\n# The parameters selected to be visualised: \n#  {}".format(param_names))
        print("\n# The type of selected parameters:  \n#  {}".format(param_types))

        data_new = data.copy()

        for name in parameters.keys():
            per = list(data[name]).count('null')/len(data)
            if per == float(1):
                print("! WARNING: all values in {} are 'null', it will be deleted for the plot!".format(name))
                param_names.pop(param_names.index(name))
            elif per > 0:
                data_new[name].replace('null', np.nan, inplace=True)

        col = data_new.count() == 0
        for i in range(len(col)):
            if col[i] and col.index[i] in param_names:
                print("! WARNING: all values in {} are NaN, it will be deleted for the plot!".format(col.index[i]) )
                param_names.pop(param_names.index((col.index[i])))

        print("#")

        if self.multi_params is None:
            num = 6
            page_num = math.ceil(len(param_names)/num)
            params = locals()
            file_names = locals()
            start = 0
            for i in range(0, page_num):
                fig, axis = plt.subplots(2, 3, sharey=False, sharex=False)
                plt.subplots_adjust(hspace=0.5, wspace=0.4, bottom=0.2)
                title = '\nPage ' + str(i+1) + ' of ' + str(page_num)
                
                if start+num-1 <= len(param_names):
                    params['params%s' % i] = param_names[start:start+num]
                else:
                    params['params%s' % i] = param_names[start:]              
                start = start+num
                file_names['plot%s' % i] = self.file_name.split('.')[0]+str(i)   

                page_plots = min(num, len(params['params%s' % i]))

                row = column = idx = 0
                for idx in range(num):
                    ax = axis[row, column]
                    if idx < page_plots:
                        name = params['params%s' % i][idx]
                        fig = sns.violinplot(data=data_new,
                            x=name, y='n_slice', 
                            inner="point", width=0.5, fliersize=1, linewidth=0.5,
                            orient='h', palette="vlag", ax=ax)
                        if len(name) > 25:
                            name = re.sub(r"(.{25})", "\\1\n", name)
                        if column != 0:
                            fig.set_ylabel('')
                        fig.set_xlabel('\n'+name, rotation=0, size=fontSize)
                    else:
                        ax.axis('off')

                    if column < 2:
                        column += 1
                    else:
                        column = 0
                        row += 1 

                plot = fig.get_figure()
                file_name = file_names['plot%s' % i]
                save_name = os.path.join(self.out_dir, file_name)

                plt.suptitle(self.title, size=fontSize)
                plot.savefig(save_name, dpi=self.dpi)
                print("# {} has been saved in {}.".format(file_name, self.out_dir))
        else:
            page_num = len(param_names)
            file_names = locals()
            start = 0
            for i in range(0, page_num):
                plt.clf()
                title = '\nPage ' + str(i+1) + ' of ' + str(page_num)

                file_names['plot%s' % i] = self.file_name.split('.')[0]+'-'+param_names[i]

                name = param_names[i]
                fig = sns.violinplot(data=data_new,
                    x=name, y='n_slice', 
                    inner="point", width=0.5, fliersize=1, linewidth=0.5,
                    orient='h', palette="vlag")
                if len(name) > 25:
                    name = re.sub(r"(.{25})", "\\1\n", name)
                fig.set_xlabel('\n'+name, rotation=0, size=fontSize)

                plot = fig.get_figure()
                file_name = file_names['plot%s' % i]
                save_name = os.path.join(self.out_dir, file_name)

                plt.suptitle(self.title, size=fontSize)
                plot.savefig(save_name, dpi=self.dpi)
                print("# {} has been saved.".format(file_name))


    def draw_heatmap(self, data, parameters):
        """
        Use provided data to draw distplot
        """

        sns.set_style('whitegrid')
        sns.set_theme(font_scale=0.9)

        param_names = []
        param_types = []
        for x in self.multi_params.split(','):
            param_names.append(x)

        for x in param_names:
            param_types.append(parameters[x]['type'])
        
        print("\n# The parameters selected to be visualised: \n#  {}".format(param_names))
        print("\n# The type of selected parameters:  \n#  {}".format(param_types))

        data_new = data.copy()
        for name in parameters.keys():
            per = list(data[name]).count('null')/len(data)
            if per == float(1):
                print("! WARNING: all values in {} are 'null', it will be deleted for the plot!".format(name))
                param_names.pop(param_names.index(name))
            elif per > 0:
                data_new[name].replace('null', np.nan, inplace=True)

        col = data_new.count() == 0
        for i in range(len(col)):
            if col[i] and col.index[i] in param_names:
                print("! WARNING: all values in {} are NaN, it will be deleted for the plot!".format(col.index[i]) )
                param_names.pop(param_names.index((col.index[i])))

        print("#")

        if len(param_names) < 2:
            pivot_table = data_new.pivot_table(index='n_slice', columns=param_names[0], values='config_id', aggfunc='count')
        else:
            pivot_table = data_new.pivot_table(index=param_names[0], columns=param_names[1], values='config_id', aggfunc='count')
        fig = sns.heatmap(pivot_table, annot=True, cmap='PuBu', fmt="g") 
        fig.get_figure().savefig(self.save_name, dpi=self.dpi)
        print("# {} has been saved.".format(self.file_name))

    def draw_jointplot(self, data, exp_names, parameters):
        """
        Use provided data to draw jointplot
        """

        dimensions = []
        if self.multi_params not in (None, 'null'):
            for x in self.multi_params.split(','):
                if parameters[x]['type'] != 'c':
                    dimensions.append(x)
                else:
                    raise OptionError("!!Categorical parameter {} is not supportable here.".format(x))

        data = pd.DataFrame(data=data, columns=dimensions)
        # x=list(map(float, data[dimensions[0]]))
        # y=list(map(float, data[dimensions[1]]))
        x=dimensions[0]
        y=dimensions[1]

        fig = sns.jointplot(
            x=x, 
            y=y,
            data=data,
            kind='hex',
            space=0.1,
            ratio=4)
        
        fig.savefig(self.save_name, dpi=self.dpi)
        print("# {} has been saved.".format(self.file_name))

    def count_values(self, column):
        data = list(column)
        count = 0
        for x in data:
            if x not in ('null', None):
                count += 1
        return count

