import copy
import json
import os
import re
import sys
import pandas as pd

from plot.containers.read_options import ReadOptions
from plot.errors import OptionError

class ReadResults:
    def __init__(self, folders, options: ReadOptions):
        """
        Read Crace results from execDir

        :param num_repetitions: the number of repetitions of Crace in the given directory
        :param exec_dir: the directory stored Crace results
        """
        self.folders = folders
        self.dataType = options.dataType.value

        self.elite_log = "race_log/elite.log"
        self.config_log = "race_log/config.log"
        self.params = "race_log/parameters.log"
        self.all_elites = "elites.log"
        self.exps_fin = ""
        self.slice_log = "race_log/slice.log"

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

        # select exact way to load Crace results via :param{drawMethod}
        # e.g.: heatmap for config_data; boxplot for performance
        self.draw_method = options.drawMethod.value

    def load_for_perfomance(self):
        """
        Read elite configurations (the best or the top 5 ) in Crace training results
        Tips: the quality of each experiments on different configurations
        """

        self.exps_fin = "race_log/exps_fin.log" if self.results_from == "training" else "race_log/test/exps_fin.log"

        if self.num_config > 0:
            all_data, exp_names, elite_ids = self.elitist_quality() if self.num_config == 1 else self.elites_quality()
            return all_data, exp_names, elite_ids
    
    def load_for_parameters(self):
        """
        Read configurations (the best or the top 5 ) in Crace results
        Tips: the value of parameters for each configuration
        """

        self.parameters = self.parse_parameters()

        if self.num_config == 1:
            all_configs, config_ids, elite_ids = self.elitist_config_parameters() 
        elif self.num_config == 5:
            all_configs, config_ids, elite_ids = self.elite_config_parameters()
        elif self.num_config == -1:
            all_configs, config_ids, elite_ids = self.all_elite_config_parameters()
        elif self.num_config == -5:
            all_configs, config_ids, elite_ids = self.all_config_parameters()
        else:
            all_configs, config_ids, elite_ids = self.else_config_parameters()

        return all_configs, config_ids, elite_ids, self.parameters

    def load_for_configurations(self):
        """
        Read configurations (all the elitist or the final 5 ) in Crace results
        Tips: the quality of each configuration
        """

        if self.results_from == "training":
            # all elite configurations run on the whole training instances 
            self.exps_fin = "race_log/test/exps_elites_train.log" 
        elif self.results_from == "test":
            # all elite configurations run on the whole test instances
            self.exps_fin = "race_log/test/exps_elites_test.log" 
        else:
            # all elite configurations from the process
            self.exps_fin = "race_log/exps_fin.log"

        if self.num_config == 5:
            all_data, config_ids, elite_ids = self.elites_config_process()
        elif self.num_config == -1:
            all_data, config_ids, elite_ids = self.all_elite_config_process()
        else:
            raise OptionError("The parameter 'num-configurations' can only be 'elites' or 'allelites'.")

        return all_data, config_ids, elite_ids

    def elites_quality(self):
        """
        Read Crace results of the at most top5 elite configurations
        """

        all_data = pd.DataFrame()
        exp_names = []
        elite_ids = {}

        for folder in self.folders:
            name = os.path.basename(folder)
            exp_names.append(name)
            elite_ids[name] = []
            i = 0
            with open(os.path.join(folder, self.elite_log), "r") as f1:
                for line in f1:
                    if i > 0 and i <= self.num_config:
                        elite_id = int(line.split(',')[-2])
                        elite_ids[name].append(elite_id)
                    i += 1
            f1.close()

            chunk_size = 5000
            tmp = []
            with open(os.path.join(folder, self.exps_fin), "r") as f2:
                line = f2.readline()
                while line:
                    chunk_size -= 1
                    line_results = json.loads(line)
                    current_id = int(line_results["configuration_id"])
                    current_quality = float(line_results["quality"])
                    if current_id in elite_ids[name]:
                        quality_dict = {'exp_name': name, 'config_id': current_id, 'quality': current_quality}
                        tmp.append(quality_dict)
                    line = f2.readline()
                    if not line or chunk_size == 0:
                        tmp = pd.DataFrame(tmp)
                        all_data = pd.concat([all_data, tmp], ignore_index=True)
                        tmp = []
                        chunk_size = 5000
            f2.close()
        return all_data, exp_names, elite_ids
        
    def elitist_quality(self):
        """
        Read Crace results for only the best configuration
        """
        all_data = pd.DataFrame()
        exp_names = []
        elitist_ids = {}
        elitist_ids['elitist'] = []
        elite_ids = {}

        for folder in self.folders:
            name = os.path.basename(folder)
            exp_names.append(name)
            elite_ids[name] = []
            i = 0
            with open(os.path.join(folder, self.elite_log), "r") as f1:
                for line in f1:
                    if i > 0 and i <= self.num_config :
                        elite_id = int(line.split(',')[-2])
                        elitist_ids['elitist'].append(elite_id)
                        elite_ids[name].append(elite_id)
                    i += 1
            f1.close()

            chunk_size = 5000
            tmp = []
            with open(os.path.join(folder, self.exps_fin), "r") as f2:
                line = f2.readline()
                while line:
                    chunk_size -= 1
                    line_results = json.loads(line)
                    current_id = int(line_results["configuration_id"])
                    current_quality = float(line_results["quality"])
                    if current_id in elite_ids[name]:
                        quality_dict = {'exp_name': name, 'config_id': current_id, 'quality': current_quality}
                        tmp.append(quality_dict)
                    line = f2.readline()
                    if not line or chunk_size == 0:
                        tmp = pd.DataFrame(tmp)
                        all_data = pd.concat([all_data, tmp], ignore_index=True)
                        tmp = []
                        chunk_size = 5000
            f2.close()

        return all_data, exp_names, elitist_ids

    def all_quality_ins(self, folder):
        """
        Read Crace results of the at most top5 elite configurations
        """

        all_data = pd.DataFrame()
        exp_names = []

        name = os.path.basename(folder)
        exp_names.append(name)

        chunk_size = 5000
        tmp = []
        with open(os.path.join(folder, self.exps_fin), "r") as f2:
            line = f2.readline()
            while line:
                chunk_size -= 1
                line_results = json.loads(line)
                current_id = int(line_results["configuration_id"])
                current_ins = int(line_results["instance_id"])
                instance_dict = {'exp_name': name, 'config_id': current_id, 'instance_id': current_ins}
                tmp.append(instance_dict)
                line = f2.readline()
                if not line or chunk_size == 0:
                    tmp = pd.DataFrame(tmp)
                    all_data = pd.concat([all_data, tmp], ignore_index=True)
                    tmp = []
                    chunk_size = 5000
            f2.close()

        n_ins = all_data.groupby(['exp_name', 'config_id']).size().reset_index(name='n_ins')
        new_data = all_data[['exp_name', 'config_id']].drop_duplicates()
        new_data = pd.merge(new_data, n_ins, on=['exp_name', 'config_id'], how='left')

        new_data.sort_values(by='config_id', inplace=True)

        return new_data

    def elite_config_parameters(self):
        """
        Read the at least top 5 configurations from the provided Crace results
        """

        # parameters need to be returned
        all_configs = pd.DataFrame()
        exp_names = []
        elite_ids = {}

        for folder in self.folders:
            name = os.path.basename(folder)
            exp_names.append(name)
            elite_ids[name] = []
            config_ids = set()
            i = 0
            # read elite_log file to get the elitist configuration id
            with open(os.path.join(folder, self.elite_log)) as f1:
                for line in f1:
                    if i > 0 and i <= self.num_config:
                        elite_id = int(line.split(',')[-2])
                        elite_ids[name].append(elite_id)
                    i += 1
            f1.close()

            self.exps_fin = "race_log/exps_fin.log"
            ins_data = self.all_quality_ins(folder)

            # read config_log file to get the details of each elitist configuration 
            chunk_size = 5000
            tmp = []
            with open(os.path.join(folder, self.config_log), 'r') as f2:
                current_line = f2.readline()
                while current_line:
                    chunk_size -= 1
                    config_id = int(current_line.split(',')[0])
                    if config_id not in config_ids and config_id in elite_ids[name]:
                        config_ids.add(config_id)
                        params = re.sub('"','',current_line.split('{')[1].split('}')[0])
                        param_dict = {'exp_name': name, 'config_id': config_id}
                        print(param_dict)
                        for pa in params.split(', '):
                            param_name = pa.split(': ')[0]
                            param_value = pa.split(': ')[1]
                            if 'i' in self.parameters[param_name]['type'] and param_value not in (None, 'null'):
                                param_value = int(pa.split(': ')[1])
                            elif 'r' in self.parameters[param_name]['type'] and param_value not in (None, 'null'):
                                param_value = float(pa.split(': ')[1])
                            param_dict[param_name] = param_value
                        tmp.append(param_dict)
                    current_line = f2.readline()
                    if not current_line or chunk_size == 0:
                        tmp = pd.DataFrame(tmp)
                        all_configs = pd.concat([all_configs, tmp], ignore_index=True)
                        tmp = []
                        chunk_size = 5000
            f2.close()    

        all_configs = pd.merge(ins_data, all_configs, on=['exp_name', 'config_id'], how='right')
        all_configs['n_ins'].fillna(0, inplace=True)
        all_configs['n_ins'] = all_configs['n_ins'].astype(int)

        return all_configs, exp_names, elite_ids

    def all_elite_config_parameters(self):
        """
        Read the at least top 5 configurations from the provided Crace results
        """

        # parameters need to be returned
        all_configs = pd.DataFrame()
        exp_names = []
        elite_ids = {}

        for folder in self.folders:
            name = os.path.basename(folder)
            exp_names.append(name)
            elite_ids[name] = []
            config_ids = set()
            # read elites.log file to get the elitist configuration id
            with open(os.path.join(folder, self.all_elites), "r") as f1:
                for line in f1:
                    elite_ids[name].append(int(re.sub("\D", "", line)))
            f1.close()

            self.exps_fin = "race_log/exps_fin.log"
            ins_data = self.all_quality_ins(folder)

            chunk_size = 5000
            tmp = []
            # read config_log file to get the details of each elitist configuration 
            with open(os.path.join(folder, self.config_log), 'r') as f2:
                line = f2.readline()
                while line:
                    config_id = int(line.split(',')[0])
                    if config_id not in config_ids and config_id in elite_ids[name]:
                        config_ids.add(config_id)
                        params = re.sub('"','',line.split('{')[1].split('}')[0])
                        param_dict = {'exp_name': name, 'config_id': config_id}
                        for pa in params.split(', '):
                            param_name = pa.split(': ')[0]
                            param_value = pa.split(': ')[1]
                            if 'i' in self.parameters[param_name]['type'] and param_value not in (None, 'null'):
                                param_value = int(pa.split(': ')[1])
                            elif 'r' in self.parameters[param_name]['type'] and param_value not in (None, 'null'):
                                param_value = float(pa.split(': ')[1])
                            param_dict[param_name] = param_value
                        tmp.append(param_dict)
                    line = f2.readline()
                    if not line or chunk_size == 0:
                        tmp = pd.DataFrame(tmp)
                        all_configs = pd.concat([all_configs, tmp], ignore_index=True)
                        tmp = []
                        chunk_size = 5000
            f2.close()

        all_configs = pd.merge(ins_data, all_configs, on=['exp_name', 'config_id'], how='right')
        all_configs['n_ins'].fillna(0, inplace=True)
        all_configs['n_ins'] = all_configs['n_ins'].astype(int)

        return all_configs, exp_names, elite_ids

    def elitist_config_parameters(self):
        """
        Read the best configuration from the provided Crace results
        """

        # parameters need to be returned
        all_configs = pd.DataFrame()
        exp_names = []
        elite_ids = {}
        elitist_ids = {}
        elitist_ids['elitist'] = []

        for folder in self.folders:
            name = os.path.basename(folder)
            exp_names.append(name)
            elite_ids[name] = []
            config_ids = set()
            i = 0
            # read elite_log file to get the elitist configuration id
            with open(os.path.join(folder, self.elite_log)) as f1:
                for line in f1:
                    if i == 1:
                        elite_id = int(line.split(',')[-2])
                        elite_ids[name].append(elite_id)
                        elitist_ids['elitist'].append(elite_id)
                        break
                    i += 1
            f1.close()

            self.exps_fin = "race_log/exps_fin.log"
            ins_data = self.all_quality_ins(folder)

            # read config_log file to get the details of each elitist configuration 
            chunk_size = 5000
            tmp = []
            with open(os.path.join(folder, self.config_log), 'r') as f2:
                line = f2.readline()
                while line:
                    config_id = int(line.split(',')[0])
                    if config_id not in config_ids and config_id in elite_ids[name]:
                        config_ids.add(config_id)
                        params = re.sub('"','',line.split('{')[1].split('}')[0])
                        param_dict = {'exp_name': name, 'config_id': config_id}
                        for pa in params.split(', '):
                            param_name = pa.split(': ')[0]
                            param_value = pa.split(': ')[1]
                            if 'i' in self.parameters[param_name]['type'] and param_value not in (None, 'null'):
                                param_value = int(pa.split(': ')[1])
                            elif 'r' in self.parameters[param_name]['type'] and param_value not in (None, 'null'):
                                param_value = float(pa.split(': ')[1])
                            param_dict[param_name] = param_value
                        tmp.append(param_dict)
                    line = f2.readline()
                    if not line or chunk_size == 0:
                        tmp = pd.DataFrame(tmp)
                        all_configs = pd.concat([all_configs, tmp], ignore_index=True)
                        tmp = []
                        chunk_size = 5000
            f2.close()

        all_configs = pd.merge(ins_data, all_configs, on=['exp_name', 'config_id'], how='right')
        all_configs['n_ins'].fillna(0, inplace=True)
        all_configs['n_ins'] = all_configs['n_ins'].astype(int)

        return all_configs, exp_names, elitist_ids

    def all_config_parameters(self):
        """
        Read the at least top 5 configurations from the provided Crace results
        """

        # parameters need to be returned
        all_configs = pd.DataFrame()
        exp_names = []
        elite_ids = {}
        slice_ids = {}

        for folder in self.folders:
            name = os.path.basename(folder)
            exp_names.append(name)
            elite_ids[name] = []
            config_ids = set()
            i = 0
            # read elite_log file to get the elitist configuration id
            with open(os.path.join(folder, self.elite_log)) as f1:
                for line in f1:
                    if i > 0 and i <= 5:
                        elite_id = int(line.split(',')[-2])
                        elite_ids[name].append(elite_id)
                    i += 1
            f1.close()

            self.exps_fin = "race_log/exps_fin.log"
            ins_data = self.all_quality_ins(folder)

            slice_ids[name] = []
            with open(os.path.join(folder, self.slice_log)) as f1:
                for line in f1:
                    # brief slice
                    # model_id = int(line.split(',')[-2].split(' ')[-1])    # no restart
                    model_id = int(line.split(',')[-2].split('/')[0].split(' ')[-1])    # with restart
                    slice_id = int(line.split(' ')[-1])
                    if (len(slice_ids[name]) < 1 and slice_id not in slice_ids[name] 
                        or slice_id != slice_ids[name][-1][1]):
                        slice_ids[name].append([model_id,slice_id])
                    else:
                        slice_ids[name][-1][0] = model_id
                    # all slices
                    # slice_id = int(line.split(' ')[-1])
                    # slice_ids[name].append(slice_id)
            f1.close()
            print(slice_ids[name])
            i_slice = 0
            # read config_log file to get the details of each elitist configuration 
            # this file may be a big file
            chunk_size = 5000
            tmp = []
            with open(os.path.join(folder, self.config_log), 'r') as f2:
                line = f2.readline()
                while line:
                    chunk_size -= 1
                    config_id = int(line.split(',')[0])
                    if config_id not in config_ids:
                        config_ids.add(config_id)
                        # check slice
                        # if i_slice < len(slice_ids[name]) and config_id > slice_ids[name][i_slice]:
                        if i_slice < len(slice_ids[name]) and config_id > slice_ids[name][i_slice][1]:
                            i_slice += 1
                            print(i_slice+1, config_id)
                        params = re.sub('"','',line.split('{')[1].split('}')[0])
                        # all slices
                        param_dict = {'exp_name': name, 'n_slice': i_slice+1, 'config_id': config_id}
                        for pa in params.split(', '):
                            param_name = pa.split(': ')[0]
                            param_value = pa.split(': ')[1]
                            if 'i' in self.parameters[param_name]['type'] and param_value not in (None, 'null'):
                                param_value = int(pa.split(': ')[1])
                            elif 'r' in self.parameters[param_name]['type'] and param_value not in (None, 'null'):
                                param_value = float(pa.split(': ')[1])
                            param_dict[param_name] = param_value
                        tmp.append(param_dict)
                    line = f2.readline()
                    if not line or chunk_size == 0:
                        tmp = pd.DataFrame(tmp)
                        all_configs = pd.concat([all_configs, tmp], ignore_index=True)
                        tmp = []
                        chunk_size = 5000
            f2.close()

        all_configs = pd.merge(ins_data, all_configs, on=['exp_name', 'config_id'], how='right')
        all_configs['n_ins'].fillna(0, inplace=True)
        all_configs['n_ins'] = all_configs['n_ins'].astype(int)

        return all_configs, exp_names, elite_ids

    def else_config_parameters(self):
        """
        Read the at least top 5 configurations from the provided Crace results
        """

        # FIXME: how to improve the efficiency when analysing a big file

        # parameters need to be returned
        all_configs = pd.DataFrame()
        exp_names = []
        elite_ids = {}

        for folder in self.folders:
            name = os.path.basename(folder)
            exp_names.append(name)
            elite_ids[name] = []
            config_ids = set()
            i = 0
            # read elite_log file to get the elitist configuration id
            with open(os.path.join(folder, self.elite_log)) as f1:
                for line in f1:
                    if i > 0 and i <= self.num_config:
                        elite_id = int(line.split(',')[-2])
                        elite_ids[name].append(elite_id)
                    i += 1
            f1.close()

            self.exps_fin = "race_log/exps_fin.log"
            ins_data = self.all_quality_ins(folder)

            # firstly read the elite configurations
            chunk_size = 5000
            tmp = []
            i = 0
            # read config_log file to get the details of each elitist configuration 
            with open(os.path.join(folder, self.config_log), 'r') as f2:
                line = f2.readline()
                while line and i < len(elite_ids[name]):
                    config_id = int(line.split(',')[0])
                    if config_id not in config_ids and config_id in elite_ids[name]:
                        i += 1
                        config_ids.add(config_id)
                        params = re.sub('"','',line.split('{')[1].split('}')[0])
                        param_dict = {'exp_name': name, 'config_id': config_id}
                        for pa in params.split(', '):
                            param_name = pa.split(': ')[0]
                            param_value = pa.split(': ')[1]
                            if 'i' in self.parameters[param_name]['type'] and param_value not in (None, 'null'):
                                param_value = int(pa.split(': ')[1])
                            elif 'r' in self.parameters[param_name]['type'] and param_value not in (None, 'null'):
                                param_value = float(pa.split(': ')[1])
                            param_dict[param_name] = param_value
                        tmp.append(param_dict)
                    line = f2.readline()
                    if i == len(elite_ids[name]) or not line or chunk_size == 0:
                        tmp = pd.DataFrame(tmp)
                        all_configs = pd.concat([all_configs, tmp], ignore_index=True)
                        tmp = []
                        chunk_size = 5000
            f2.close()

            chunk_size = 5000
            tmp = []
            # secondly read the other configurations not in the elites list
            lines = self.tail_file(file_path=os.path.join(folder, self.config_log), 
                                   n_lines=self.num_config-1)
            for line in lines:
                config_id = int(line.split(',')[0])
                if config_id not in config_ids:
                    i += 1
                    config_ids.add(config_id)
                    params = re.sub('"','',line.split('{')[1].split('}')[0])
                    param_dict = {'exp_name': name, 'config_id': config_id}
                    for pa in params.split(', '):
                        param_name = pa.split(': ')[0]
                        param_value = pa.split(': ')[1]
                        if 'i' in self.parameters[param_name]['type'] and param_value not in (None, 'null'):
                            param_value = int(pa.split(': ')[1])
                        elif 'r' in self.parameters[param_name]['type'] and param_value not in (None, 'null'):
                            param_value = float(pa.split(': ')[1])
                        param_dict[param_name] = param_value
                    tmp.append(param_dict)
            tmp = pd.DataFrame(tmp)
            all_configs = pd.concat([all_configs, tmp], ignore_index=True)

        all_configs = pd.merge(ins_data, all_configs, on=['exp_name', 'config_id'], how='right')
        all_configs['n_ins'].fillna(0, inplace=True)
        all_configs['n_ins'] = all_configs['n_ins'].astype(int)

        return all_configs, exp_names, elite_ids

    def elites_config_process(self):
        """
        Read the quality of the final elite configurations sampled at the the process
        """

        all_configs = pd.DataFrame()
        exp_names = None

        folder = self.folders[0]
        name = os.path.basename(folder)
        dirname = os.path.basename(os.path.dirname(folder))
        exp_names = dirname + '/' + name
        elite_ids = []
        i = 0
        with open(os.path.join(folder, self.elite_log), "r") as f1:
            for line in f1:
                if i > 0 and i <= self.num_config:
                    elite_id = int(line.split(',')[-2])
                    elite_ids.append(elite_id)
                i += 1
        f1.close()
        
        tmp = []
        print("#   exps file:", os.path.join(folder, self.exps_fin))
        # read config_log file to get the details of each elitist configuration 
        with open(os.path.join(folder, self.exps_fin), 'r') as f2:
            for line in f2:
                line_result = json.loads(line)
                current_id = str(line_result["configuration_id"])
                if int(current_id) in elite_ids:
                    current_ins = str(line_result["instance_id"])
                    current_quality = line_result["quality"]
                    tmp_dict = {'exp_name':name, 'config_id':current_id, 'instance_id':current_ins, 
                                'quality':current_quality}
                    tmp.append(tmp_dict)
            tmp = pd.DataFrame(tmp)
            all_configs = pd.concat([all_configs, tmp], ignore_index=True)
        f2.close()    
        return all_configs, exp_names, elite_ids

    def all_elite_config_process(self):
        """
        Read the quality of all elitist configurations sampled during the process
        """

        # parameters need to be returned
        all_configs = pd.DataFrame()
        exp_names = None
        elite_ids = []

        folder = self.folders[0]
        name = os.path.basename(folder)
        dirname = os.path.basename(os.path.dirname(folder))
        exp_names = dirname + '/' + name
        # read elites.log file to get the elitist configuration id
        with open(os.path.join(folder, self.all_elites), "r") as f1:
            for line in f1:
                elite_ids.append(int(re.sub("\D", "", line)))
        f1.close()
        elitist_id = elite_ids.pop()

        tmp = []
        print("#   exps file:", os.path.join(folder, self.exps_fin))
        # read config_log file to get the details of each elitist configuration 
        with open(os.path.join(folder, self.exps_fin), 'r') as f2:
            for line in f2:
                line_result = json.loads(line)
                current_id = str(line_result["configuration_id"])
                if int(current_id) in elite_ids:
                    current_ins = str(line_result["instance_id"])
                    current_quality = line_result["quality"]
                    # if int(current_id) == elitist_id:
                    #     current_id = "*%(num)s*" % {"num": elitist_id}
                    tmp_dict = {'exp_name':name, 'config_id':current_id, 'instance_id':current_ins, 
                                'quality':current_quality}
                    tmp.append(tmp_dict)
            tmp = pd.DataFrame(tmp)
            all_configs = pd.concat([all_configs, tmp], ignore_index=True)
        f2.close() 

        elite_ids[elite_ids.index(elitist_id)] = "*%(num)s*" % {"num": elitist_id}

        return all_configs, exp_names, elite_ids

    def parse_parameters(self):
        """
        load the range of each parameter in the provided Crace results
        """
        
        # load the information from the provided Crace results
        # Only once
        flag = True
        for folder in self.folders:
            if flag is not True:
                break
            else:
                flag = False
                with open(os.path.join(folder, self.params), "r") as f:
                    line = json.loads(f.read())
                    tmp_params = line[0]
                f.close()
        
        # parse all parameters
        all_parameters = {}
        tmp_params = json.loads(json.dumps(tmp_params))
        for p in tmp_params:
            all_parameters[p['name']] = {}
            all_parameters[p['name']]['type'] = p['type']
            all_parameters[p['name']]['domain'] = p['domain']

        return(all_parameters)

    def tail_file(self, file_path, n_lines=10, block_size=10000):
        """
        Read several lines from the end of the provided file.
        """
        with open(file_path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            end_position = f.tell()
            block_end_position = end_position
            buffer = ''
            lines_found = []

            while len(lines_found) < n_lines and block_end_position > 0:
                block_start_position = max(0, block_end_position - block_size)
                f.seek(block_start_position)
                # read the block and prepend it to the buffer
                buffer = f.read(block_end_position - block_start_position).decode('utf-8') + buffer
                # Update the position for the next block
                block_end_position = block_start_position
                # Split the buffer into lines
                lines = buffer.split('\n')
                # check if there is an incomplete line
                if block_start_position > 0:
                    buffer = lines.pop(0)
                else:
                    buffer = ''
                # Add the lines to the list of found lines
                # make sure not to exceed the required number of lines
                for line in reversed(lines):
                    if line.strip():
                        lines_found.append(line)
                    if len(lines_found) == n_lines:
                        break

        # return the required lines
        # in correct order
        return lines_found[-n_lines:][::-1]  
