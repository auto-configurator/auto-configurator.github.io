import os
import sys
import json

from plot.errors import PlotError, OptionError, FileError, ParameterValueError
from plot.containers.options import IntegerOption, RealOption, StringOption, BooleanOption, FileOption, \
    ExeOption, EnablerOption

def option_decoder(obj):
    """
    instantiates the correct class for an option
    :param obj: option (dictionary) obtained from the json settings definition
    :return: an instance of the Option class
    :raises OptionError: if type attribute is not found in obj or if type was not recognized
    """
    o = None
    if not ('type' in obj):
        raise OptionError("Option has no type.")
    if obj['type'] == 'i':
        o = IntegerOption(obj)
    if obj['type'] == 'r':
        o = RealOption(obj)
    if obj['type'] == 'b':
        o = BooleanOption(obj)
    if obj['type'] == 's':
        o = StringOption(obj)
    if obj['type'] == 'p':
        o = FileOption(obj)
    if obj['type'] == 'x':
        o = ExeOption(obj)
    if obj['type'] == 'e':
        o = EnablerOption(obj)

    if o is None:
        raise OptionError("Option cannot be read: " + obj['name'])
    return o

def print_crace_plot_header():
    # TODO: make version automatic
    print('\n#------------------------------------------------------------------------------')
    print('# crace plot: An implementation in python of a results analysis for Crace')
    print('# Version: 0.1')
    print('# Copyright (C) 2023')
    print('# Yunshuang XIAO    <yunshuang.xiao@ulb.be>')
    print('#')
    print('# This is free software, and you are welcome to redistribute it under certain')
    print('# conditions.  See the GNU General Public License for details. There is NO')
    print('# WARRANTY; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.')
    print('#------------------------------------------------------------------------------')

def print_help():
    """
    Print help information of crace-plot
    """
    print_crace_plot_header()
    print("# called with: --help (-h)")
    package_dir, file = os.path.split(__file__)
    filename = package_dir + "/../settings/plot_options.json"
    with open(filename, "r") as read_file:
        all_options = json.loads(read_file.read(), object_hook=option_decoder)

    options = []
    for o in all_options:
        options.append(o.name)
        if o.short is not None and o.short != "":
            print("#\n# {:<2}, {:<22}{}".format(o.short, o.long, o.description))
        else:
            print("#\n# {:<4}{:<22}{}".format(o.short,o.long, o.description))
        if o.default is not None or o.default != "":
            print("# {:<26}Default: {}".format('', o.default))
        if o.type in ['i', 's']:
            print("# {:<26}Domain: {}".format('', o.domain))
    print('#\n#------------------------------------------------------------------------------')

def print_info():
    print("!   You can check the information of HELP (-h, --help).")

def print_version():
    """
    Print the version information of crace-plot
    """
    print_crace_plot_header()

class ParseOptions:
    """
    Parse arguments from the input
    """
    def from_input(arguments=None, silent = False):
        if not silent:
            options = ReadOptions(arguments=arguments)
        else:
            options = ReadOptions(arguments=arguments)

        return options

class ReadOptions:
    """
    Class contains the set of options that crace plot defined for the results analysis.
    The options are defined in a setting file provided by the package.
    The results are record from an exsiting Crace execution.
    """
    def __init__(self, arguments):
        """
        Initialization of an LoadOptions object. Arguments must be provided.

        :param arguments: command line arguments provided by the user (it is not None)
        """
        
        # TODO: option definition file should be set in the module/package.
        # load options definition from json file
        package_dir, file = os.path.split(__file__)
        filename = package_dir + "/../settings/plot_options.json"
        with open(filename, "r") as read_file:
            all_options = json.loads(read_file.read(), object_hook=option_decoder)

        # create option variables (default values are put in place)
        self.options = []
        for o in all_options:
            self.options.append(o.name)
            setattr(self, o.name, o)

        # arguments is the whole line of input
        self.arguments = arguments

        # load option values
        try:
            print('\n#------------------------------------------------------------------------------')
            print("# Reading crace plot options...")
            print("# Arguments: {}".format(arguments))
            print("#\n# Loading crace plot options...")
            self._load_options(arguments)
            self.check_options()
        except OptionError as err:
            print("#\n! There was an error while reading irace options :")
            print(err)
            print_info()
            sys.exit()
        except FileError as err:
            print("#\n! There was an error while reading irace options file :")
            print(err)
            sys.exit()
        except Exception as err:
            print("#\n! There was an error: ")
            print(err)
            sys.exit()

        print_crace_plot_header()
        self.print_options()


    def _load_options(self, arguments):
        """
        Load option values from arguments 

        :param arguments: command line arguments provided by the user (it is not None here)
        :return: none
        """
        arguments = arguments[1:]

        # # check for help command
        # if self.help.short in arguments or self.help.long in arguments:
        #     print_help()
        #     sys.exit()

        # # check for version command
        # if self.version.short in arguments or self.version.long in arguments:
        #     print_version()
        #     sys.exit()
        
        index = {}
        for o in self.options:
            if self.get_option(o).short in arguments:
                index[o] = arguments.index(self.get_option(o).short)
            elif self.get_option(o).long in arguments:
                index[o] = arguments.index(self.get_option(o).long)
            else:
                continue
        
        for o in index.keys():
            if isinstance(self.get_option(o), EnablerOption): 
                self.set_option(o, True)
                arguments[index[o]] = None
            elif self.get_option(o).short != '-m': 
                self.set_option(o, arguments[index[o]+1])
                arguments[index[o]] = arguments[index[o]+1] = None
            else:
                i = 1
                l = []
                for x in arguments[index[o]+1:]:
                    if (index[o]+i) not in index.values():
                        l.append(x)
                        arguments[index[o] + i] = None
                    else:
                        break
                    i += 1
                self.set_option(o, ','.join(l))
                arguments[index[o]] = None

        for x in arguments:
            if x is not None:
                raise OptionError("!   Argument not recognized: " + x)

    def check_options(self):
        """
        Checks the crace plot options are correct
        :return: True if the options are correct
        :raises OptionError: if there is any error in the option values
        """
        # FIXME: check drawMethod options
        # FIXME: check execDir is avaliable or not

        exp_folders = sorted([subdir for subdir, dirs, files in os.walk(self.execDir.value) \
                      for dir in dirs if dir == 'irace_log' ])
        if len(exp_folders) != self.numRepetitions.value:
            self.numRepetitions.value = len(exp_folders)

        if self.drawMethod.value is not None and self.execDir.value is None:
            raise ParameterValueError("!   The 'execDir' must be provided when 'drawMethod' is not None!")
        elif self.drawMethod.value is None:
            raise ParameterValueError("!   The 'drawMethod' must be provided!")
        elif not os.path.exists(self.execDir.value):
            raise ParameterValueError("!   The directory '{}' dose not exist!".format(self.execDir.value))
        elif self.title.value is None:        
            path_name1 = os.path.basename(self.execDir.value)
            path_name2 = os.path.basename(os.path.dirname(self.execDir.value))
            self.title.value = self.drawMethod.value+': '+path_name2+'/'+path_name1
            
        if self.drawMethod.value not in "(parallelcoord, parallelcat, piechart, \
            scattermatrix, histplot, jointplot)" and self.multiParameters.value is not None:
            raise ParameterValueError("!   Parameter 'multiParameters'(-m) should not be provided here!")
      
        if self.drawMethod.value in "(jointplot, parallelcat)":
            if self.multiParameters.value is None or len(self.multiParameters.value.split(',')) != 2:
                raise ParameterValueError("!   When '{}' is called, two parameter names must be provided!".format(self.drawMethod.value))

        self.fileName.value = os.path.join(self.outDir.value, self.fileName.value)

        if self.configType.value == 'test' and self.numConfigurations.value == 'all':
            raise ParameterValueError("!   There is no 'allConfigurations' in test part!")
        
        if self.numConfigurations.value == 'allelites':
            print(self.execDir.value)
            if not os.path.exists(os.path.join(self.execDir.value, "elites.log")):
                raise ParameterValueError("!   There is no 'elites.log' in exeDir.")

        if self.numConfigurations.value != 'else':
            self.elseNumConfigs.value = 'null'

    def get_option(self, option_name):
        """
        get a option in the LoadOption object
        :param option_name: name of the option to be returned
        :return: value of the option
        :raises OptionError: if the option name is unknown
        """
        if option_name not in self.options:
            raise OptionError("Attempt to get unknown option " + option_name)
        return getattr(self, option_name)

    def set_option(self, option_name, option_value):
        """
        set the value of an option in the LoadOptions object
        :param option_name: name of the option
        :param option_value: value to be set
        :return: none
        :raises OptionError: if the option name is unknown
        """
        if option_name not in self.options:
            raise OptionError("Attempt to set unknown option " + option_name + " with value " + option_value)
        getattr(self, option_name).set_value(option_value)


    def is_set(self):
        return not (self.value is None)

    def print_options(self):
        """
        Print all options
        """
        print('# Crace plot options: ')
        for o in self.options:
            v = self.get_option(o).value
            if self.get_option(o).type != 'e':
                print("#   {}: {}".format(o, v))
        print('#------------------------------------------------------------------------------')



