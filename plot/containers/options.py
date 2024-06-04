import os
from plot.errors import OptionError
import re

class Option:
    """
    Class Option handles options of the configurator
    :var general_options: List of variables associated with an Option
       name: option name
       type: option type (see option_types)
       short: short flag
       long: long flag
       section: section (group) to which the option belongs
       vignettes: text for the documentation of the option
       description: general description of the option
    :var option_types: List of option types available
       e: enabler option (activates behaviour)
       i: integer option
       r: real option
       s: string option
       b: boolean option
       p: file path option
       x: script file or function to be parsed. Both should be executable

    :ivar value: option value
    :ivar default: option default value
    """

    general_options = ["name", "type", "short", "long", "section", "vignettes", "description"]

    option_types = ["e", "i", "r", "s", "b", "p", "x"]

    def __init__(self):
        """
        Creates an Option object
        """
        self.value = None
        self.default = None

    @staticmethod
    def check_general_options(obj):
        """
        checks that all options in Options.general_options are defined in obj
        :param obj: Dictionary of options read from the json settings file
        :return: none
        :raises OptionError: if an option is not included in obj
        """
        for key in Option.general_options:
            if not (key in obj.keys()):
                raise OptionError("All crace plot options must define attribute " + key + "!")

    def set_general_options(self, obj):
        """
        Creates variables in Option.general_options list and assigns
        the values for these variables in  obj
        :param obj: Dictionary of variable values, keys are the names
                    in Option.general_options
        """
        for key in Option.general_options:
            if key == "name":
                # In current parameter definition some options have a name starting by .
                # this is not allow in Python
                setattr(self, key, obj[key].replace(".", ""))
            else:
                setattr(self, key, obj[key].replace(".", ""))

    def is_set(self):
        return not (self.value is None)


class IntegerOption(Option):
    """
    class the handles integer crace plot options
    :ivar domain: domain of the option as a two element list [lower, upper]
    """
    def __init__(self, obj):
        """
        Creates an Integer Option
        :param obj: dictionary of the option read from the json setting file
        """
        # add general variables
        super().check_general_options(obj)
        super().set_general_options(obj)

        # other variables
        self.domain = self.parse_domain(obj['domain'])
        self.default = self.parse_default(obj['default'])
        self.value = self.default

    def set_value(self, value):
        """
        sets the value of the option and validates it
        :param value: the value to be set
        :return: none
        """
        self.value = self.parse_value(value, True)

    def parse_value(self, value: str, check: bool = False):
        """
        parses the option value from a string
        :param value: string that has the value of the option
        :param check: boolean that indicates if the value must be validated
        :return: parsed value. If the value is None or empty, the default value is returned
        :raises OptionError: if value is not integer
        """
        # if value is empty return default
        if value is None or value == "":
            v = self.default
        elif isinstance(value, str):
            v = eval(value)
            if not isinstance(v, int):
                raise OptionError("! Option" + self.name + " must be integer but has value/default: " + value)
        else:
            v = value

        if check:
            self.check_value(v)
        return v

    def parse_default(self, value: str):
        """
        parse the default value of the option, no validation is performed
        :param value: string with the default value
        :return: the parsed value (integer) or None if default is None or empty
        """
        if value is None or value == "":
            return None
        elif isinstance(value, str):
            return eval(value)
        else:
            return value

    def check_value(self, value):
        """
        checks a value has the correct type (integer) and is in the option domain
        :param value: value to be checked
        :return: none
        :raises OptionError: if value is no integer o value is not within domain
        """
        if not isinstance(value, int):
            raise OptionError("Option " + self.name + " type error value " + str(value) + " is not int")
        elif (value < self.domain[0]) or (value > self.domain[1]):
            raise OptionError("! Option " + self.name + " value/default : " + str(value) +
                              " is not within its integer domain (" + str(self.domain[0]) + "," +
                              str(self.domain[1]) + ")")

    def parse_domain(self, domain: str):
        """
        parses the domain of an option
        :param domain: string that defines the domain in the format (lower, upper)
        :return: returns a list of size two with first element the lower bound and second element the upper bound [lower, upper]
        :raises OptionError: if the provided domain cannot be parsed, or if the domain is not correct lower > upper
        """
        p = re.compile("(?P<d1>\d+),(?P<d2>\d+)")
        m = p.match(domain.strip("()"))
        if m is None:
            raise OptionError("Domain for integer option " + self.name + " is not correct : " + domain)
        d = [eval(m.group("d1")), eval(m.group("d2"))]
        if d[0] > d[1]:
            raise OptionError("Domain for integer option " + self.name + " is not correct : " + domain)
        return d

    def is_default(self):
        """
        Returns True if value assigned is the default value
        """
        return self.value == self.default


class RealOption(Option):
    """
    class the handles real valued crace plot options

    :ivar domain: domain of the option as a two element list [lower, upper]
    """
    def __init__(self, obj):
        """
        Creates a Real Option object
        :param obj: dictionary of the option read from the json setting file
        """
        # add general variables
        super().check_general_options(obj)
        super().set_general_options(obj)

        # other variables
        self.domain = self.parse_domain(obj['domain'])
        self.default = self.parse_default(obj['default'])
        self.value = self.default

    def set_value(self, value: str):
        """
        sets the value of the option and validates it
        :param value: the value to be set
        :return: none
        """
        self.value = self.parse_value(value, True)

    def parse_value(self, value: str, check: bool = False):
        """
        parses the option value from a string
        :param value: string that has the value of the option
        :param check: boolean that indicates if the value must be validated
        :return: parsed value. If the value is None or empty, the default value is returned
        :raises OptionError: if value is not float
        """
        if value is None or value == "":
            return self.default
        if isinstance(value, str):
            v = float(eval(value))
            if not isinstance(v, float):
                raise OptionError("! Option " + self.name + " must be real but has value/default: " + value)

        else:
            v = value

        if check:
            self.check_value(v)
        return v

    def parse_default(self, value: str):
        """
        parse the default value of the option, no validation is performed
        :param value: string with the default value
        :return: the parsed value (float) or None if default is None or empty
        """
        if value is None or value == "":
            return None
        elif isinstance(value, str):
            return eval(value)
        else:
            return value

    def check_value(self, value):
        """
        checks a value has the correct type (float) and is in the option domain
        :param value: value to be checked
        :return: none
        :raises OptionError: if value is not float or value is not within domain
        """
        if not isinstance(value, float):
            raise OptionError("Option " + self.name + " type error value " + str(value) + " is not float")
        elif (value < self.domain[0]) or (value > self.domain[1]):
            raise OptionError("! Option " + self.name + " value/default : " + str(value) + " is not within its real domain (" +
                              str(self.domain[0])+","+str(self.domain[1])+")")

    def parse_domain(self, domain: str):
        """
        parses the domain of an option
        :param domain: string that defines the domain in the format (lower, upper)
        :return: returns a list of size two with first element the lower bound and second element the upper bound [lower, upper]
        :raises OptionError: if the provided domain cannot be parsed, or if the domain is not correct lower > upper
        """
        p = re.compile("(?P<d1>[0-9]+\.*[0-9]*),(?P<d2>[0-9]+\.*[0-9]*)")
        m = p.match(domain.strip("()"))
        if m is None:
            raise OptionError("Domain for real option " + self.name + " is not correct :" + domain)
        d = [eval(m.group("d1")), eval(m.group("d2"))]
        if d[0] > d[1]:
            raise OptionError("Domain for real option " + self.name + " is not correct :" + domain)
        return d

    def is_default(self):
        """
        Returns True if value assigned is the default value
        """
        return self.value == self.default


class StringOption(Option):
    """
    class the handles string crace plot options

    :ivar domain: domain of the option as list [value1, value2, value3,...]
    """
    def __init__(self, obj):
        """
        Creates a String Option object
        :param obj: dictionary of the option read from the json setting file
        """
        # add general variables
        super().check_general_options(obj)
        super().set_general_options(obj)

        # other variables
        # if obj['name'] in ('title', 'fileName', 'keyParameter', 'catx', 'caty'):
        if 'domain' not in obj.keys():
            self.domain = None
        else:
            self.domain = self.parse_domain(obj['domain'])
        self.default = self.parse_default(obj['default'])
        self.value = self.default

    def set_value(self, value: str):
        """
        sets the value of the option and validates it
        :param value: the value to be set
        :return: none
        """
        self.value = self.parse_value(value, True)

    def parse_value(self, value: str, check: bool = False):
        """
        parses the option value from a string
        :param value: string that has the value of the option
        :param check: boolean that indicates if the value must be validated
        :return: parsed value. If the value is None or empty, the default value is returned
        """
        if value is None or value == "":
            return self.default
        v = str(value)
        if check:
            self.check_value(v)
        return v

    def parse_default(self, value: str):
        """
        parse the default value of the option, no validation is performed
        :param value: string with the default value
        :return: the parses value or None if default is None or empty
        """
        if value is None or value == "":
            return None
        elif not isinstance(value, str):
            return str(value)
        else:
            return value

    def check_value(self, value):
        """
        checks a value is in the option domain
        :param value: value to be checked
        :return: none
        :raises OptionError: if value is not string or value is not within domain
        """
        if not isinstance(value, str):
            raise OptionError("Option " + self.name + " type error value " + value + " is not string")
        elif self.domain is None:
            return
        elif not (value in self.domain):
            raise OptionError("! Option " + self.name + " value/default: " + str(value) + " is not within the domain (" + ", ".join(self.domain) + ")")

    def parse_domain(self, domain: str):
        """
        parses the domain of an option
        :param domain: string that defines the domain in the format (value1, value2, value3, ...)
        :return: returns a list of size N  with the different values [value1, value2, value3, ..., valueN]
        :raises OptionError: if the provided domain has repeated values or is empty
        """
        d = domain.strip("()").split(",")
        # TODO: how to validate string crace plot option domains when some options do not have domain
        if d[0] == "":
            return None
        #if d[0]=="":
        #    raise OptionError("Domain for string option " + self.name + " is empty or cannot be parsed")
        if len(d) > len(set(d)):
            raise OptionError("Domain for string option " + self.name + " has repeated values in domain: " + domain)
        d = [x.strip()for x in d]
        return d

    def is_default(self):
        """
        Returns True if value assigned is the default value
        """
        return self.value == self.default


class BooleanOption(Option):
    """
    class the handles boolean crace plot options

    """

    def __init__(self, obj):
        """
        Creates a Boolean Option object
        :param obj: dictionary of the option read from the json setting file
        """
        # add general variables
        super().check_general_options(obj)
        super().set_general_options(obj)

        # other variables
        self.default = self.parse_default(obj['default'])
        self.value = self.default

    def set_value(self, value: str):
        """
        sets the value of the option
        :param value: the value to be set
        :return: none
        """
        self.value = self.parse_value(value)

    def parse_value(self, value: str):
        """
        parses the option value from a string
        :param value: string that has the value of the option
        :return: parsed value. If the value is None or empty, the default value is returned
        """
        if value is None or value == "":
            return self.default
        elif not isinstance(value, bool):
            if isinstance(value, int):
                return bool(value)
            elif value in ("true", "TRUE"):
                return bool(True)
            elif value in ("false", "FALSE"):
                return bool(False)
            else:
                return bool(eval(value))
        else:
            return value

    def parse_default(self, value: str):
        """
        parse the default value of the option
        :param value: string with the default value
        :return: the parse value or None if default is None or empty
        """
        if value is None or value == "":
            return None
        elif not isinstance(value, bool):
            if isinstance(value, int):
                return bool(value)
            else:
                return bool(eval(value))
        else:
            return value

    def is_default(self):
        """
        Returns True if value assigned is the default value
        """
        return self.value == self.default


class FileOption(Option):
    """
    class the handles file and directory path crace plot options

    """
    def __init__(self, obj):
        """
        Creates a File Option object
        :param obj: dictionary of the option read from the json setting file
        """
        # add general variables
        super().check_general_options(obj)
        super().set_general_options(obj)

        # other variables
        self.input_value = obj['default']
        self.default = self.parse_default(obj['default'])
        self.value = self.default

    def set_value(self, value: str, check_file: bool = False, check_parent_dir: bool=True):
        """
        sets the value of the option and validates it
        :param value: the value to be set
        :param check_file: boolean that indicated if the file should be validated
        :return: none
        """
        self.input_value = value.strip()
        self.value = self.parse_value(value, check_file, check_parent_dir)

    def parse_value(self, value: str, check_file: bool = False, check_parent_dir: bool = True):
        """
        parses the option file or directory path from a string.
        :param value: string that has the path
        :param check_file: boolean that indicates the type of validation to be performed:
        - False: only the directory where the file/directory is located should be validated as existing
        - True: the file or directory should be validated as existing
        :return: parsed absolute path. If the value is None or empty, the default value is returned
        """
        if value is None or value == "":
            return None
        value = os.path.expanduser(value)
        fvalue = os.path.abspath(value)
        self.check_value(fvalue, check_file, check_parent_dir)
        return fvalue

    def parse_default(self, value: str):
        """
        parse the default path of the option, no validation is performed
        :param value: string with the default path
        :return: the parsed absolute path or None if default is None or empty
        """
        if value is None or value == "":
            return None
        fvalue = os.path.abspath(value)
        return fvalue

    def check_value(self, value, check_file: bool = False, check_parent_dir:bool =True):
        """
        checks if the file or directory provided exists
        :param value: path to be checked
        :param check_file:  boolean that indicates the type of validation to be performed:
        - False: only the directory where the file/directory is located should be validated as existing
        - True: the file or directory should be validated as existing
        :return: none
        :raises OptionError: if the directory/file does not exists
        """
        if value is None:
            return
        if value == "":
            raise OptionError("Option " + self.name + ": provided file " + value + " is not valid.")
        if check_file:
            if not os.path.exists(value):
                raise OptionError("Option " + self.name + ": provided file " + value + " does not exist.")
        elif check_parent_dir:
            dir_path = os.path.dirname(value)
            if not os.path.exists(dir_path):
                raise OptionError("Option " + self.name + ": provided directory " + dir_path + " does not exist (option value " + value + ")")

    def exists_file(self):
        """
        checks if the file or directory set as option value exists
        :return: True if the value file or directory exists, else False
        """
        if os.path.exists(self.value):
            return True
        return False

    def check_value_dir(self, writable: bool = False):
        """
        checks if the parent directory of the file/directory set as option value exists and its writable
        :param writable: boolean that indicates if the directory should be checked for writing permissions
        :return: none
        :raises OptionError: if the directory does not exists or does not have writing permissions
        """
        dir_path = os.path.dirname(self.value)
        if not os.path.exists(dir_path):
            raise OptionError("Option " + self.name + ": directory " + dir_path + " does not exist (option value " + self.value + ")")
        if writable and (not os.access(dir_path, os.W_OK)):
            raise OptionError("Option " + self.name + ": directory " + dir_path + " is not writable (option value " + self.value + ")")

    def check_value_file(self, writable: bool = False):
        """
        checks if the file set as option value exists and its writable
        :param writable: boolean that indicates if the file should be checked for writing permissions
        :return: none
        :raises OptionError: if the file does not exists or does not have writing permissions
        """
        if not os.path.exists(self.value):
            raise OptionError("Option " + self.name + ": provided file " + self.value + " does not exist.")
        if writable and (not os.access(self.value, os.W_OK)):
            raise OptionError("Option " + self.name + ": provided directory " + self.value + " is not writable")


class ExeOption(Option):
    """
    class the handles executable file path crace plot options

    """
    def __init__(self, obj):
        """
        Creates and Executable Option
        :param obj: dictionary of the option read from the json setting file
        """
        # add general variables
        super().check_general_options(obj)
        super().set_general_options(obj)

        # other variables
        self.default = self.parse_default(obj['default'])
        self.value = self.default

    def set_value(self, value: str):
        """
        sets the value of the option and validates it
        :param value: the value to be set
        :return: none
        """
        self.value = self.parse_value(value, True)

    def parse_value(self, value: str, check: bool = False):
        """
        parses the option file path from a string.
        :param value: string that has the path
        :param check: boolean that indicates if validation should be performed:
        :return: parsed absolute path. If the value is None or empty, the default value is returned
        """
        if value is None or value == "":
            return self.default
        if check:
            check_return,value_ = self.check_value(value)
            if check_return:
                return value_
        value = os.path.expanduser(value)
        fvalue = os.path.abspath(value)
        return fvalue

    def parse_default(self, value: str):
        """
        parse the default path of the option, no validation is performed
        :param value: string with the default path
        :return: the parsed absolute path or None if default is None or empty
        """
        if value is None or value == "":
            return None
        return str(value)

    def check_value(self, value):
        """
        checks if the file provided exists and it has execution permissions
        :param value: path to be checked
        :return: none
        :raises OptionError: if the file does not exists or does not have execution permissions
        """
        if re.search('def', value) == None:
            if not os.path.isfile(value):
                raise OptionError("Option " + self.name + " provided file " + value + " does not exist.")
            if not os.access(value, os.X_OK):
                raise OptionError("Option " + self.name + " provided file " + value + " does  not have correct permissions.")
        else:
            try:
                exec(value)
                return True,value
            except SyntaxError:
                raise OptionError("Option " + self.name + " provided function " + value +" has syntax errors.")


class EnablerOption(Option):
    """
    class the handles crace plot options that signal execution modes (e.g. --help, --onlytest, --check, --version)

    """
    def __init__(self, obj):
        """
        Creates an Enabler Option object
        :param obj: dictionary of the option read from the json setting file
        """
        # add general variables
        super().check_general_options(obj)
        super().set_general_options(obj)

        # other variables
        self.value = False
        self.default = False

    def set_value(self, value: bool = True):
        """
        sets a boolean representing if the option was selected
        :param value: boolean that indicated if the option was encountered
        :return: none
        """
        self.value = value
