import sys
import traceback

from plot.containers.read_options import ReadOptions
from plot.draw.performance import Performance
from plot.draw.parameters import Parameters
from plot.draw.configurations import Configurations

class DrawPlot:
    """
    Class defines the procedure to select one drawMethod for drawing plot
    :ivar drawMethod: the method used to draw plot for provided Crace results
    """
    def __init__(self, reader: ReadOptions ) -> None:

        self.out_dir = reader.outDir.value
        self.drawMethod = reader.drawMethod.value
        self.dataType = reader.dataType.value
        methods = ('parallelcoord', 'parallelcat', 'sunburst')

        try:
            if self.dataType == "performance":
                load = Performance(reader)
            elif self.dataType == "parameters":
                load = Parameters(reader)
            else:
                load = Configurations(reader)
            getattr(load, self.drawMethod)()

            print('#\n# Succeeded! ')
            if self.drawMethod not in methods:
                print('# Now you can check the plot file in', self.out_dir)
            print('#------------------------------------------------------------------------------')

        except:
            print("#\n! There was an error: ")
            err = traceback.format_exc()
            print(err)
            sys.exit()

