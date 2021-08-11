import subprocess

# file = "map_menu_struct.py"

# python -m cProfile -o profiling/report.prof couple.py
# subprocess.call("python -m cProfile -o profiling/report.prof {0}".format(file))

# startswith

# snakeviz profiling/report.prof
# subprocess.call("snakeviz profiling/report.prof")

# MEMORY
# from guppy import hpy
# h = hpy()
# h.heap()



import cProfile
import pstats
from functools import wraps
import os
import random


def profile(output_file=None, sort_by='cumulative', lines_to_print=None, strip_dirs=False, main_file_only=True):
    """A time profiler decorator.
    Inspired by and modified the profile decorator of Giampaolo Rodola:
    http://code.activestate.com/recipes/577817-profile-decorator/
    Args:
        output_file: str or None. Default is None
            Path of the output file. If only name of the file is given, it's
            saved in the current directory.
            If it's None, the name of the decorated function is used.
        sort_by: str or SortKey enum or tuple/list of str/SortKey enum
            Sorting criteria for the Stats object.
            For a list of valid string and SortKey refer to:
            https://docs.python.org/3/library/profile.html#pstats.Stats.sort_stats
        lines_to_print: int or None
            Number of lines to print. Default (None) is for all the lines.
            This is useful in reducing the size of the printout, especially
            that sorting by 'cumulative', the time consuming operations
            are printed toward the top of the file.
        strip_dirs: bool
            Whether to remove the leading path info from file names.
            This is also useful in reducing the size of the printout
    Returns:
        Profile of the decorated function
    """

    def inner(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            _output_file = output_file or func.__name__ + '.prof'
            pr = cProfile.Profile()
            pr.enable()
            retval = func(*args, **kwargs)
            pr.disable()
            pr.dump_stats(_output_file)

            # print(func.__name__)
            # print(__name__)
            filename = os.path.basename(__file__)
            print(filename)
            pstats.Stats(pr).print_stats(filename)
            with open(_output_file, 'w') as f:
                ps = pstats.Stats(pr, stream=f)
                if strip_dirs:
                    ps.strip_dirs()
                if isinstance(sort_by, (tuple, list)):
                    ps.sort_stats(*sort_by)
                else:
                    ps.sort_stats(sort_by)
                ps.print_stats(filename, lines_to_print)
            return retval

        return wrapper

    return inner





if __name__ == '__main__':
	random.seed(20)
	def create_products(num):
	    """Create a list of random products with 3-letter alphanumeric name."""
	    return [''.join(random.choices('ABCDEFG123', k=3)) for _ in range(num)]

	# version1
	@profile(sort_by='cumulative', lines_to_print=10, strip_dirs=True)
	def product_counter_v1(products):
	    """Get count of products in descending order."""
	    counter_dict = create_counter(products)
	    sorted_p = sort_counter(counter_dict)
	    return sorted_p

	def create_counter(products):
	    counter_dict = {}
	    for p in products:
	        if p not in counter_dict:
	            counter_dict[p] = 0
	        counter_dict[p] += 1
	    return counter_dict

	def sort_counter(counter_dict):
	    return {k: v for k, v in sorted(counter_dict.items(),
	                                    key=lambda x: x[1],
	                                    reverse=True)}

	# ===========
	# Analysis starts here
	# ===========
	num = 1_000_000  # assume we have sold 1,000,000 products
	products = create_products(num)
	# Let's add profile decorator to product_counter_v1 function
	counter_dict = product_counter_v1(products)