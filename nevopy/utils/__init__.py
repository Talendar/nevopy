# MIT License
#
# Copyright (c) 2020 Gabriel Nogueira (Talendar)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================

""" Exposes the main utility functions and classes within this package.
"""

# From `deprecation.py`
from nevopy.utils.deprecation import deprecated

# From `gym_utils.py`:
from nevopy.utils.gym_utils import GymEnvFitness

# From `utils.py`:
from nevopy.utils.utils import align_lists
from nevopy.utils.utils import chance
from nevopy.utils.utils import clear_output
from nevopy.utils.utils import Comparable
from nevopy.utils.utils import is_jupyter_notebook
from nevopy.utils.utils import make_table_row
from nevopy.utils.utils import make_xor_data
from nevopy.utils.utils import min_max_norm
from nevopy.utils.utils import pickle_load
from nevopy.utils.utils import pickle_save
from nevopy.utils.utils import rank_prob_dist
