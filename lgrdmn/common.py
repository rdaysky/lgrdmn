# File encoding: UTF-8

import sys

from .common_all import *

if "django" in sys.modules:
    from .common_django import *

if "django.contrib.gis" in sys.modules:
    from .common_geodjango import *

