from __future__ import print_function

import sys, os

def django_manage(pythonpath=None, settings_module="settings"):
    if not pythonpath:
        pythonpath = os.path.dirname(os.path.abspath(sys.argv[0]))
    #sys.path += [pythonpath]
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", settings_module)

    try:
        from django.core.management import execute_from_command_line
        execute_from_command_line(sys.argv)
    except ImportError as django_exception:
        #try:
        #    import importlib
        #    importlib.import_module(os.environ["DJANGO_SETTINGS_MODULE"])
        #except ImportError:
        #    print(django_exception.message, file=sys.stderr)
        #    raise

        print(sys.path)
        raise
