import logging
import os
import sys


if sys.platform == "win32":
    config_for_pytensor_cache_script = "cxx=,device=cpu"
    pytensor_flags = (
        os.environ["PYTENSOR_FLAGS"] if "PYTENSOR_FLAGS" in os.environ else ""
    )
    if pytensor_flags:
        pytensor_flags += ","
    pytensor_flags += config_for_pytensor_cache_script
    os.environ["PYTENSOR_FLAGS"] = pytensor_flags

import pytensor
import pytensor.compile.compiledir
from pytensor import config
from pytensor.link.c.basic import get_module_cache


_logger = logging.getLogger("pytensor.bin.pytensor-cache")


def print_help(exit_status):
    if exit_status:
        print(f"command \"{' '.join(sys.argv)}\" not recognized")
    print('Type "pytensor-cache" to print the cache location')
    print('Type "pytensor-cache help" to print this help')
    print('Type "pytensor-cache clear" to erase the cache')
    print('Type "pytensor-cache list" to print the cache content')
    print('Type "pytensor-cache unlock" to unlock the cache directory')
    print(
        'Type "pytensor-cache cleanup" to delete keys in the old ' "format/code version"
    )
    print('Type "pytensor-cache purge" to force deletion of the cache directory')
    print(
        'Type "pytensor-cache basecompiledir" '
        "to print the parent of the cache directory"
    )
    print(
        'Type "pytensor-cache basecompiledir list" '
        "to print the content of the base compile dir"
    )
    print(
        'Type "pytensor-cache basecompiledir purge" '
        "to remove everything in the base compile dir, "
        "that is, erase ALL cache directories"
    )
    sys.exit(exit_status)


def main():
    if len(sys.argv) == 1:
        print(config.compiledir)
    elif len(sys.argv) == 2:
        if sys.argv[1] == "help":
            print_help(exit_status=0)
        if sys.argv[1] == "clear":
            # We skip the refresh on module cache creation because the refresh will
            # be done when calling clear afterwards.
            cache = get_module_cache(init_args=dict(do_refresh=False))
            cache.clear(
                unversioned_min_age=-1, clear_base_files=True, delete_if_problem=True
            )

            # Print a warning if some cached modules were not removed, so that the
            # user knows he should manually delete them, or call
            # pytensor-cache purge, # to properly clear the cache.
            items = [
                item
                for item in sorted(os.listdir(cache.dirname))
                if item.startswith("tmp")
            ]
            if items:
                _logger.warning(
                    "There remain elements in the cache dir that you may "
                    "need to erase manually. The cache dir is:\n  %s\n"
                    'You can also call "pytensor-cache purge" to '
                    "remove everything from that directory." % config.compiledir
                )
                _logger.debug(f"Remaining elements ({len(items)}): {', '.join(items)}")
        elif sys.argv[1] == "list":
            pytensor.compile.compiledir.print_compiledir_content()
        elif sys.argv[1] == "cleanup":
            pytensor.compile.compiledir.cleanup()
            cache = get_module_cache(init_args=dict(do_refresh=False))
            cache.clear_old()
        elif sys.argv[1] == "unlock":
            pytensor.compile.compilelock.force_unlock(config.compiledir)
            print("Lock successfully removed!")
        elif sys.argv[1] == "purge":
            pytensor.compile.compiledir.compiledir_purge()
        elif sys.argv[1] == "basecompiledir":
            # Simply print the base_compiledir
            print(pytensor.config.base_compiledir)
        else:
            print_help(exit_status=1)
    elif len(sys.argv) == 3 and sys.argv[1] == "basecompiledir":
        if sys.argv[2] == "list":
            pytensor.compile.compiledir.basecompiledir_ls()
        elif sys.argv[2] == "purge":
            pytensor.compile.compiledir.basecompiledir_purge()
        else:
            print_help(exit_status=1)
    else:
        print_help(exit_status=1)


if __name__ == "__main__":
    main()
