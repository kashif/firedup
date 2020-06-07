version_info = (0, 1, 0)
# format:
# ('fireup_major', 'fireup_minor', 'fireup_patch')


def get_version():
    "Returns the version as a human-format string."
    return "%d.%d.%d" % version_info


__version__ = get_version()
