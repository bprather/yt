
import configparser

def parse_parthenon_input(string):
    """Parse the Parthenon parameter file format to produce a Python dict.
    The format is:

    | <header/subheader>
    | name = value
    | name2 = big, list, with &
    |        line, continuations

    This parser relies on two conventions not technically enforced by Parthenon:
    1. Comments begin with `#`
    2. Continuation lines are indented
    """

    # This parser expects [section], and uses indents for line continuation
    # Otherwise it should match the Parthenon input deck spec exactly & flexibly
    config = configparser.ConfigParser(inline_comment_prefixes=(';','#','&'))
    config.read_string(string.replace("<","[").replace(">","]"))
    return config