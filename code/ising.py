
import argparse
import json
import logging

def argument_parser():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug")
    arguments = parser.parse_args()
    return arguments

if __name__ == "__main__":
    arguments = argument_parser()
    print(arguments)
