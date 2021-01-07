#!/usr/bin/python3

import subprocess

commands = [
	"sphinx-apidoc -f -o ./ ../nevopy",
    "make clean",
    "make html"
]


if __name__ == "__main__":
	for c in commands:
		subprocess.run(c, shell=True)
