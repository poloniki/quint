#!/usr/bin/env python3

import argparse
import logging
import os
import requests
import sys

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s]: %(message)s'
)
logger = logging.getLogger(
    name="chunk_paragraphs",
)

argparser = argparse.ArgumentParser(
    description="chunk_paragraphs.py takes in text or a text file and runs it through Quint's paragraph chunker.",
    epilog="example: python chunk_paragraphs.py -h 192.168.1.102:8000 -i ./input_directory/ -o ./output_directory/"
)
argparser.add_argument(
    '-H',
    metavar="IP:PORT",
    type=str,
    help="The IP and PORT of your Quint API server",
    required=True
)
argparser.add_argument(
    '-i',
    metavar="INPUT",
    type=str,
    help="Single file or directory with text files",
    required=True
)
argparser.add_argument(
    '-o',
    metavar="OUTPUT",
    type=str,
    help="Single file or directory to put processed text",
    required=True
)
args = argparser.parse_args()
host: str
port: str


def chunk_paragraphs_file(input_file: str, output: str):
    logger.info("Reading %s...", input_file)
    with open(input_file, "r") as input_file:
        _input_contents: str = input_file.read()

    _input_contents = _input_contents.encode("ascii", errors="ignore").decode().replace("\r\n", " ").replace("\n", " ")
    logger.info("Sending to API server on %s...", args.H)
    r = requests.post("http://" + args.H + "/chunk", json={
        "body": _input_contents
    })

    try:
        logger.info("Writing to %s...", output)
        with open(output, "w+") as output_file:
            output_file.write("\n\n".join([i.strip() for i in r.json()["output"]]))
        logger.info("Done!")
    except:
        logger.error("Error, skipping file %s", _file)
        with open(output_dir + "\\failed_files.txt", "a+") as failed_files:
            failed_files.write(_file + "\n")


def chunk_paragraphs_dir(input_dir: str, output_dir: str):
    _files = os.listdir(input_dir)
    logger.info("Found %s files in %s", len(_files), input_dir)
    for _idx, _file in enumerate(_files):
        if _file[-3:] != "txt":
            logger.info("%s: Not txt file. Skipping...", _file)
            continue
        _out_file = f"{_file}_out.txt"
        with open(f"{input_dir}\{_file}", "r") as input_file:
            _input_contents: str = input_file.read().encode("ascii", errors="ignore").decode().replace("\r\n", " ").replace("\n", " ")

        logger.info(
            "[%s/%s] Sending file contents of %s/%s to API server on %s...",
            _idx + 1,
            len(_files),
            input_dir,
            _file,
            args.H
        )
        r = requests.post("http://" + args.H + "/chunk", json={
            "body": _input_contents
        })

        try:
            logger.info("Writing to %s/%s...", output_dir, _out_file)
            with open(f"{output_dir}\{_out_file}", "w+") as output_file:
                output_file.write("\n\n".join([i.strip() for i in r.json()["output"]]))
        except:
            logger.error("Error, skipping file %s", _file)
            with open(output_dir + "\\failed_files.txt", "a+") as failed_files:
                failed_files.write(_file + "\n")
            continue
    logger.info("Done! Processed files into %s", output_dir)


def main():
    try:
        global host, port
        host, port = args.H.split(":")
    except ValueError:
        logger.error("Could not parse args. Host and/or port not valid")
        sys.exit(1)
    if not host and not isinstance(int(port), int):
        logger.error("Could not parse args. Host and/or port not valid")
        sys.exit(1)

    if os.path.isdir(args.i):
        logger.info("Input is a directory, assuming batch mode")
        if not os.path.exists(args.o):
            _c = input("Path: %s does not exist. Create directory? [Y,n]" % args.o)
            if _c not in ["n", "N", "No", "NO", "no"]:
                logger.info("Creating directory: %s...", args.o)
                os.mkdir(args.o)
            else:
                logger.warning("Exiting due to no usable output directory...")
                sys.exit(1)
        chunk_paragraphs_dir(args.i, args.o)
    elif os.path.isfile(args.i):
        logger.info("Input is a single file, assuming single mode")
        chunk_paragraphs_file(args.i, args.o)
    else:
        logger.error("Could not parse args. Please provide path to a single file or directory and check the directory "
                     "exists.")
        sys.exit(1)


if __name__ == "__main__":
    main()
