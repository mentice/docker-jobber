#!/usr/bin/python

# Docker Jobber Runner - manages launching and checkpointing jobber containers

import sys, os, pickle, codecs, click
from jobber import Jobber

@click.command()
@click.argument('image_name')
def runner(image_name):
	# Jobber.run_image passes the config encoded in base64
	config = pickle.loads(codecs.decode(os.environ['JOBBER_CONFIG'].encode(), "base64"))
	jobber = Jobber(config)
	return_code = jobber.launch(image_name)
	exit(return_code)

if __name__ == "__main__":
	runner()

