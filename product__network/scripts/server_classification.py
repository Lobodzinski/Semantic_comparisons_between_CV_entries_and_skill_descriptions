# coding: utf-8
# libraries:
import os, sys
import pandas as pd
import glob
import numpy as np
import time
import external, model_definitions
import subprocess

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

import logging
import warnings
warnings.filterwarnings("ignore")


params_input, params_output, params_parameters, params_process = external.initialize_parameters()
DIRECTORY_TO_WATCH = params_input['input_directory']
output_dir = params_output['output_directory_classification']
BATCH_SIZE = int(params_parameters['batch_size'])
MAX_LEN = int(params_parameters['max_len'])
PRE_TRAINED_MODEL_NAME = str(params_parameters['pretrained_model'])
model_dir = params_parameters['model_directory']
model_name = params_parameters['model_name']
n_classes = int(params_parameters['nr_of_classes'])
debug = params_process['debug']

# functions:
def read_pdf_file(f_):
	try:
		p = external.MyParser(f_)
	
		input_ = p.records
		df=pd.DataFrame()
		df['text'] = input_
		df['labels'] = 0

		# clean up df
		# remove rows with empty sentence
		df.dropna(subset=['text'], inplace=True)
		df.drop_duplicates(subset=['text'],keep='first',inplace=True)
		# lower letters
		df['text_orig'] = df['text']
		df['text'] = df['text'].str.lower()

		return df
	except:
		# error - wrong file ?
		return pd.DataFrame()


# Class used for automatic detection of new pdf files in the Input Directory.
# 
class Handler(FileSystemEventHandler):

       	@staticmethod
        def on_any_event(event):
                ''' 
                event.event_type:
                        'modified' | 'created' | 'moved' | 'deleted'
                event.is_directory
                        True | False
                event.src_path
                        path/to/detected/file
                '''
                if event.is_directory:
                        return None

                elif event.event_type == 'created':
                        # Take any action here when a file is first created
                        time.sleep(1) 
                        if debug:
                        	print ("DEBUG: Received created file: %s;" % event.src_path)
                        # check if copying of the file is finished:
                        size_ = -1
                        while size_ < 0.:
                            size_ = os.path.getsize(event.src_path)
                            if debug:
                            	print ('DEBUG: file (' + event.src_path + ') is copied: size=',size_)

                        filename_ = event.src_path
                        cv_name = os.path.basename(filename_).split('.pdf')[0]
                        start = time.time()
                        # read a given pdf file:
                        pdfs_ = glob.glob(filename_)
                        df = read_pdf_file(filename_)
                        if df.shape[0] == 0:
                            # error
                            print ('ERROR: file (Wrong pdf file format - file is removed.')
                            os.remove(filename_)
                        # classification:
                        if debug:
                            print ('DEBUG: Classification stage: started ...')
                        resulting_df, resulting_filename = model_definitions.tuned_classifier(df, model, tokenizer, cv_name, output_dir, MAX_LEN, BATCH_SIZE, debug)
                        if debug:
                            print ('DEBUG: Classification stage: done')
                            
                        resulting_filename_ = os.path.join(output_dir,resulting_filename)

                        # start comparison with skills:
                        if debug: 
                            print ('DEBUG: start matching the file ' + resulting_filename_ + ' ...')
                        
                        python_path =  os.path.join(external.get_root_directory(),'envs/py37_env2/bin/python')
                        cmd_ = python_path + ' ./matching.py -f ' + resulting_filename_ 
                        p = subprocess.Popen(cmd_, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=True)
                        p.wait()
                        std_ = p.stdout.read()
                        err_ = p.stderr.read()

                        if debug: 
                            print ('DEBUG: stdout=',std_)
                            print ('DEBUG: stderr=',err_)
                            print ('DEBUG: done - for the time being !')


# This is the watchdog function:
class Watcher:

	def __init__(self):
		self.observer = Observer()
		self.debug = debug
		self.DIRECTORY_TO_WATCH = DIRECTORY_TO_WATCH

	def run(self):
		event_handler = Handler()
		self.observer.schedule(event_handler, self.DIRECTORY_TO_WATCH, recursive=True)
		self.observer.start()

		try:
			while True:
				time.sleep(1)
		except:
			self.observer.stop()
			print ("Error")

		self.observer.join()


# main function:
#root_dir = os.getcwd()
if __name__ == '__main__':
	print ('Initialization of the model ...')
	model, tokenizer = model_definitions.initialize_classifier(model_dir, model_name, PRE_TRAINED_MODEL_NAME, n_classes, device='cpu')
	print ('Initialization is succesfull !')
	w = Watcher()
	w.run()

exit()
