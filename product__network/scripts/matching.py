# coding: utf-8
# libraries:
import numpy as np
np.random.seed(0)
import pandas as pd
import glob
import re
import external
import model_matching
import os, sys
from argparse import ArgumentParser

params_input, params_output, params_parameters, params_process = external.initialize_parameters()
skills_dir = params_input['skill_directory']
skills_filename = params_input['skill_filename']
output_dir = params_output['output_directory']
output_dir_classification = params_output['output_directory_classification']
model_transformer = params_parameters['model_transformer']
debug = params_process['debug']

# functions:

# main:
if __name__ == '__main__':

	parser = ArgumentParser(description="Comparing CVs with the required skills")
	parser.add_argument(
		"-f", "--file",
		dest = "Filename",
		default=False,
		help="Input df (csv file) with classified phrases for comparison with requested skills."
	)

	args = parser.parse_args()
	if len(sys.argv)!=3:
		parser.print_help(sys.stderr)
		sys.exit(1)

	# check if input exists:
	filename_ = args.Filename
	isfile = os.path.isfile(filename_)
	if isfile:
		pass
	else:
		print ('ERROR: No input file: ' + filename_ + '; exit !' )
		sys.exit(1)
	if ('resulting_df_' in filename_) & (filename_.endswith('.csv')):
		pass
	else:
		print ('ERROR: the input file: ' + filename_ + ' is not properly saved !; exit !' )
		sys.exit(2)

	if debug:
		print ('DEBUG: script matching.py, input=', filename_)
	# read data for comparisons:
	cv_df, skills_df = model_matching.df_reader(filename_, skills_dir, skills_filename)
	df_ = model_matching.matching_procedure(cv_df, skills_df, model_transformer)
	if debug:
		print ('DEBUG: matching_procedure: df.shape=',df_.shape)
		
	result_df = model_matching.matched_features_selection(df_, skills_df)
	# save results as the dinal output:
	resulting_filename_ = os.path.basename(filename_)
	resulting_filename_ = os.path.join(output_dir, resulting_filename_)
	result_df.to_csv(resulting_filename_,index=False)
	# check if final resulting_filename_ exists:
	isfile = os.path.isfile(resulting_filename_)
	if isfile:
		print ('final output: df (' + resulting_filename_ + ') is ready !')
		# remove filename_:
		os.remove(filename_)
		pass
	else:
		print ('ERROR: No output file: ' + resulting_filename_ + '; check stderr, exit !' )
		print ('ERROR: the intermediate file (' + filename_ + ') remains')
		sys.exit(1)

exit(0)