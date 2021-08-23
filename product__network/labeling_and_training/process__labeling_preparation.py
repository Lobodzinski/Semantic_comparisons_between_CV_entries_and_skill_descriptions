# coding: utf-8
# libraries:
import os, sys
import pandas as pd
import glob
import numpy as np
import time
import external, model_definitions
from argparse import ArgumentParser

import logging
import warnings
warnings.filterwarnings("ignore")

# Constants:
BADDIR = 'missing' 
HELPINFO = 'usage: <this_script> -d | --input_dir <my_input_dir_with_CDs_for_the_labeling_process>'

# functions:

def dir_path(string):
	if os.path.isdir(string):
		return string
	else:
		print ("Input directory with CVs for the labeling and training processes is missing ! exit().")
		#raise NotADirectoryError(string)
		return BADDIR

def read_pdf_file(f_):
	try:
		p = external.MyParser(f_)
	
		input_ = p.records
		df=pd.DataFrame()
		df['entry'] = input_
		df['tag'] = 0

		# clean up df
		# remove rows with empty sentence
		df.dropna(subset=['entry'], inplace=True)
		df.drop_duplicates(subset=['entry'],keep='first',inplace=True)
		# lower letters
		df['entry_orig'] = df['entry']
		df['entry'] = df['entry'].str.lower()

		return df
	except:
		# error - wrong file ?
		return pd.DataFrame()


# main:
if __name__ == '__main__':

		parser = ArgumentParser(description="Preparation of the data for labeling.")
		parser.add_argument("-d", "--input_dir", type=dir_path, help=HELPINFO)

		args = parser.parse_args()
		if (args.input_dir == BADDIR ) | (args.input_dir == None):
			print (HELPINFO)
			exit()

		# correct dir:
		# Check if the file contains files with the extension pdf:
		# 
		# choose a local directory to read the data.
		local_input_files=glob.glob(args.input_dir+'/*.pdf')
		if len(local_input_files) == 0:
			print ("Input directory " + args.input_dir + " does not contain files with extension pdf,! exit().")
			exit()

		# else: continue preparation of the final df file:
		# create directory of the final df as:
		dir_ = 'df_final'
		final_df_dir_ = os.path.join(os.path.dirname(args.input_dir),dir_)
		# create final_df_dir_
		try:
			os.makedirs(final_df_dir_)
		except OSError:
			pass
		filename_ = 'df_final.csv'
		f_ = os.path.join(final_df_dir_,filename_)
		print ('The final csv file containing the extracted phrases from the available CVs for the tagging process will be available as:\n',f_)

		df_final = pd.DataFrame()
		for item_ in local_input_files:
			df = read_pdf_file(item_)
			df_final = df_final.append(df)
		# save all:
		df_final.to_csv(f_, index=False)

#the end !!!!
exit()
