# coding: utf-8
# libraries:
import os, sys
import pandas as pd
import numpy as np

# deep learning libraries:
import transformers
#from transformers import BertModel, BertTokenizer
from transformers.modeling_bert import BertModel
from transformers import BertTokenizer

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# models:
# classified skills or tasks.
def get_new_predictions(model, data_loader, device='cpu'):
	model = model.eval()
	review_orig_texts = []
	review_texts = []
	predictions = []
	prediction_probs = []
	ids = []

	ind = 0
	with torch.no_grad():
		for d in data_loader:
			text_orig = d["text_orig"]
			texts = d["review_text"]
			JD_ids = d["job_id"]
			input_ids = d["input_ids"].to(device)
			attention_mask = d["attention_mask"].to(device)
			
			outputs = model(
				input_ids=input_ids,
				attention_mask=attention_mask
			)

			_, preds = torch.max(outputs, dim=1)
			review_orig_texts.extend(text_orig)
			review_texts.extend(texts)
			predictions.extend(preds)
			prediction_probs.extend(outputs)
			ids.extend(JD_ids)
			
			ind += 1
			if (ind % 1000) == 0:
				print ('progress =',ind)

	predictions = torch.stack(predictions).cpu()
	prediction_probs = torch.stack(prediction_probs).cpu()

	return ids, review_orig_texts, review_texts, predictions, prediction_probs

@torch.jit.script
def mish(input):
	return input * torch.tanh(F.softplus(input))
  
class Mish(nn.Module):
	def forward(self, input):
		return mish(input)
	
# 
class SentenceClassifier(nn.Module):
	def __init__(self, n_classes, PRE_TRAINED_MODEL_NAME):
		super(SentenceClassifier, self).__init__()
		self.n_classes = n_classes
		self.base_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
		self.base_model_output_size=768 # Constraint from Bert 
		self.dropout = 0.1              # fixed 
		self.out = nn.Linear(self.base_model.config.hidden_size, self.n_classes)

		self.classifier = nn.Sequential(
			nn.Dropout(self.dropout),
			nn.Linear(self.base_model_output_size, self.base_model_output_size),
			Mish(),
			nn.Dropout(self.dropout),
			nn.Linear(self.base_model.config.hidden_size, self.n_classes)
		)

		for layer in self.classifier:
			if isinstance(layer, nn.Linear):
				layer.weight.data.normal_(mean=0.0, std=0.02)
				if layer.bias is not None:
					layer.bias.data.zero_()

	def forward(self, input_ids, attention_mask): 
		X, attention_mask = input_ids, attention_mask

		#hidden_states, pooled_output = self.base_model(X, attention_mask=attention_mask)
		hidden_states = self.base_model(X, attention_mask=attention_mask)		
		output = self.classifier(hidden_states[0])
		# mean pooling over tokens
		output=torch.mean(output, dim=1)
		# to check: min, max pooling or additional RNN layers .
		# or just return a specific representation of tokens (up to max_len):
		# return self.classifier(hidden_states[:, 0, :])
	
		return output

# prepare data for classification:
class ReviewDataset():

	def __init__(self, text_orig, reviews, ids, targets, tokenizer, max_len):
		self.text_orig = text_orig
		self.reviews = reviews
		self.ids = ids
		self.targets = targets
		self.tokenizer = tokenizer
		self.max_len = max_len
  
	def __len__(self):
		return len(self.reviews)
  
	def __getitem__(self, item):
		text_orig = str(self.text_orig[item])
		review = str(self.reviews[item])
		target = self.targets[item]
		id_ = self.ids[item]

		encoding = self.tokenizer.encode_plus(
			review,
			add_special_tokens=True,
			max_length=self.max_len,
			return_token_type_ids=False,
			pad_to_max_length=True,
			return_attention_mask=True,
			return_tensors='pt',
			truncation=True
			)

		return {
			'text_orig' : text_orig, 
			'review_text': review,
			'job_id': id_,
			'input_ids': encoding['input_ids'].flatten(),
			'attention_mask': encoding['attention_mask'].flatten(),
			'targets': torch.tensor(target, dtype=torch.long)
		}

def create_data_loader(df, tokenizer, max_len, batch_size):
	# be sure that job_id is not a problem:
	if 'job_id' not in df.columns:
		df['job_id'] = 0

	ds = ReviewDataset(
		text_orig=df.text_orig.to_numpy(),
		reviews=df.text.to_numpy(),
		ids=df.job_id.to_numpy(),
		targets=df.labels.to_numpy(),
		tokenizer=tokenizer,
		max_len=max_len
		)

	return DataLoader(
		ds,
		batch_size=batch_size,
		num_workers=4
		)

# dowload models and initialize classification task:
def initialize_classifier(model_dir, model_name, PRE_TRAINED_MODEL_NAME, n_classes, device='cpu'):
	#try:
	if True:
		device = 'cpu' # fixed for the time being
		tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
		filename_model_ = os.path.join(model_dir ,model_name)
		model = SentenceClassifier(n_classes, PRE_TRAINED_MODEL_NAME)
		model.load_state_dict(torch.load(filename_model_, map_location=device))
		model = model.to(device)

		return model, tokenizer
	else:
		#except:
		print ('Something is wrong, we have to stop the server.')
		exit()

# classification task:
def tuned_classifier(df, model, tokenizer, cv_name, output_dir, MAX_LEN, BATCH_SIZE, debug):
	
	extract_data_loader = create_data_loader(df, tokenizer, MAX_LEN, BATCH_SIZE)
	job_ids, y_review_orig_texts, y_review_texts, y_pred, y_pred_probs = get_new_predictions(model, extract_data_loader)

	# build resulting df:
	print ('Preparing final output ... ')
	resulting_df = pd.DataFrame()
	resulting_df['job_id'] = job_ids
	resulting_df['text_orig'] = y_review_orig_texts
	resulting_df['text'] = y_review_texts
	resulting_df['label'] = y_pred
	resulting_df['proba'] = y_pred_probs
	if debug:
		print ('DEBUG: resulting_df.shape=',resulting_df.shape)
	# save results:
	resulting_filename = 'resulting_df__'+cv_name+'.csv'
	f_ = os.path.join(output_dir, resulting_filename)
	resulting_df.to_csv(f_, index=False)
	if debug:
		print ('DEBUG: resulting_df saved to:', f_)

	return resulting_df, resulting_filename
