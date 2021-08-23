# coding: utf-8
#
import pandas as pd
from tqdm import tqdm, trange
import os, re
import gc
from kneed import KneeLocator
from sentence_transformers import SentenceTransformer, util

#
# read iput cv:
def df_reader(cv_classification_df, skills_dir, skill_filename):
	# download classified cv entries:
	cv_df = pd.read_csv(cv_classification_df)
	# download skills:
	skill_filename_ = os.path.join(skills_dir, skill_filename)
	skills_df = pd.read_csv(skill_filename_)
	return cv_df, skills_df

# clean-up strings:
def removal_signs(sentence):
	# replace signs by whitespace:
	custom_punctuation='!"$%&\'()*-/;=?@[\\]^_`{|}~’”…•●®·❖➢�½▪ㅡ→‰§>◦'
	table = str.maketrans(custom_punctuation,' '*47) 
	sentence = sentence.translate(table)
	# remove multiplicated whitespaces: 
	sentence = re.sub('\s+', ' ', sentence)
	return sentence

#
def create_embeddings(bi_encoder, skills_df, key, top_k, input_list):
	
	top_k = len(input_list) + 1
	corpus_embeddings = bi_encoder.encode(input_list, convert_to_tensor=True, show_progress_bar=False)
    
	g_ = skills_df.groupby(by="Category", dropna=False)
	queries = list(g_.get_group(key)['Name'].values)
	q_corrected = []
	for item_ in queries:
		item_ = item_.replace('/in',' ')
		item_ = item_.replace('/ in',' ')
		q_corrected.append(removal_signs(item_))
            
	queries_embeddings = bi_encoder.encode(q_corrected, convert_to_tensor=True, show_progress_bar=False)

	return corpus_embeddings, queries_embeddings, q_corrected


# given cv_df & skills_df:
def matching_procedure(cv_df, skills_df, model_name):

	bi_encoder = SentenceTransformer(model_name, device='cpu')
	kpi_ = util.cos_sim 
	top_k = 100
	result_df = pd.DataFrame()
	category_list = []
	query_list = []
	sent_list = []
	score_1_list = []
	len_list = []

	skills_categories = list(skills_df.Category.unique())
	for key in skills_categories:
		#print ('key=',key)
		# cv:
		if key == 'Language':
			all_ = list(cv_df[cv_df.label == 1].text_orig.values)
		if key == 'Education':
			all_ = list(cv_df[cv_df.label == 2].text_orig.values)
		if (key == 'Experience') | (key == 'Additional_Qualification') |\
			(key == 'Special_Knowledge') | (key == 'Occupational_Healthcare'):
			all_ = list(cv_df[(cv_df.label == 3) | (cv_df.label == 4)].text_orig.values)
        
		#print ('all=',all_)
        

		if len(all_) > 0:
			corpus_embeddings, queries_embeddings, q_corrected = create_embeddings(bi_encoder, skills_df, key, top_k, all_)
			for ident_, item_ in enumerate(q_corrected):
				hits = util.semantic_search(queries_embeddings[ident_], corpus_embeddings, 
                                    top_k=top_k, score_function=kpi_)
        
				iter_ = 1
				for h_ in hits[0]:
					idx_ = h_['corpus_id']
					score_ = h_['score']
					l_ = 1.*len(all_[idx_].split())

					query_list.append(item_)
					sent_list.append(all_[idx_])
					score_1_list.append(score_)
					len_list.append(l_)
					category_list.append(key)
					iter_ = iter_+1
        
		else:
			pass
    
	result_df['category'] = category_list
	result_df['query'] = query_list
	result_df['sentence'] = sent_list
	result_df['score'] = score_1_list
	result_df['sentence_length'] = len_list

	return result_df

def matched_features_selection(result_df, skills_df):

	min_val_ = 0.56  # 
	category_list = []
	query_list = []
	matched_phrases_list = []
	matched_phrases_scores_list = []

	skills_categories = list(skills_df.Category.unique())
	for category in skills_categories:
		min_val_ = 0.56
		if (category == 'Language') | (category == 'Education'):
			min_val_ = 0.5
		for query in result_df[result_df['category']==category]['query'].unique():
			min_ = result_df[( result_df['category']==category) & (result_df['query'] == query)]['score'].min()
			max_ = result_df[( result_df['category']==category) & (result_df['query'] == query)]['score'].max()

			if max_ > min_val_:
				sentences_ = result_df[( result_df['category']==category) & (result_df['query'] == query)]['sentence'].values
    
				y1 = result_df[( result_df['category']==category) & (result_df['query'] == query)]['score'].values
				# determne the elbow value:
				x0 = list(range(len(y1)))
				try:
					kn = KneeLocator(x0, y1, S=1., curve='convex', direction='decreasing') 
					elbow_ = kn.knee + 1
				except:
					elbow_ = 0
				#print ('TEST: matched_features_selection: query, elbow_=',query, '=>',elbow_)

				category_list.append(category)
				query_list.append(query)
				matched_phrases_list.append(sentences_[:(elbow_+1)])
				matched_phrases_scores_list.append(y1[:(elbow_+1)])
				
			else:
				pass
        
	Selected_features = pd.DataFrame()
	Selected_features['Category'] = category_list
	Selected_features['Query'] = query_list
	Selected_features['Matches'] = matched_phrases_list
	Selected_features['Matches_scores'] = matched_phrases_scores_list

	return Selected_features



#================================