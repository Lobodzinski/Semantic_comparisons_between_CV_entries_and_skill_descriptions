## Main Idea of the Proposed Process:

relies on an prior, automated semantic comparison of an input resume (CV) in pdf file format or a set of CVs (a set of pdf files) with all the skill categories catalogued in the Connex database. 
Using the semantic similarities thus defined, it is possible to automatically select those resumes that most closely match the given skills profile. 

## What is not done:
despite efforts, there is no algorithm that can associate dates of experience with the description of that experience in the input pdf document of the CV. The main problem here is that the authors of the CVs are very inventive in creating multiple columns and separate areas in the CV, using different schemes for labeling dates and descriptions of experience. To the author's knowledge, the problem has not yet been solved in a global sense.   

## What requires manual work:
the model used to classify the phrases/sentences read from the CV input file in pdf format, should be updated regularly. 
This requires, unfortunately, the manual creation of training files.
The detailed procedure is described in the section *'Model training'*. 
Ideally, the creation of new training files should be done by someone experienced in reading candidate resumes.

## Part 0:
General overview how the semantc comparison machine is working:

In general, the semantic comparison workflow between the text in the candidate's Resume (CV) and the skill description that is part of the database consists of 2 parts:  
1. the supervised part, which consists of 
         - extracting phrases from the CV file in pdf format,
         - training a model which classifies phrases into one of the groups: <br/>
         *personal (category/label/tag 0), languages (1), education (2), experience (3), skills (4) and summary (5)*.
2. unsupervised part: which consists in calculating similarity between phrases and skill descriptions.

The model works for German and English.
The above description is very abbreviated, and the whole algorithm is more complicated. Mainly because we have to deal with very different texts: edited in a very non-standard way, with not properly used punctuation marks, with non-standard abbreviations and jargon vocabulary. 

## Part 1:
Location of files and directories:

I have prepared 2 mods of machine operation:
1. fully local, which, after creating environments, can be used in fully local mode (without network access): *'product__fully_local'* .
Its docker image is 14.7GB
2. Using network allocated models (*'product__network'*). Its docker image size is 7.88GB.

You can select a module by creating a link <br/>
```$ ln -s <selected_mode> product```

### Proper structure of files and directories in the main directory *product*.<br/>
This structure must be reproduced in order for all components of the algorithm to work correctly.<br/>
files:
```
├── Dockerfile
├── output_classification
├── product_rel_01.tgz
├── requirements__py37_env1.txt
├── requirements__py37_env2.txt
```
Directories with listed files:
```
├── input
│   ├── .
│   ├── ..
├── output
│   ├── .
│   ├── ..
├── envs
│   ├── py37_env1
│   └── py37_env2
├── config
│   ├── config.ini
│   └── config.ini__net_access
├── labeling_and_training
│   ├── external.py -> ../scripts/external.py
│   ├── __init__.py
│   ├── model_classification.py -> ../scripts/model_classification.py
│   ├── model_definitions.py -> ../scripts/model_definitions.py
│   ├── models
│   ├── p2_sentence_classification__training_model_BERT_classifier__official.ipynb
│   ├── process__labeling_preparation.py
│   └── __pycache__
├── models
│   ├── bert-base-multilingual-cased-finetuned-conll03-dutch
│   ├── best_model_state.bin
│   ├── best_train_model_state.bin
│   ├── history.npy
│   └── msmarco-distilbert-multilingual-en-de-v2-tmp-lng-aligned
├── scripts
│   ├── external.py
│   ├── __init__.py
│   ├── matching.py
│   ├── model_classification.py
│   ├── model_definitions.py
│   ├── model_matching.py
│   ├── __pycache__
│   └── server_classification.py
└── skills
    ├── query_result_2021-05-12T12 45 13.528106Z.csv
    └── skills.csv -> query_result_2021-05-12T12 45 13.528106Z.csv
```

## Part 2:
Config file:
```
% cat config/config.ini
[Input]
input_directory: __root_dir__/input
skill_directory: __root_dir__/skills
skill_filename: skills.csv

[Output]
output_directory_classification: __root_dir__/output_classification
output_directory: __root_dir__/output

[Parameters]
model_directory: __root_dir__/models
model_name: best_model_state.bin
model_transformer: __root_dir__/models/msmarco-distilbert-multilingual-en-de-v2-tmp-lng-aligned
batch_size: 10
max_len: 120
pretrained_model: __root_dir__/models/bert-base-multilingual-cased-finetuned-conll03-dutch
nr_of_classes: 6
threshold: 0.2

[Debug]
debug: True
```

The file *'config.ini'* contains a couple of sections. <br/>
Below is a list with their description:
1. *'Input'*: contains a list of files necessary for the comparison. 
*'skill_filename'* is the name of the file in csv format to be loaded from the *'Skill'* table.
2. *'output'*: defines the internal files used to classify the phrases extracted from the CV files.
3. *'Parameters'*: defines the pretrained models and the parameters of the models.
4. *'Debug'*: defines whether the algorithm should work in debug mode (possible values: *'False | True'*).

Locations of files and directories are preceded by the *'__root_dir__'* variable.
This variable is set automatically by the framework, **it must remain unchanged**.    

### Pretrained models:

The whole framework is based on 2 pretrained models:
1. for phrase classification: 
*'https://huggingface.co/dbmdz/bert-base-multilingual-cased-finetuned-conll03-dutch'*
with description:
*'https://metatext.io/models/dbmdz-bert-base-multilingual-cased-finetuned-conll03-dutch*' .
3. used for asymmetric semantic comparison: 
*'https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/v0.2/msmarco-distilbert-multilingual-en-de-v2-tmp-lng-aligned.zip'*
with interesting discussion and description:
*'https://github.com/Lobodzinski/tests/blob/main/README.md'* 




## Part 3:
Data Labeling and model training:

preparation of the dataframe for tagging:
1. start :
- go to the 'product' directory <br/>
```$ cd product``` ,
- create and activate env *py37_env1* using packages listed in the file: *requirements__py37_env1.txt*,
- prepare directory with CV's collection in pdf format ready for model training (e.g. */tmp/my_CV_collection/*),
- start the creation of the training dataframe:
```
$ cd labeling_and_training
$ python process__labeling_preparation.py -d | --input_dir /tmp/my_CV_collection/
```
The final csv file containing the extracted phrases from the available CVs for the tagging process will be available as:<br/>
*/tmp/my_CV_collection/df_final/df_final.csv*

2. Output:
the final data frame prepared for the manual tagging operation is saved as:<br/>
*/tmp/my_CV_collection/df_final/df_final.csv*

3. Tagging operation:
The data frame contains 3 columns named: <br/>
*'entry','tag','entry_orig'*<br/>
where the *'entry'* and *'entry_orig'* columns contain the extracted phrases that should be classified . The phrases in the *'entry'* are written with small letters and without special signs. Original formulation is saved in the *'entry_orig'* column . This is due to the requirements of the model related to the list of special characters.  <br/>
The phrases should be classified according to the scheme:

class of the phrase: | tag/label value: 
------------ | -------------
personal     |             0
languages     |            1
education      |           2
experience       |         3
skills             |       4
summary              |     5

The result of tagging/classifying phrases should be stored in the *'tag'* column as natural numbers.
After completing this process and saving the data, we can proceed to the model training step.

4. Model training: 
**Training a model is very complicated and this is the task for an expert ! **.<br/>
It requires knowledge of many details related to the architecture and operation of neural networks, especially the Transformer model (https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)).

The model training can also be a test of a modified neural network architecture. 
Therefore, training should preferably be done using the jupyter-notebook framework. 
The Jupyter notebook 
*p2_sentence_classification__training_model_BERT_classifier__official.ipynb *
script contains the necessary notes to learn the general principle of the model.

Please note that this model is highly customized and is a composite of a public, pretrained model <br/>
(*PRE_TRAINED_MODEL_NAME = 'dbmdz/bert-base-multilingual-cased-finetuned-conll03-dutch'*) <br/>
with customized additional layers. 
This combination allows us to achieve good results in phrase classification specific to our problem using a small sample of tagged input data (phrases).

Hints for better models:<br/>
- check regularly the public transformer base of pretrained models (https://huggingface.co/dbmdz) and entire discussion about transformers. 
Perhaps there will be better pretrained models than those currently available (see section **part 2**, **Pretrained models**) <br/>
- Investigate how the new public pretrained models affect the final training result. 
If the results are better (confussion matrix, F1-score) then use the new model in the production process. 

5. Current model:
The classification report of the currently operating model is as follows:
```
              precision    recall  f1-score   support

    personal       0.86      0.77      0.81        39
   languages       0.75      1.00      0.86         3
   education       0.80      0.86      0.83        14
  experience       0.89      0.94      0.92        86
      skills       0.82      0.70      0.76        20
     summary       0.62      0.62      0.62         8

    accuracy                           0.85       170
   macro avg       0.79      0.82      0.80       170
weighted avg       0.85      0.85      0.85       170
```
The confusion matrix is shown on the Figure *'Classification_report__confusion_matrix.png'*.

6. Model update:

If the newly created model has better values (precision, recall, and especially **f1-score**) then you should replace the existing model with the new one.

7. Model replacement:

copy the new model to the directory defined in your *product/config/config.ini* and update the name of the model in th file.
So, in the section *'Parameters'*: 
the model location is defined by value of the parameter *'model_directory'* and 
the model name as value of the *'model_name'* parameter.

The part 2 ('Data Labeling and model training') is done.

## Part 4:
Preparing Docker images and getting started:

1. Prepare docker env: 

   1.1. install docker (not a part of this description),
  
   1.2. prepare the docker group (example for ubuntu),<br/>
   ```
   $ sudo groupadd docker
   $ sudo usermod -aG docker $USER
   $ newgrp docker
   ```
       
2. Creation of the Docker image

   2.1. copy tar gz package to the final directory (*<cwd>*): <br/>
   ```
   $ cd <cwd><br/>
   $ cp <location_path>/product_rel_01.tgz .<br/>
   ```
   2.2. upack the tar gz package & remove it aftewords: <br/>
   ```
   $ tar zxvf product_rel_01.tgz
   $ rm product_rel_01.tgz
   ```
   2.3. build the docker file: in *<cwd>* directory start the command (Don't forget the dot at the end of the command !): <br/>
   ```
   $ docker build -t <your_docker_container_name> .
   ```
   due to the large size of the torch packages  (> 800MB) be sure that the connection to the network is stable and fast !

   2.4. save the container ( *<your_docker_container_name>* ) to the file <br/>
   ```
   $ docker save <your_docker_container_name> -o <your_docker_container_name>.tar
   ```
   be careful: the final size of the <your_docker_container_name>.tar is ~9G or even 14GB !

   2.5. The file ( *<your_docker_container_name>.tar* ) could be transferred to any other hostwith installed Docker Engine, loaded and started as a standalone classification process.

3. How to user the docker file:
         
   3.1 load the docker image to the memory: <br/>
   ```
   $ docker load --input <your_docker_container_name>.tar
   ```
   and check if the container is properly loaded: <br/>
   ```
   $ docker images
   ```
   the output should list the uploaded container with name *<your_docker_container_name>* .
   If the operation needs to be repeated, remove the image from memory: <br/>
   ```
   $ docker image rm -f <your_docker_container_name>
   ```

   3.2 in the *<cwd>* directory create additional directories: *Input_Entry*, *Output_Entry* which will be use as the input directory ( *Input_Entry* ) for cv to be classified and for final df saved (in the csv format) in the output directory  ( *Output_Entry* ). <br/>
   ```
   $ mkdir -p Input_Entry
   $ mkdir -p Output_Entry
   ```

   3.3 start the uploaded container with commands:<br/>
   for Windows:<br/>
   ```      
   > $myPath = (Resolve-Path .).Path
   > docker run --network=host -a stdout -i \
     --mount type=bind,source=$myPath/Input_Entry,target=/root/src/input \
     --mount type=bind,source=$myPath/Output_Entry,target=/root/src/output \
     -t <your_docker_container_name>
   ```

   for linux:<br/>
   ```
   $ myPath=`pwd`
   $ docker run --network=host -a stdout -i \
     --mount type=bind,source=$myPath/Input_Entry,target=/root/src/input \
     --mount type=bind,source=$myPath/Output_Entry,target=/root/src/output \
     -t <your_docker_container_name>
   ```

   this command will start the standalone daemon which will process each new CV entering the input directory *Input_Entry* .<br/>
   First, the models are initialized and the program needs a minute before all the initiations steps have been completed.
   The program is ready for semantic comparisons of the CV entries with the requested skills if the text "Initialization is succesfull !" appears.

   3.4 An example of an output of the program for the CV processing. The input file: *EdibXIsic.pdf*:<br/>
   ```
   DEBUG: Received created file: /root/src/input/EdibXIsic.pdf;
   DEBUG: file (/root/src/input/EdibXIsic.pdf) is copied: size= 176269
   Classification stage: started ...
   Preparing final output ... 
   DEBUG: resulting_df.shape= (148, 5)
   DEBUG: resulting_df saved to: /root/src/output_classification/resulting_df__EdibXIsic.csv
   DEBUG: Classification stage: done
   DEBUG: start matching the file /root/src/output_classification/resulting_df__EdibXIsic.csv ...
   DEBUG: stdout= b'DEBUG: script matching.py, input= /root/src/output_classification/resulting_df__EdibXIsic.csv
   DEBUG: matching_procedure: df.shape=(52658, 5) final output: df (/root/src/output/resulting_df__EdibXIsic.csv) is ready !
   DEBUG: stderr= b''
   DEBUG: done - for the time being !
   ```
   Most control texts are preceded by the word 'DEBUG'. <br/>
   You can disable it by changing the options from <br/>
   *debug: True*<br/>
   to<br/>
   *debug: False*<br/>
   in *product/config/config.ini*, in the *[Debug]* category.

4. Format of the Output Data Frame:<br/>
The output data is created in a data frame with columns: 
*'Category'*,*'Query'*,*'Matches'*,*'Matches_scores'*<br/>
where:<br/>
* 'Category'*: is the group corresponding to the skill categories from the Connex database,
* 'Query'*: are the skills present in the given group from 'Category' class,
* 'Matches'*:  is the list of phrases found in the input CV that semantically correspond to the given value of 'Query',
* 'Matches_scores'*: the list of numerical values of similarity between elements of *'Matches'* and *'Query'* (corresponds to the cosine distance metric).

The example :<br/>
*'Category'*,*'Query'*,*'Matches'*,*'Matches_scores'*<br/>
Experience,Bauwesen,<br/>
"['Architecture and hands on Engineering.',<br/>
 'Bombardier is the company that shapes the future of mobility, it bridges distances and bring people together',<br/>
 'Operation of build and deployment systems.',<br/>
 'VIVAAN Tech Aws Devops Engineer','and production servers',<br/>
 'Experience in setting up the pipeline and multibranch pipeline process in Jenkins and helping the',<br/>
 'project artifacts are deployed automatically to various environments using Jenkins.']",<br/>
 "[0.60041583,0.46913487,0.41668913,0.36673641,0.35518056,0.33034697,0.32709813]"<br/>
....<br/>

## Part 5:
Final Setup or Placing the semantic comparison inside the production pipeline:

In this section (not implemented) I would like to add some comments about the use of semantic comparison in the candidate classification process. <br/>
- **How to get best CVs:**<br/>
since the comparison of phrases from the input CV with queries (*'Name'* values) is done globally,
it cannot take into account priorities determined locally on the basis of a given Job Ad (Job Ad under consideration).
For a given Job Ad, and specified skill composition(s) (skill + tier pairs) along with priorities (created by a user), we search for those CVs (in the output of global semantic comparisons) that show similarity to the requested skills (i.e. non-zero number of phrases in the *'Matches'* or *'Matches_scores'* columns) and compute the selected metric(s) calculated based on the *'priority'* criterion (metrics - see next point 3.). For such selected set of CVs we choose the best resume(s) (with the best values of the metric) for the next stage of recruiting process.<br/>
This part of the implementation must be integrated with the user action (creation of the skill compositions). <br/>

- **Proposal of metrics:**<br/>
here I would like to suggest some types of metrics that can be used to select the best CVs.<br/>
By metric, I mean a single number indicating the value of a given CV for a given list of skill compositions.
The higher the value, the better the CV.<br/>
In fact, the issue is complicated because we do not have a global scale of comparison between different CVs. We compare the writing skills of the CV writers.<br/>
In proposed metrics I use the following notation:<br/>
  * i - index numbering the Categories (set of unique values in *'Category'* column of the *'Skill'* table),<br/>
  * j - index numbering elements in a given Category i,<br/>
  * Matches_scores_{i,j} - j-th matched score between extracted j-th phrase and query from a given category i,<br/>
  * priority_{i,j} - j-th priority of the skill from categpry i-th and specified for a given Job Ad.  <br/>
  * average_{over j}{} - denotes average value over all elements belonging to the given category i.<br/>

  *Proposition 1:*<br/>
  metric1 = \sum_{i \in {Categories}} Matches_scores_{i,j} /(priority_{i,j} + 1)

  *Proposition 2:*<br/>
  metric2 = \sum_{i \in {Categories}} max{ Matches_scores_{i,j} /(priority_{i,j} + 1) }

  *Proposition 3:*<br/> 
  metric3 = \sum_{i \in {Categories}} average_{over j}{ Matches_scores_{i,j} /(priority_{i,j} + 1) }

  *Proposition 4:*<br/>
  metric4 = \sum_{i \in {Categories}} [ \sum_{j} Matches_scores_{i,j}*(priority_{i,j} + 1) / ( \sum_{j} (priority_{i,j} + 1)) ]

  The list above does not complete the full set of possible formulas.
  It is not know which metric is the best. Therefore, one should compute all metrics and make a test phase to see how many of them make sense.
  This part of the implementation of the semantic comparison part can be implemented in any language.
