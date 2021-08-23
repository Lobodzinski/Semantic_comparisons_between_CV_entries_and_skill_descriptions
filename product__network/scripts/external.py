# coding: utf-8
"""
Helping functions for data preprocessing step
"""
# libraries:
import sys
import re
import os
from configparser import ConfigParser

# pdf reader part:
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFTextExtractionNotAllowed
from io import StringIO
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams, LTTextBox, LTTextBoxHorizontal
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator

#
#import rootpath
#root_path = rootpath.detect()
#sys.path.append(root_path)


# functions:
def root_path():
    """
    Return highest relative path of the package for loading file from subdirectories
    Returns:
        file_dir_path (str) : highest relative package path
    """
    from pathlib import Path
    file_dir_path = (Path(__file__) / '..').resolve()
    return file_dir_path.as_posix()

#
def get_root_directory():
    return os.getcwd().split('scripts')[0]

# reader of config file:
def config(section='Input', filename='../config/config.ini'):
    root_dir_ = get_root_directory()
    # create a parser
    parser = ConfigParser()
    # read config file
    parser.read(filename)

    # get sectiion
    files = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            s_ = param[1].replace('__root_dir__', root_dir_)
            files[param[0]] = s_
    else:
        raise Exception('Section {0} not found in the {1} file'.format(section, filename))

    return files

# 
# initial functions:
# params_inputs, params_lake, params_cloud, params_process = initialize_parameters()
def initialize_parameters():
    # read config parameters (files, paths, etc)
    params_inputs = config(section='Input')
    params_lake = config(section='Output')
    params_cloud = config(section='Parameters')
    params_process = config(section='Debug')
    return params_inputs, params_lake, params_cloud, params_process


# based on:
# https://stackoverflow.com/questions/22898145/how-to-extract-text-and-text-coordinates-from-a-pdf-file/22898159#22898159
# 
class MyParser(object):
    def __init__(self, pdf_):
        
        # parameters:
        self.acronym_list = ['b.sc.', 'b. sc.', 'm.sc.', 'm. sc.', 'm.s.', 'm. s.', 'b.s', 'b. s.', 
             'm.a.', 'm. a.', 'asp.net', 'asp .net', '.net', 'c#', 'no.', 'nr.', 'am.', 
             'bbf.', 'abzgl.', 'a.d.', 'adr.', 'b.w.', 'bzgl.', 'med.', 'phil.', 'priv.',
             'bzw.', 'ca.', 'dir.', 'i.e.', 'e.g.', 'e.v.', 'e. v.', 'geb.', 'pl.', 'südd.', 
             'u.a.', 'u.ä.', 'u.a.m.', 'u.a.w.g.', 'usw.', 'u.s.w', 'z.b.', 'ärztl.', 'exam.', 
             'etc.', 'i.w.', 'staatl.','gepr.', 'tel.', 
             'i.a.', 'i.b.', 'i.h.', 'i.j.', 'inkl.',
            's.a.s. & co.', 's.a.s.&co.', 's.a.s.', 'co.', 'co.kg', 'co. kg', 'ag.', 'dr.', 'ca.', 'h2o.ai', 
            'c++', 'c/c++', 'ing.', 'dipl.', 'etc.', 'v&v', '...', '..', 
            'ui/ux', 'ux/ui', 'ui/ ux', 'ux/ ui', 'ux / ui', 'ui / ux', 
            'dipl.', 'vue.js', 'ci/cd', 'ci/ cd', 'ci / cd', 'ci /cd', 
            'a.i.', 'a. i.', 'prof.', 'vb.net', 'vb .net', 'ltd.', 'pvt.', 'st.', 'str.',
            'draw.io', 's.r.o.', 's. r. o.', 'inc.', 's.a', 's.a.', 
            'node.js', 'express.js', 'react.js' ] 
        
        self.sign_ = ']]'
        self.ratio_threshold_ = 0.26 # counts ratio between nr of white spaces and the length of string
        self.min_words = 0 # defines min number of words in sentence
        self.words_with_digits_ = 3 # defines min number of words with digits only in sentence
        self.ratio_digits_ = 3./3. # defines the ratio between nr of digits And the length of sentence
        self.min_nr_of_words = 1
        # Create a PDF resource manager object 
        # Create a buffer for the parsed text
        rsrcmgr = PDFResourceManager()
        # Spacing parameters for parsing
        laparams = LAParams()
        device = PDFPageAggregator(rsrcmgr, laparams=laparams)
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        
        fp = open(pdf_, 'rb')
        
        # check if the document is readable:
        parser = PDFParser(fp)
        document = PDFDocument(parser)
        # Try to parse the document
        if not document.is_extractable:
            raise PDFTextExtractionNotAllowed
            
        pages = PDFPage.get_pages(fp)
        
        self.loc_dict_ = {}
        for page in pages:
            interpreter.process_page(page)
            layout = device.get_result()
            for lobj in layout:
                if isinstance(lobj, LTTextBox):
                    x, y, self.text = lobj.bbox[0], lobj.bbox[3], lobj.get_text()
                    self.verify_text()
                    # replace months:
                    self.replace_months()
                    if len(self.text) > 0:
                        x_ = "{:.2f}".format(x)
                        if x_ in self.loc_dict_:
                            self.loc_dict_[x_].append(self.text)
                        else:
                            self.loc_dict_[x_] = [self.text]
        
        self.records = []
        self.verify_dict()

    def replace_acronyms(self, input_str_):
        start = 0
        stop = len(input_str_)

        for elem_ in self.acronym_list:
            res_ = input_str_.lower().find(elem_, start, stop)
        
            if res_ > -1:
                # prepare elem_
                elem__ = "".join(elem_.split()).replace('.', self.sign_)
                # update the input:
                input_str_ = input_str_[:res_] + elem__ + input_str_[res_ + len(elem_):]
            
        return input_str_        

                
    def verify_dict(self):
            for k_ in self.loc_dict_.keys():
                # count values:
                v_ = [x.strip() for x in self.loc_dict_[k_]]
                # recheck lengths of v_:
                res_ = []
                for item_ in v_:
                    # detect & modify acronyms: 
                    item_ = self.replace_acronyms(item_)
                    # detects dots and replace back self.sign_ by dots ('.'):
                    splitter_ = '. '
                    if '. ' in item_:
                        res_.extend([ x_.replace(self.sign_,'.') for x_ in item_.split(splitter_)])
                    else:
                        res_.append(item_.replace(self.sign_,'.'))
                v_ = list(res_)
            
                if len(v_) > 1:    
                    self.records.extend(v_)
                if (len(v_) == 1):
                    v_ = list(set(v_))
                    if len(v_[0].split()) > self.min_nr_of_words:
                        self.records.extend(v_)
        
    def verify_text(self):
        # remove emails, webadresses 
        self.text = ' '.join([item for item in self.text.split() if ('@' not in item) ])
        # remove web addresses from text:
        self.text = re.sub(r'https?:\/\/\S*', '', self.text, flags=re.MULTILINE)
        
        l_ = [x for x in self.text.split() if 
                                  (re.findall(r"(?<!\d)\d{4}\d+(?!\d)", x) or 
                                   re.findall(r'(\w+[.]\w+[.]\w+[.]\w+)', x) or
                                   re.findall(r'([www.]\w+[.]\w+[.]\w+)', x) or
                                   re.findall(r'(\w+[.]\w+[.]\w+)', x))
                                  ]

        if True:
            # replacements: 
            ratio = self.detect_anomalies(self.text)
            if (ratio<self.ratio_threshold_):
                _RE_COMBINE_WHITESPACE = re.compile(r"\s+")
                self.text = self.text.replace('• ',' ')
                self.text = self.text.replace('- ',' ')
                self.text = self.text.replace('# ',' ')
                self.removal_signs()
                self.text = _RE_COMBINE_WHITESPACE.sub(" ", self.text).strip()
                # remove duplicated:
            else:
                self.text = ''
                
        else:
            self.text = ''
            
    def split_text(self):
        splitter_ = '\. '
        # split text using '.' if the line too long
        vec_ = self.text.split(splitter_)
        return vec_
            
        
    def removal_signs(self):
        # replace signs by whitespace:
        custom_punctuation='!"$%&\'()*-/;=?@[\\]^_`{|}~’”…•●®·❖➢�½▪ㅡ→‰§>◦'
        table = str.maketrans(custom_punctuation,' '*47) # official set of punctuation 
        self.text = self.text.translate(table)
        # remove multiplicated whitespaces: 
        self.text = re.sub('\s+', ' ', self.text)
        removals = ['\uf02d','\u200b',' \uf0b7','\uf02a','\uf029','\uf0ac','\uf003','\uf0e1','\uf095','\uf029',
                   '\uf095','MOOC']
        for item_ in removals:
            self.text = self.text.replace(item_,'')
   
    def detect_anomalies(self, line):
        # 1. by counting ration of the nr of white spaces/length of line
        nr_ws = len([item_ for item_ in line if item_ == ' ']) 
        len_line = len(line)+1
        ratio = nr_ws/len_line
        
        return ratio
    
    def replace_months(self):
        months_ = {
            'january':1,'jan.':1,'jan':1,'januar':1,
            #
            'february':2,'feb.':2,'feb':2,'februar':2,
            #
            'march':3,'mar.':3,'mar':3,'märz':3,'maerz':3,
            #
            'april':4,'apr.':4,'apr':4,
            #
            'may':5,'mai':5,'May':5,
            #
            'june':6,'jun.':6,'jun':6,'juni':6,
            #
            'july':7,'jul.':7,'jul':7,'juli':7,
            #
            'august':8,'aug.':8,'aug':8,
            #
            'september':9,'sep.':9,'sep':9,
            #
            'october':10,'oct.':10,'oct':10,'oktober':10,
            #
            'november':11,'nov.':11,'nov':11,
            #
            'dezember':12,'dec.':12,'dec':12,'dez.':12,'dez':12
        }
    
        months_ = dict((re.escape(k), v) for k, v in months_.items()) 
        pattern = re.compile("|".join(months_.keys()))
        res_ = ''
        for x in self.text.split():
            if x.replace('.','').strip().lower() in list(months_.keys()):
                try:
                    r_ = pattern.sub(lambda m: str(months_[re.escape(m.group(0))]), str(x).lower())
                except:
                    r_ = ''
            else:
                r_ = x
            res_ = ' '.join([res_, str(r_)])
        
        self.text = res_

        