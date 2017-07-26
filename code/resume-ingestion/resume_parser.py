from __future__ import unicode_literals
from itertools import chain
import nltk
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import sklearn
from glob import glob
import os
import pycrfsuite
import re
import pandas as pd
import xml.etree.ElementTree as ET
from docx import Document
import mammoth
from tqdm import tqdm
import unicodedata
from bs4 import BeautifulSoup
import collections
from glob import glob
import spacy
from pymongo import MongoClient
import html2text
import sys
import HTMLParser
reload(sys)
sys.setdefaultencoding('utf-8')


# CRF FUNCS
def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    features = [
        'bias',
        'word.lower=' + word.lower(),
        'word[-3:]=' + word[-3:],
        'word[-2:]=' + word[-2:],
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
        'postag=' + postag,
        'postag[:2]=' + postag[:2],
    ]
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(),
            '-1:postag=' + postag1,
            '-1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('BOS')

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper(),
            '+1:postag=' + postag1,
            '+1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('EOS')

    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]


# READ DOCX
def get_header_text(filepath):
    document = Document(filepath)
    namespace = dict(w="http://schemas.openxmlformats.org/wordprocessingml/2006/main")
    header_uri = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/header"
    headers = []
    for rel, val in document._part._rels.iteritems():
        if val._reltype == header_uri:
            xml = val._target._blob
            root = ET.fromstring(xml)
            text_elements = root.findall(".//w:t", namespace)
            for text_element in text_elements:
                if text_element.text is not None:
                    headers.append(text_element.text)
    return ''.join(headers)

def normalize_encoding(txt):
    return unicodedata.normalize('NFKD', txt)

def upper_to_title(txt):
    lines = []
    for line in txt.split('\n'):
        nline = ' '.join([x.title() if x.isupper() else x for x in line.split(' ')])
        lines.append(nline)
    return '\n'.join(lines)

def read_docx(filepath):
    header_text = get_header_text(filepath)
    with open(filepath, 'rb') as f:
        raw_text = mammoth.extract_raw_text(f)
        docx_html = mammoth.convert_to_html(f).value
    docx_text = header_text + '\n' + raw_text.value
    docx_text = normalize_encoding(docx_text)
    docx_text = '\n'.join([upper_to_title(line) for line in docx_text.split('\n')])
    docx_html = normalize_encoding(docx_html)
    return docx_text, docx_html

def read_doc(filepath):
    cmd = 'soffice --headless --invisible --convert-to htm:HTML {} --outdir tmp/'.format(filepath)
    os.system(cmd)
    newfilename = os.path.basename(filepath).replace('.doc', '.htm')
    newfilepath = 'tmp/'+newfilename
    if not os.path.exists(newfilepath):
        raise Exception('ERROR: doc to html conversion failed')
    with open(newfilepath, 'rb') as f:
        html = f.read()
    text = html2text.html2text(html)
    text = '\n'.join([upper_to_title(line) for line in text.split('\n')])
    text = text.replace('*', '').replace('_', '')
    os.remove(newfilepath)
    return text, html

def read_rtf(filepath):
    cmd = 'soffice --headless --invisible --convert-to htm:HTML {} --outdir tmp/'.format(filepath)
    os.system(cmd)
    newfilename = os.path.basename(filepath).replace('.rtf', '.htm')
    newfilepath = 'tmp/'+newfilename
    if not os.path.exists(newfilepath):
        raise Exception('ERROR: rtf to html conversion failed')
    with open(newfilepath, 'rb') as f:
        html = f.read()
    text = html2text.html2text(html)
    text = '\n'.join([upper_to_title(line) for line in text.split('\n')])
    text = text.replace('*', '').replace('_', '')
    os.remove(newfilepath)
    return text, html

def read_pdf(filepath):
    filename = os.path.basename(filepath)
    newfilepath = 'tmp/' + os.path.basename(filename).replace('.pdf', '.html')
    cmd = 'pdf2txt.py -o {} {}'.format(newfilepath, filepath)
    print cmd
    os.system(cmd)
    if not os.path.exists(newfilepath):
        raise Exception('ERROR: pdf to html conversion failed')
    with open(newfilepath, 'rb') as f:
        html = f.read()
    text = html2text.html2text(html)
    text = '\n'.join([upper_to_title(line) for line in text.split('\n')])
    text = text.replace('*', '').replace('_', '')
    # os.remove(newfilepath)
    return text, html    


# SEGMENTATION
def get_career_title(tokens):
    d = ['professional', 'work experience', 'career summary',
                      'professional experience', 'project expertise', 'experience', 'experience summary']
    title = ''
    for token in tokens:
        if any([x in token.lower() for x in d]):
            title = token
            break
    return title

def get_skills_title(tokens):
    title = ''
    for token in tokens:
        if any([x in token.lower() for x in ['skill', 'technical']]):
            title = token
            break
    return title

def get_acads_title(tokens):
    title = ''
    for token in tokens:
        if any([x in token.lower() for x in ['education', 'academic', 'academia', 'qualification']]):
            title = token
            break
    return title

def get_bio_title(tokens):
    d = ['personal details', 'bio', 'personal']
    title = ''
    for token in tokens:
        if any([x in token.lower() for x in d]):
            title = token
            break
    return title

def get_segment_titles(tokens):
    career = get_career_title(tokens)
    skills = get_skills_title(tokens)
    acads = get_acads_title(tokens)
    bio = get_bio_title(tokens)
    return {'career': career, 'skills': skills, 'acads': acads, 'bio': bio}

def get_segmented_resume(html, text):
    soup = BeautifulSoup(html, 'html.parser')
    tokens = []
    tokens.extend(soup.find_all('h1'))
    tokens.extend(soup.find_all('h2'))
    tokens.extend(soup.find_all('h3'))
    tokens.extend(soup.find_all('h4'))
    tokens.extend([s for s in soup.find_all('strong') if len(s.text.strip().split(' ')) <= 4])
    tokens.extend([p for p in soup.find_all('p') if len(p.text.strip().split(' ')) <= 4])
    #segment_titles = get_segment_titles([el.text.replace('\n', ' ') for el in tokens])
    segment_titles = get_segment_titles([html2text.html2text(el.text) for el in tokens])
    titles = []
    t = HTMLParser.HTMLParser().unescape(text.lower().replace('\n', ' '))
    for key in segment_titles:
        title = segment_titles[key].replace('\t', ' ')
        if title == '': continue
        titles.append((title, t.index(title.strip().lower()), key))
    sorted_titles = sorted(titles, key=lambda x: x[1])
    # print sorted_titles
    segmented_resume = []
    for i, sorted_title in enumerate(sorted_titles):
        if i == 0 and i == len(sorted_titles) - 1:
            segmented_resume.append({'heading': 'start', 'content': text[0: sorted_title[1]], 'key': 'start'})
            segmented_resume.append({'heading': sorted_title[0], 'content': text[sorted_title[1]:], 'key': sorted_title[2]})
        elif i == 0:
            segmented_resume.append({'heading': 'start', 'content': text[0: sorted_title[1]], 'key': 'start'})
            segmented_resume.append({'heading': sorted_title[0], 'content': text[sorted_title[1]:sorted_titles[i+1][1]], 'key': sorted_title[2]})
        elif i == len(sorted_titles) - 1:
            segmented_resume.append({'heading': sorted_title[0], 'content': text[sorted_title[1]:], 'key': sorted_title[2]})
        else:
            segmented_resume.append({'heading': sorted_title[0], 'content': text[sorted_title[1]:sorted_titles[i+1][1]], 'key': sorted_title[2]})
    return segmented_resume

def parse_matches(doc, matches):
    matched = []
    for match in matches:
        matched.append({'type':nlp.vocab.strings[match[1]],'text': doc[match[2]:match[3]].text})
    return matched

def get_matches(txt):
    doc = nlp(txt)
    matches = matcher(doc)
    return parse_matches(doc, matches)

# CAREER
def get_role(txt):
    matches = get_matches(txt)
    role = [x['text'] for x in matches if x['type'] == 'ROLE']
    if len(role):
        role = role[0]
    else:
        role = ''
    return role

def get_experience(txt):
    exp = re.findall(r'([\d+.]+) [yY]ear.*xperience', txt)
    if len(exp) == 0:
        exp = re.findall(r'xperience.*([\d+.]+) [yY]ear', txt)
    if len(exp) == 0:
            exp = re.findall(r'([\d+.]+)[yY]ear.*xperience', txt)
    exp = exp[0] if len(exp) > 0 else None
    return exp

# def get_skills(txt):
#     skills = []
#     for token in re.split(r'(\n|,|\.\s|\s{2,}|\||:)', txt):
#         token = token.strip()
#         if token.replace(' ','-').lower() in all_skills:
#             skills.append(token)
#     return list(set(skills))

def get_skills(txt):
    txt = txt.replace('/', ' ').replace('-', ' ')
    matches = get_matches(txt)
    skills = [x['text'] for x in matches if x['type'] == 'SKILL']
    skills = list(set(skills))
    skills = None if len(skills) == 0 else skills
    return skills


# BIO
def extract_name_regex(txt):
    t = txt.split('mail')[0]
    name = re.findall(r'\b[A-Z][a-z]+\.[A-Z]\b',t)
    if len(name) == 0:
      name = re.findall(r'\b[A-Z]\.[A-Z][a-z]+\b',t)
    name = name[0] if len(name) != 0 else None
    return name

def is_person_name(token):
    return 'PER' in token

def extract_name_crf(txt):
    person_name = ''
    for line in re.split(r'(\n|\s{3,})', txt):
        tokens = nltk.word_tokenize(line)
        tags = tagger.tag(sent2features(nltk.pos_tag(tokens)))

        for index, tag in enumerate(tags):
            if is_person_name(tag):
                person_name = person_name + " " + tokens[index]
            else:
                if person_name != "":
                    break

        if person_name != '':
            break
    return person_name

def extract_name_spacy(txt):
    doc = nlp(txt)
    for ent in doc.ents:
        if ent.label_ == 'PERSON':
            return ent.text

def extract_email(txt):
    email = re.findall(r'[a-zA-Z0-9_.+]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', txt)
    return email[0] if len(email) > 0 else None

def extract_phone(txt):
    phone = re.findall(r'\+?\d[\d -]{8,12}\d', txt)
    phone = phone[0] if len(phone) > 0 else None
    return phone

def extract_location(txt):
    matches = get_matches(txt)
    location = [x['text'] for x in matches if x['type'] == 'LOCATION']
    location = location[0] if len(location) > 0 else None
    return location

def extract_location_nearest_to_name(txt, name):
    words = re.findall(r'\w+', txt)
    locations = []
    for word in words:
        if word in all_locations:
            locations.append(word)
    locations = list(set(locations))
    name_ix = txt.index(name.strip())
    best_delta = 1e8
    best_location = None
    for location in locations:
        loc_ixs = [m.start() for m in re.finditer(location, txt)]
        for ix in loc_ixs:
            if abs(name_ix - ix) < best_delta:
                best_location = location
                best_delta = abs(name_ix - ix)
    return best_location

def get_bio(segmented_resume):
    txt = '\n'.join([x['content'] for x in segmented_resume])
    name = extract_name_regex(txt)
    if name is None:
        name = extract_name_crf(txt)
    # name = name.replace('\xef','').replace('\xa0','').replace('\x80','').replace('\x83','').replace('\x89','')
    email = extract_email(txt)
    phone = extract_phone(txt)
    bio_segment = [x for x in segmented_resume if x['key'] == 'bio']
    if len(bio_segment) != 0:
        location = extract_location(bio_segment[0]['content'])
        if location is None:
            location = extract_location_nearest_to_name(txt, name)
    else:
        location = extract_location_nearest_to_name(txt, name)
    return [{'key':'name', 'value':name},
             {'key':'email', 'value': email},
             {'key':'phone', 'value': phone},
             {'key':'location', 'value': location}]


def upsert_resume(resume_path):
    print 'upserting resume: {}'.format(resume_path)
    tenant_id = os.path.basename(os.path.dirname(resume_path))
    filename = os.path.basename(resume_path)

    resumes.delete_one({'tenant_id': tenant_id, 'filename': filename})
    d_obj = {}
    d_obj["tenant_id"] = tenant_id
    d_obj["filename"] = filename;
    d_obj["doc_id"] = '.'.join((filename.split('.'))[:-1])

    try:
        file_ext = os.path.splitext(resume_path)[1]
        if file_ext == '.docx':
            txt, html = read_docx(resume_path)
        elif file_ext == '.doc':
            txt, html = read_doc(resume_path)
        elif file_ext == '.rtf':
            txt, html = read_rtf(resume_path)
        elif file_ext == '.pdf':
            txt, html = read_pdf(resume_path)
        else:
            raise Exception('ERROR: Unsupported file type')


        d = []
        d.append(tenant_id)
        d.append(filename)

        segmented_resume = []
        try:
            segmented_resume = get_segmented_resume(html, txt)
        except Exception as e:
            d_obj["name"] = extract_name_crf(txt)
            d_obj["email"] = extract_email(txt)
            d_obj["phone"] = extract_phone(txt)
            d_obj["skills"] = ', '.join(get_skills(txt))
            d_obj["segmented"] = "no"
            print 'inserting resume - {}: {}'.format(tenant_id, filename)
            resumes.insert_one(d_obj)
            return
        
        print [x['key'] for x in segmented_resume]
        if not len(segmented_resume):
            # print txt[:100]
            raise Exception('ERROR: Could not find Career, Skills, Academic or Bio segment')

        for b in get_bio(segmented_resume):
            d.append(b['value'])
        exp = get_experience(txt)
        d.append(exp)
        role = get_role(txt)
        d.append(role)
        # d.append(', '.join([x['heading'].strip() for x in segmented_resume]))
        skills = get_skills('\n'.join([x['content'] for x in segmented_resume if x['key'] in ['skills', 'career']]))
        d.append(', '.join(skills))

        # d = pd.DataFrame([d], columns=['tenant_id', 'filename', 'name', 'email', 'phone', 'location', 'experience', 'role','sections', 'skills'])

        fields = ['tenant_id', 'filename', 'name', 'email', 'phone', 'location', 'experience', 'role', 'skills']

        for ix, field in enumerate(fields):
            d_obj[field] = d[ix]

        d_obj["segmented"] = "yes"
    except Exception as e:
        d_obj['error'] = str(e)

    print 'inserting resume - {}: {}'.format(tenant_id, filename)
    resumes.insert_one(d_obj)

def delete_resume(resume_path):
    tenant_id = os.path.basename(os.path.dirname(resume_path))
    filename = os.path.basename(resume_path)
    print 'deleting resume - {}: {}'.format(tenant_id, filename)
    resumes.delete_one({'tenant_id': tenant_id, 'filename': filename})


print('setting up mongo connection')
client = MongoClient('localhost', 27017)
db = client.resume_matcher
resumes = db.resumes

print('setting up spacy')
nlp = spacy.load('en')

print('creating ontologies')
ontology = []
roles = []
for filepath in glob('../../data/input/ontologies/*.roles'):
    with open(filepath, 'r') as f:
        roles.extend(f.read().split('\n'))
roles = list(set(roles))
roles = [role for role in roles if ',' not in role]
ontology.extend([{'type': 'ROLE', 'text': role} for role in roles])

print('loading crf model')
tagger = pycrfsuite.Tagger()
tagger.open('conll2002-esp.crfsuite')

print('loading locations')
df_locations = pd.read_csv('../../data/input/ontologies/all_india_pin_code.csv')
all_locations = df_locations.Districtname.unique()
ontology.extend([{'type': 'LOCATION', 'text': location} for location in all_locations])

print('loading skills')
with open('../../data/input/ontologies/skills.txt', 'r') as f:
    all_skills = f.read().split('\n')
for skill in all_skills:
    # ontology.extend([{'type': 'SKILL', 'text': skill }])
    ontology.extend([{'type': 'SKILL', 'text': skill.replace('-', ' ') }])

print('setting up ontology matcher')
matcher = spacy.matcher.Matcher(nlp.vocab)
for i, o in enumerate(ontology):
    specs = []
    for w in o['text'].lower().split(' '):
        specs.append({spacy.attrs.LOWER: w})
    specs = [specs]
    matcher.add(entity_key=str(i),
            label=o['type'],
            attrs={},
            specs=specs)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        upsert_resume(filepath)

# resume_filepaths = []
# for dr, dirs, files in os.walk('../../data/input/Resumes/'):
#     for filename in files:
#         if os.path.splitext(filename)[1] in ['.pdf']:
#             resume_filepaths.append(os.path.join(dr, filename))

# print len(resume_filepaths)
# for ix, filepath in enumerate(resume_filepaths):
#     if ix > 30: continue
#     try:
#         upsert_resume(filepath)
#     except Exception as e:
#         print e
#
