from functools import wraps
from flask import Flask, redirect, request, current_app, jsonify, Response
import pickle
import pandas as pd
from io import StringIO, BytesIO
import json
import requests
import time
import base64
import pandas as pd
import re
from glob import glob
import spacy
from pymongo import MongoClient

print('setting up mongo connection')
client = MongoClient('localhost', 27017)
db = client.resume_matcher
resumes = db.resumes

app = Flask(__name__, static_url_path='')

def check_auth(username, password):
    return username == 'admin' and password == 'Aev54I0Av13bhCxM'

def authenticate():
    """Sends a 401 response that enables basic auth"""
    return Response(
    'Could not verify your access level for that URL.\n'
    'You have to login with proper credentials', 401,
    {'WWW-Authenticate': 'Basic realm="Login Required"'})

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated

# API FUNCS
def remove_overlapping_matches(matches): #TODO fix
    m = []
    for i, match in enumerate(matches):
        if i == 0:
            m.append(match)
            continue
        prev_match = m[-1]
        if prev_match[2] == match[2] or prev_match[3] == match[3]:
            s1 = match[3] - match[2]
            s2 = prev_match[3] - prev_match[2]
            if s1 > s2:
                m.remove(prev_match)
                m.append(match)
        else:
            m.append(match)
    return m

def get_matches(txt):
    doc = nlp(txt)
    matches = matcher(doc)
    matches = remove_overlapping_matches(matches)
    matched = []
    for match in matches:
        matched.append({'type':nlp.vocab.strings[match[1]],
                        'text': doc[match[2]:match[3]].text})
    return matched

def get_role(txt):
    matches = get_matches(txt)
    role = [x['text'] for x in matches if x['type'] == 'ROLE']
    role = role[0] if len(role) > 0 else None
    return role

def get_experience(txt):
    exp = re.findall(r'(\d+ - \d+) [yY]ear', txt)
    if len(exp) == 0:
        exp = re.findall(r'(\d+-\d+) [yY]ear', txt)
    if len(exp) == 0:
        exp = re.findall(r'(\d+ to \d+) [yY]ear', txt)
    if len(exp) == 0:
        exp = re.findall(r'([\d+]+) [yY]ear', txt)
    exp = exp[0].replace('+', '') if len(exp) > 0 else None
    return exp

def get_skills(txt):
    txt = txt.replace('/', ' ').replace('-', ' ')
    matches = get_matches(txt)
    skills = [x['text'] for x in matches if x['type'] == 'SKILL']
    skills = list(set(skills))
    skills = None if len(skills) == 0 else skills
    return skills

# def extract_location(txt):
#     words = re.findall(r'\w+', txt)
#     location = None
#     for word in words:
#         if word in locations:
#             location = word
#             break
#     return location

def extract_location(txt):
    matches = get_matches(txt)
    location = [x['text'] for x in matches if x['type'] == 'LOCATION']
    location = location[0] if len(location) > 0 else None
    return location

def get_parsed_jd(jd):
	role = get_role(jd)
	exp = get_experience(jd)
	skills = get_skills(jd)
	location = extract_location(jd)
	return role, exp, skills, location

def get_response(tenant_id, jd, weights):
	role, exp, skills, location = get_parsed_jd(jd)

	role_weight = int(weights['role'])
	exp_weight = int(weights['exp'])
	skills_weight = int(weights['skills'])
	location_weight = int(weights['location'])

	weights_sum = 0.
	if role is not None:
		weights_sum += role_weight
		roles = [r for r in role_synonyms if role.lower() in r]
		roles = roles[0].split(',') if len(roles) > 0 else [role.lower()]		
	if exp is not None:
		weights_sum += exp_weight
		if 'to' not in exp and '-' not in exp:
			exp_range = False
			exp = float(exp)
		else:
			exp_range = True
			exp_ge = float(re.split(r'to|-', exp)[0].strip())
			exp_le = float(re.split(r'to|-', exp)[1].strip())
	if skills is not None:
		weights_sum += skills_weight
		jd_skills = [skill.lower() for skill in skills]
	if location is not None:
		weights_sum += location_weight
		locations = [l for l in loc_synonyms if location.lower() in l]
		locations = locations[0].split(',') if len(locations) > 0 else [location.lower()]
	# print 'weights_sum', weights_sum

	def calculate_score(row):
		role_score = 0.
		exp_score = 0.
		skills_score = 0.
		loc_score = 0.
		test = row['name'] in [" Shashank Bansode", " Debjit Banerjee"]
		test = False

		if role is not None and row['c_role'] in roles:
			role_score = role_weight
			if test: print 'setting role score'
		
		if exp is not None:
			resume_exp = row['c_experience']
			if exp_range:
				if exp_ge <= resume_exp <= exp_le:
					exp_score = exp_weight
					if test: print 'setting exp score'
			else:
				if resume_exp >= exp:
					exp_score = exp_weight
					if test: print 'setting exp score'
		
		if skills is not None:
			resume_skills = [skill.lower() for skill in row['skills']]
			intersection = set(jd_skills).intersection(resume_skills)
			s_score = 1. * len(intersection)/len(jd_skills)
			skills_score = s_score * skills_weight
			if test: print row['name']
			if test: print intersection
			if test: print 'len(intersection)', len(intersection)
			if test: print 'skills score', s_score * skills_weight

		if location is not None and row['c_location'] in locations:
			loc_score += location_weight
			if test: print 'setting location score'

		total_score = role_score + exp_score + skills_score + loc_score
		total_score = (total_score/weights_sum * 100)
		role_score = (role_score/weights_sum * 100)
		exp_score = (exp_score/weights_sum * 100)
		skills_score = (skills_score/weights_sum * 100)
		loc_score = (loc_score/weights_sum * 100)
		return {'total': total_score, 'role': role_score, 'exp': exp_score,
							'skills': skills_score, 'loc': loc_score,}

	r_array = []
	for resume in resumes.find({"tenant_id": tenant_id}):
		r_array.append(resume)
	if(len(r_array) == 0):
		return {'parsed_jd': jd, 'resumes': []}

	df_resumes = pd.DataFrame(r_array)

	df_resumes.experience.fillna('0',inplace=True)
	df_resumes.location.fillna('None',inplace=True)
	df_resumes.skills.fillna('',inplace=True)
	df_resumes.role.fillna('',inplace=True)
	df_resumes['phone'] = df_resumes.phone.fillna('0')
	df_resumes['skills'] = df_resumes.skills.apply(lambda x: x.split(', '))
	df_resumes['c_experience'] = df_resumes.experience.apply(lambda x:  0 if(x=='+') else float(x.replace('+', '')))
	df_resumes['c_role'] = df_resumes.role.apply(lambda x: x.lower())
	df_resumes['c_location'] = df_resumes.location.apply(lambda x: x.lower())

	if weights_sum == 0:
		df_resumes['score'] = 0
		df_resumes['total_score'] = 0
	else:
		df_resumes['score'] = df_resumes.apply(calculate_score, axis=1)
		df_resumes['total_score'] = df_resumes.score.apply(lambda x: x['total'])

	df_resumes.sort('total_score', ascending=False, inplace=True)
	df_resumes = df_resumes[df_resumes.score.ge(50)]

	drop_cols = [col for col in df_resumes.columns.tolist() if 'c_' in col]
	drop_cols.remove('doc_id')
	drop_cols.append('_id')
	df_resumes.drop(drop_cols, axis=1, inplace=True)

	#JD OBJECT
	jd = []
	jd.append({'key': 'role', 'value': role})
	jd.append({'key': 'experience', 'value': exp})
	jd.append({'key': 'location', 'value': location})
	jd.append({'key': 'skills', 'value': skills})

	return {'parsed_jd': jd, 'resumes': json.loads(df_resumes.to_json(orient='records').decode('utf-8', 'ignore'))}


# REQUEST HANDLERS
@app.route('/')
@requires_auth
def root():
	 return app.send_static_file('index.html')

@app.route('/parse_jd', methods=['POST'])
@requires_auth
def parse_jd():
	data = request.get_json()
	if 'jd' not in data:
		return 'Missing param "jd"', 422
	role, exp, skills, location = get_parsed_jd(data['jd'])
	return json.dumps({'role': role, 'exp': exp, 'skills': skills, 'location': location})

@app.route('/get_parsed_resume', methods=['POST'])
@requires_auth
def get_parsed_resume():
	data = request.get_json()
	if 'tenant_id' not in data:
		return 'Missing param "tenant_id"', 422
	tenant_id = data['tenant_id']
	if 'doc_id' not in data:
		return 'Missing param "doc_id"', 422
	doc_id = data['doc_id']
	resume = resumes.find_one({'tenant_id': tenant_id, 'doc_id': doc_id})
	if resume is None:
		return '{}'
	else:
		response_fields = ['filename', 'name', 'email', 'phone','doc_id','error','tenant_id']
		response = {}
		for field in response_fields:
			if (field in resume):
				response[field] = resume[field]
			else:
				response[field] = ''
		return json.dumps(response)

@app.route('/find_resumes', methods=['POST'])
@requires_auth
def find_resumes():
	data = request.get_json()
	if 'tenant_id' not in data:
		return 'Missing param "tenant_id"', 422
	tenant_id = data['tenant_id']
	if 'jd' not in data:
		return 'Missing param "jd"', 422
	jd = data['jd']
	if 'weights' not in data:
		return 'Missing param "weights"', 422
	weights = data['weights']
	missing_param = 'role'
	try:
		role_weight = int(weights['role'])
		missing_param = 'exp'
		exp_weight = int(weights['exp'])
		missing_param = 'skills'
		skills_weight = int(weights['skills'])
		missing_param = 'location'
		location_weight = int(weights['location'])
	except Exception as e:
		return 'Missing  or invalid value for param "' + missing_param + '" in weights', 422
	# print 'response'
	response = get_response(tenant_id, jd, weights)
	return json.dumps(response)
    

with open('../../data/input/ontologies/role-synonyms', 'r') as f:
	role_synonyms = f.read().split('\n')

with open('../../data/input/ontologies/loc-synonyms', 'r') as f:
	loc_synonyms = f.read().split('\n')	

print 'loading locations'
locations = pd.read_csv('../../data/input/ontologies/all_india_pin_code.csv').Districtname.unique().tolist()
locations.extend([loc.title() for line in loc_synonyms for loc in line.split(',')])
locations.extend(pd.read_csv('../../data/input/ontologies/us_states.csv').Name.tolist())
locations = list(set(locations))
locations = [str(loc).lower() for loc in locations]

print 'loading skills'
with open('../../data/input/ontologies/skills.txt', 'r') as f:
    all_skills = f.read().split('\n')


print 'loading roles'
ontology = []
roles = []
for filepath in glob('../../data/input/ontologies/*.roles'):
    with open(filepath, 'r') as f:
        roles.extend(f.read().split('\n'))
roles = list(set(roles))
roles = [role for role in roles if ',' not in role]
ontology.extend([{'type': 'LOCATION', 'text': location} for location in locations])
ontology.extend([{'type': 'ROLE', 'text': role} for role in roles])
for skill in all_skills:
	# ontology.extend([{'type': 'SKILL', 'text': skill }])	
	ontology.extend([{'type': 'SKILL', 'text': skill.replace('-', ' ') }])	
print 'setting up matcher'
nlp = spacy.load('en')
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

# app.run(debug=False, use_reloader=False, host='0.0.0.0', port=8000)

