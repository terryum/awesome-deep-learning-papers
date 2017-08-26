'''
Author: doodhwala
Python3 script to fetch the top 100 papers
'''

import os, re, requests, codecs

directory = 'papers'
if not os.path.exists(directory):
    os.makedirs(directory)

papers = []
with codecs.open('README.md', encoding='utf-8', mode='r', buffering=1, errors='strict') as f:
    lines = f.read().split('\n')
    heading, section_path = '', ''
    for line in lines:
        if('###' in line):
            heading = line.strip().split('###')[1]
            win_restricted_chars = re.compile(r'[\^\/\\\:\*\?\"<>\|]')
            heading = win_restricted_chars.sub("", heading)
            section_path = os.path.join(directory, heading)
            if not os.path.exists(section_path):
                os.makedirs(section_path)
        if('[[pdf]]' in line):
            # The stars ensure you pick up only the top 100 papers
            # Modify the expression if you want to fetch all other papers as well
            result = re.search('\*\*(.*?)\*\*.*?\[\[pdf\]\]\((.*?)\)', line)
            if(result):
                paper, url = result.groups()
                paper = win_restricted_chars.sub("", paper)
                # Auto - resume functionality
                if(not os.path.exists(os.path.join(section_path, paper + '.pdf'))):
                    print('Fetching', paper)
                    try:
                        response = requests.get(url)
                        with open(os.path.join(section_path, paper + '.pdf'), 'wb') as f:
                            f.write(response.content)
                    except requests.exceptions.RequestException as e:
                        print("Error: {}".format(e))
