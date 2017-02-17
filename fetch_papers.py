'''

Author: doodhwala
Python3 script to fetch the top 100 papers

Instructions:
    1) clone repo
    2) python3 fetch_papers.py
    3) enjoy :)

'''

import os, re, requests

directory = 'papers'

if not os.path.exists(directory):
    os.makedirs(directory)

papers = []
with open('README.md', 'r') as f:
    lines = f.read().split('\n')
    heading, section_path = '', ''
    for line in lines:
        if('###' in line):
            heading = line.strip().split('###')[1]
            heading = heading.replace('/', '|')
            section_path = os.path.join(directory, heading)
            if not os.path.exists(section_path):
                os.makedirs(section_path)
        if('[[pdf]]' in line):
            # The stars ensure you pick up only the top 100 papers
            # modify this if you want to fetch all other papers as well :)
            result = re.search('\*\*(.*?)\*\*.*?\[\[pdf\]\]\((.*?)\)', line)
            if(result):
                paper, url = result.groups()
                if(not os.path.exists(os.path.join(section_path, paper + '.pdf'))):
                    print('Fetching', paper)
                    response = requests.get(url)
                    with open(os.path.join(section_path, paper + '.pdf'), 'wb') as f:
                        f.write(response.content)

# Sorry for the quick and dirty approach
# Feel free to add features
