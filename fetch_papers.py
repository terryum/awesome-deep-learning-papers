'''
Author: doodhwala
Python3 script to fetch the top 100 papers
'''

import codecs
import os
import re

import requests

# Your proxies settings
proxies = {
    "http": "socks5://127.0.0.1:1723",
    "https": "socks5://127.0.0.1:1723",
}


def rename(name):
    return name.replace(':', '-').replace('?', '')


directory = 'papers'
if not os.path.exists(directory):
    os.makedirs(directory)

papers = []
with codecs.open('README.md', encoding='utf-8', mode='r', buffering=1, errors='strict') as f:
    lines = f.read().split('\n')
    heading, section_path = '', ''
    for line in lines:
        if ('###' in line):
            heading = line.strip().split('###')[1]
            heading = heading.replace('/', '&').replace(':', '-')
            section_path = os.path.join(directory, heading)
            if not os.path.exists(section_path):
                os.makedirs(section_path)
        if ('[[pdf]]' in line):
            # The stars ensure you pick up only the top 100 papers
            # Modify the expression if you want to fetch all other papers as well
            result = re.search('\*\*(.*?)\*\*.*?\[\[pdf\]\]\((.*?)\)', line)
            if (result):
                paper, url = result.groups()
                # Auto - resume functionality
                filename = rename(paper)
                if (not os.path.exists(os.path.join(section_path, filename + '.pdf'))):
                    print('Fetching', paper)
                    try:
                        response = requests.get(url, proxies=proxies)
                        with open(os.path.join(section_path, filename + '.pdf'), 'wb') as f:
                            f.write(response.content)
                    except:
                        print("Exception occur.")
