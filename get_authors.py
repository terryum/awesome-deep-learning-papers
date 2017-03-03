
# coding: utf-8

import re
import requests
from html.parser import HTMLParser
import codecs

search_engine="https://www.semanticscholar.org/search?q="
post_fix = "&sort=relevance&ae=false"

class AuthorParser( HTMLParser ):
    tail_string = "" #contains the last tag's name which point to author field
    m_Stop = False
    m_authors = []
    def handle_starttag(self, tag, attr):
        if self.m_Stop:
            return
        if tag == 'article':
            self.tail_string += tag
            return
        if self.tail_string != "":
            #print("search already kick-off")
            self.tail_string = self.tail_string+"."+tag
            #print(self.tail_string)
    def handle_endtag(self, tag):
        if self.m_Stop :
            return
        if self.tail_string == "article":
            # ONLY handle the first article
            self.m_Stop = True
        if self.tail_string != "":
            tags = self.tail_string.split('.')
            tags.reverse()
            for t in tags:
                if t == tag:
                    tags.remove(t)
                    break
            self.tail_string = ""
            tags.reverse()
            for i,t in enumerate(tags):
                self.tail_string = self.tail_string + "." + t if i > 0 else t

    def handle_data(self, data):
        if self.m_Stop:
            return
        if self.tail_string == "article.header.ul.li.span.span.a.span.span":
            #print(data)
            self.m_authors.append(data)

    def get_authors(self):
        return self.m_authors

    def clean(self):
        self.m_authors = []
        self.tail_string= ""
        self.m_Stop = False


def getPaperNames( readme_file ):
    paper_list = []
    with codecs.open( readme_file,encoding='utf-8',mode='r',buffering = 1, errors='strict' ) as f:
        lines = f.read().split('\n')
        heading, section_path = '', ''
        for line in lines:
            if('###' in line):
                heading = line.strip().split('###')[1]
                heading = heading.replace('/', '|')

            if('[[pdf]]' in line):
                # The stars ensure you pick up only the top 100 papers
                # Modify the expression if you want to fetch all other papers as well
                result = re.search('\*\*(.*?)\*\*.*?\[\[pdf\]\]\((.*?)\)', line)
                if(result):
                    paper, url = result.groups()
                    paper_list.append(paper)

    return paper_list

all_papers = getPaperNames("README.md")

author_parser = AuthorParser()
author_dict = {}
for index,paper in enumerate(all_papers):
    paper.replace(" ", "%20")
    search_result = requests.get(search_engine + paper + post_fix)
    author_parser.feed(search_result.text)
    #print( paper, '==>', author_parser.get_authors() )
    authors = author_parser.get_authors()
    for weight, author in enumerate( authors):
        if author not in author_dict.keys():
            author_dict[author] = []
                    
        author_dict[author].append( (weight+1,paper))
    author_parser.clean()
    print("Processed %d |"%(index), paper)

# example usage of author information
with open( "author.csv",'w') as fcsv:
    for (author, papers) in author_dict.items():
        score = 0.0
        for (weight, paper) in papers:
            score += 1.0/weight
        print(author," score: %.2f"%score)
        fcsv.write( author+','+"%.2f"%score)



