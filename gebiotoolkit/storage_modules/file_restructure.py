#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 00:46:36 2019

@author: plxiv
"""

import re
from urllib.request import urlopen

import wikipedia
from bs4 import BeautifulSoup


def id_retriever(name, lan):
    pagename = ''
    if lan != 'en':
        wikipedia.set_lang("en")
        page = wikipedia.page(name)
        soup = BeautifulSoup(urlopen(page.url))
        for el in soup.select('li.interlanguage-link > a'):
            if lan == el.get('lang'):
                pagename = el.get('title').split(' – ')[0]
        wikipedia.set_lang(lan)
    else:
        pagename = name

    if not pagename:
        return 'None'
    page = wikipedia.page(pagename)
    idd = page.pageid
    print(idd)
    return idd


def include_sentence(sens):
    name = sens.split(':')[0]
    valid_sentence = re.sub(name + ': ', '', sens)
    valid_sentence = re.sub('\n', '', valid_sentence)
    return valid_sentence, name[:len(name) - 1]


def store_sentences(filestore, name, all_sentences, lan, gender):
    seg = 1
    try:
        idd = id_retriever(name, lan)
    except:
        idd = None
    filestore.write(f'<doc docid="{name}" wpid="{idd}" language="{lan}" gender="{gender}">\n')
    filestore.write(f'<title>{name}</title>\n')

    for z in all_sentences:
        filestore.write(f'<seg id="{str(seg)}">{z}<\\seg>\n')
        seg += 1

    filestore.write('</doc>\n')
