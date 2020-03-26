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
    name = sens.split(':')[0].strip()
    valid_sentence = re.sub(name + ': ', '', sens)
    valid_sentence = re.sub('\n', '', valid_sentence)
    return valid_sentence, name


def store_sentences(filestore, name, person_sentences, lang=None, gender=None, format=None):
    """
    Writes :param person_sentences in the given :param filestore:
    If :name is specified, the function assumes that
    :param filestore:
    :param person_sentences:
    :param lang:
    :param gender:
    :param name:
    """
    format = format or 'xml'
    if format == 'xml':
        seg = 1
        try:
            idd = id_retriever(name, lang)
        except:
            idd = None
        filestore.write(f'<doc docid="{name}" wpid="{idd}" language="{lang}"  gender="{gender}">\n')
        filestore.write(f'<title>{name}</title>\n')

        for p_sentence in person_sentences:
            filestore.write(f'<seg id="{str(seg)}">{p_sentence}<\\seg>\n')
            seg += 1

        filestore.write('</doc>\n')
    else:
        NotImplementedError('The only format currently supported is xml.')
