import argparse
import os
import pyfreeling
import sys

from gender_bias.data_driver import DataDriver


def print_tree(ptree, depth):
    """
    Outputs a parse tree.
    :param ptree:
    :param depth:
    """
    node = ptree.begin()

    print(''.rjust(depth * 2), end='')
    info = node.get_info()
    if (info.is_head()): print('+', end='')

    nch = node.num_children()
    if (nch == 0):
        w = info.get_word()
        print('({0} {1} {2})'.format(w.get_form(), w.get_lemma(), w.get_tag()), end='')

    else:
        print('{0}_['.format(info.get_label()))

        for i in range(nch):
            child = node.nth_child_ref(i)
            print_tree(child, depth + 1)

        print(''.rjust(depth * 2), end='')
        print(']', end='')

    print('')


# ------------  output a parse tree ------------
def print_depth_tree(dtree, depth):
    node = dtree.begin()

    print(''.rjust(depth * 2), end='')

    info = node.get_info()
    link = info.get_link()
    linfo = link.get_info()
    print('{0}/{1}/'.format(link.get_info().get_label(), info.get_label()), end='')

    w = node.get_info().get_word()
    print('({0} {1} {2})'.format(w.get_form(), w.get_lemma(), w.get_tag()), end='')

    nch = node.num_children()
    if (nch > 0):
        print(' [')

        for i in range(nch):
            d = node.nth_child_ref(i)
            if (not d.begin().get_info().is_chunk()):
                print_depth_tree(d, depth + 1)

        ch = {}
        for i in range(nch):
            d = node.nth_child_ref(i)
            if (d.begin().get_info().is_chunk()):
                ch[d.begin().get_info().get_chunk_ord()] = d

        for i in sorted(ch.keys()):
            print_depth_tree(ch[i], depth + 1)

        print(''.rjust(depth * 2), end='')
        print(']', end='')

    print('')


def retrieve_args():
    parser = argparse.ArgumentParser(
        description='Use FreeLing library in order to change the morphological gender of the given text file.')
    parser.add_argument('-cl', '--command-line',
                        action='store_true',
                        help='If set, it will get its input from command line. Usage: invert_genders.py < text_file.txt')
    parser.add_argument('-f', '--file', help='text file to analyze', default='../gender_bias/biographies/en_he.filtered.xml')
    args = parser.parse_args()
    return args


def main():
    args = retrieve_args()
    # check whether we know where to find FreeLing data files
    if "FREELINGDIR" not in os.environ:
        if sys.platform == "win32" or sys.platform == "win64":
            os.environ["FREELINGDIR"] = "C:\\Program Files"
        else:
            os.environ["FREELINGDIR"] = "/usr/local/"
        print("FREELINGDIR environment variable not defined, trying ", os.environ["FREELINGDIR"], file=sys.stderr)

    # check if folder exists
    if not os.path.exists(os.environ["FREELINGDIR"] + "/share/freeling"):
        print("Folder", os.environ["FREELINGDIR"] + "/share/freeling",
              "not found.\nPlease set FREELINGDIR environment variable to FreeLing installation directory",
              file=sys.stderr)
        sys.exit(1)

    # Location of FreeLing configuration files.
    DATA = os.environ["FREELINGDIR"] + "/share/freeling/"

    # Init locales
    pyfreeling.util_init_locale("default")

    # create options set for maco analyzer. Default values are Ok, except for data files.
    LANG = "es"
    op = pyfreeling.maco_options(LANG)
    op.set_data_files("",
                      DATA + "common/punct.dat",
                      DATA + LANG + "/dicc.src",
                      DATA + LANG + "/afixos.dat",
                      "",
                      DATA + LANG + "/locucions.dat",
                      DATA + LANG + "/np.dat",
                      DATA + LANG + "/quantities.dat",
                      DATA + LANG + "/probabilitats.dat")

    # create analyzers
    tokenizer = pyfreeling.tokenizer(DATA + LANG + "/tokenizer.dat")
    splitter = pyfreeling.splitter(DATA + LANG + "/splitter.dat")
    sess_id = splitter.open_session()
    morph_analyzer = pyfreeling.maco(op)

    # activate morphological modules to be used in next call
    morph_analyzer.set_active_options(False, True, True, True,  # select which among created
                          True, True, False, True,  # submodules are to be used.
                          True, True, True, True)  # default: all created submodules are used

    # create tagger, sense anotator, and parsers
    tagger = pyfreeling.hmm_tagger(DATA + LANG + "/tagger.dat", True, 2)
    sense_annotator = pyfreeling.senses(DATA + LANG + "/senses.dat")
    parser = pyfreeling.chart_parser(DATA + LANG + "/chunker/grammar-chunk.dat")
    dependencies = pyfreeling.dep_txala(DATA + LANG + "/dep_txala/dependences.dat", parser.get_start_symbol())

    # process input text
    if args.command_line:
        line = sys.stdin.readline()
        while line:
            analyze_sentence(line, tokenizer, splitter, sess_id, morph_analyzer, tagger, sense_annotator, parser, dependencies)
    else:
        dd = DataDriver('../gender_bias/biographies/', languages={'es'})
        docs, _ = dd._get_biographies_corpus()
        sentences_list = docs['es_he']
        # for line in open(args.file).readlines():
        for sentence in sentences_list:
            analyze_sentence(' '.join(sentence), tokenizer, splitter, sess_id, morph_analyzer, tagger, sense_annotator, parser, dependencies)


def analyze_sentence(sentence, tokenizer, splitter, sess_id, morph_analyzer, tagger, sense_annotator, parser, dependencies):
    """
    The tagset is available here: https://freeling-user-manual.readthedocs.io/en/v4.1/tagsets/tagset-{lang} where lang is the alpha2code of the
    language in use
    :param sentence:
    :param tokenizer:
    :param splitter:
    :param sess_id:
    :param morph_analyzer:
    :param tagger:
    :param sense_annotator:
    :param parser:
    :param dependencies:
    :return:
    """
    token_str = tokenizer.tokenize(sentence)
    ls = splitter.split(sess_id, token_str, False)
    ls = morph_analyzer.analyze(ls)
    ls = tagger.analyze(ls)
    ls = sense_annotator.analyze(ls)
    ls = parser.analyze(ls)
    ls = dependencies.analyze(ls)

    # iterate over each sentence and output the result
    for s in ls:
        ws = s.get_words()
        for w in ws:
            print(w.get_form() + " " + w.get_lemma() + " " + w.get_tag() + " " + w.get_senses_string())
        print("")

        tr = s.get_parse_tree()
        print_tree(tr, 0)

        dp = s.get_dep_tree()
        print_depth_tree(dp, 0)

    splitter.close_session(sess_id)


if __name__ == '__main__':
    main()