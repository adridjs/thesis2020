from File_selection import File_selection
import collections
import argparse

class language_identification(object):
    def __init__(self, dictionary_of_names, languages, save_folder = 'cat/'):
        self.dictionary_of_names = dictionary_of_names
        self.languages = languages
        self.p = File_selection(self.name_dictionaries, self.languages)
        self.file_name = '_namelist.txt'
        self.folder = save_folder

    def add_english(self):
        self.languages.append('en')
    
    def find_titles_by_language(self):
        by_languages = {i:[] for i in self.languages}
        by_languages['en'] = []
        for k, v in self.p.selected_people.items():
            by_languages['en'].append(k + ' – ' + 'English')
            for lan in v:
                if lan[0] in by_languages.keys():
                    by_languages[lan[0]].append(lan[1])        
        return by_languages
    
    def generate_parse(self):
        self.add_english()
        by_language = self.find_titles_by_language()
        position = {i:[] for i in self.languages}
        for k, v in by_language.items():
            for name in v:
                position[k] = position[k] + name.split(' – ')
        for k, v in position.items():
            print('Generating namelists: ' + k )
            counter = collections.Counter(position[k])
            counter = counter.most_common()
            f = open(self.folder + k + self.file_name, "w")
            for i in range(1,len(counter)):
                f.write(counter[i][0] + '\n')
            f.close()


def retrieve_args():
    parser = argparse.ArgumentParser(description='Generate lists of names of the selected languages by using the csv generated in Petscan, these list are stored in a folder')
    parser.add_argument('-l','--languages', nargs='+', help='Languages in which the lists will be generated', default='')
    parser.add_argument('-s','--store_folder', required=True, help='folder where the files will be stored')
    parser.add_argument('-c','--csv', required=False, help='CSV generated on Petscan', default='categories/dF7mxWq4.csv')
    args = parser.parse_args()
    return args

def main():
    dictionary_of_names = 'dictionary_of_names/'
    args = retrieve_args()
    languages = args.languages
    save_folder = args.store_folder
    c = language_identification(dictionary_of_names, languages, save_folder = save_folder)
    c.generate_parse()

if __name__ == '__main__':
    main()

    


