# GebioToolkit

We present GebioToolkit for extracting multilingual parallel corpora at sentence level, with document and gender information from Wikipedia biographies. Despite the gender inequalities present in Wikipedia, the toolkit has been designed to extract corpus balanced in gender. 
While our toolkit is able to customize the languages for which we are extracting the multilingual corpus, in this work, we extracted a corpus of 2000 sentences in English, Spanish and Catalan, which has been post-edited by native speakers to be valid as test dataset for machine translation.

## Dependencies

* Python 3.6
* Numpy, tested with 1.16.4
* LASER (https://github.com/facebookresearch/LASER)

## Usage

### Corpus Extraction

_Change domain_

To generate files with sentences from the desired domain, we need to get a list of wikipedia entries. The easiest way to
 achieve this is by using the petscan tool (https://petscan.wmflabs.org/). We then execute the following command.

    python3 wp_api_language_search.py -csv new_list.csv 

We can also extract page files by giving a `$NAMELIST_FILE` to the WikiExtractor, together with the `$WIKIDUMP_PATH`
 and an `$OUTPUT_PATH`.  

    python 3 we_modified.py \
    $WIKIDUMP_PATH  \
    -o $OUTPUT_PATH \
    --filter_category $NAMELIST_FILE \
    -l

### Corpus Alignment
In order to align the extracted sentences from Wikipedia, we need 
to provide a white-spaced list of languages via the `-l` argument; 
a corpus_folder, `-f`; a save path, `-s`; and the path to the LASER 
repository. LASER will normally reside in your home folder, that is, 
`$LASER='$HOME/LASER'` .

    python 3 align.py \
    -l [LANGUAGES] \
    -f ../corpus_extraction/wiki \
    -s aligned/ \
    -e $LASER/models/bilstm.93langs.2018-12-26.pt
   
## References

For more information, please check https://arxiv.org/pdf/1912.04778.pdf

