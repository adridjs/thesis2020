# GebioToolkit

We present GebioToolkit for extracting multilingual parallel corpora at sentence level, with document and gender information from Wikipedia biographies. Despite the gender inequalities present in Wikipedia, the toolkit has been designed to extract corpus balanced in gender. 
While our toolkit is able to customize the languages for which we are extracting the multilingual corpus, in this work, we extracted a corpus of 2000 sentences in English, Spanish and Catalan, which has been post-edited by native speakers to be valid as test dataset for machine translation.

## Dependencies

* Python 3.6
* Numpy, tested with 1.16.4
* LASER (https://github.com/facebookresearch/LASER)

## Usage

### Corpus extractor

_Change domain_

To generate files with sentences from the desired domain, we need to get a list of wikipedia entries. The easiest way to
 achieve this
 is by using the petscan tool (https://petscan.wmflabs.org/). We then execute the following command.

    python3 wp_api_language_search.py -csv new_list.csv 

We can also extract page files by giving a `$NAMELIST_FILE` to the WikiExtractor, together with the `$WIKIDUMP_PATH`
 and an `$OUTPUT_PATH`.  

    python 3 we_modified.py \
    $WIKIDUMP_PATH  \
    -o $OUTPUT_PATH \
    --filter_category $NAMELIST_FILE \
    -l
    
Number of names in english and spanish, respectively.
```
cat en_namelist.txt | wc -l
33604
cat es_namelist.txt | wc -l
31373
```

Number of names that appear in both namelists.
```
24675
```

Number of lines after aligning. Take into account that for each person, there are 3 lines that don't contain
sentences:
*  `<doc>` and  `</doc>` (contains document metadata)  
* `<title>`
```
~/thesis2020$ cat gebiotoolkit/corpus_alignment/aligned/en_she.txt | wc -l
27012
~/thesis2020$ cat gebiotoolkit/corpus_alignment/aligned/en_he.txt | wc -l
47750
```

## References

For more information, please check https://arxiv.org/pdf/1912.04778.pdf

