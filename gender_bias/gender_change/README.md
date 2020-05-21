In order to use the gender_change script, you need to install 
[Freeling](http://nlp.lsi.upc.edu/freeling/index.php/node/1). Here are the steps that worked for me, but
 I recommend checking the [documentation](https://freeling-user-manual.readthedocs.io/en/v4.1/) for a more detailed
  description.
#Install Swig
In order to install Freeling, we first need to install Swig in our system. The following commands install the needed
 dependencies.
```
sudo apt-get install g++
sudo apt-get install libpcre3 libpcre3-dev
```
Go to [swig](http://www.swig.org/) and download the linux version if you want to compile it from source. A faster way
 is to install it via apt:
```
sudo apt install swig
```
Verify swig installation.
```
swig -version
```
#Install Freeling
The flag `-DPYTHON3_API` is used in order to be able to import Freeling libraries from Python 3. The default
 installation directory is `/usr/local/share/freeling/`
```
sudo apt install cmake
sudo apt-get install zlib1g-dev build-essential libboost-all-dev
sudo apt-get install -y libicu-dev
mkdir build
cd build/
cmake .. -DPYTHON3_API=ON
sudo make install
```
Make sure that python script can see both `pyfreeling.py` and `_pyfreeling.so`
```
export PYTHONPATH=$PYTHONPATH:/usr/local/share/freeling/APIs/python3
```

#Modules
###Gender Change
In this module we will try to augment data by inverting the gender of each sentence, if applicable. In order
to do so, we will need to define rulesets for each language, that will be defined 

`gender_change_spacy.py` writes to a `log_filename` if provided, in this format:
```
key: lang_gender | {doc} -----> {changed_doc}'
```