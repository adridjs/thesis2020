In order to use the gender_change script, you need to install 
[Freeling](http://nlp.lsi.upc.edu/freeling/index.php/node/1). Here are the steps that worked for me, but
 I recommend checking the [documentation](https://freeling-user-manual.readthedocs.io/en/v4.1/).
#Install Swig
In order to install Freeling, we first need to install Swig in our system. The following commands install the needed
 dependencies.
```
sudo apt-get install g++
sudo apt-get install libpcre3 libpcre3-dev
```
Go to [swig](http://www.swig.org/) and download the linux version.
When the download has completed, go to the folder where the package has been saved and execute the following commands. 
```
chmod 777 swig-x.y.z.tar.gz
tar -xzvf swig-x.y.z.tar.gz
```
Specify swig `INSTALL_DIRECTORY`.
```
export INSTALL_DIRECTORY=$HOME
./configure --prefix=$INSTALL_DIRECTORY
```
Compile and install.
```
sudo make
sudo make install
```
Define `SWIG_PATH` environment variable and add it in `PATH` environment variable. 
```
sudo vim /etc/profile
export SWIG_PATH=$INSTALL_DIRECTORY/bin
export PATH=$SWIG_PATH:$PATH
source /etc/profile
```
Verify swig installation.

```
swig -version
```
#Install Freeling
The flag `-DPYTHON3_API` is used in order to be able to import Freeling libraries from Python 3.
```
sudo apt install cmake
sudo apt-get install zlib1g-dev build-essential libboost-all-dev
sudo apt-get install -y libicu-dev
mkdir build
cd build/
cmake .. -DPYTHON3_API=ON
```
