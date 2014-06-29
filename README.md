# README

This repo was initially forked from the <a href="http://kaldi.sourceforge.net/">Kaldi</a> project (Revision: 1493). 

The focus of this version would be the noise robustness in speech recognition and especially for the deep neural network based ASR systems. 

***Note for Mac OS Marvericks***: 
It is better to use Homebrew to install gcc and replace the default gcc. gcc46 was tested.
```
brew install gcc46
sudo ln -s `which gcc-4.6` `which gcc`
sudo ln -s `which g++-4.6` `which g++`
sudo ln -s `which gcov-4.6` `which gcov`
sudo ln -s `which c++-4.6` `which c++`
sudo ln -s `which cpp-4.6` `which cpp`
```

### Highlights


### TODOs:
* VTS-GMM
* VTS-MVN for DNN
* Spectral Masking
