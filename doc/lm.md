## Individual language models

If you plan to let the tool generate individual language models per text,
you have to get (essentially build) [KenLM](https://kheafield.com/code/kenlm/).
Before doing this, you should install its [dependencies](https://kheafield.com/code/kenlm/dependencies/).
For Debian based systems this can be done through:
```bash
$ sudo apt-get install build-essential libboost-all-dev cmake zlib1g-dev libbz2-dev liblzma-dev 
```

With all requirements fulfilled, there is a script for building and installing KenLM
and the required DeepSpeech tools in the right location:
```bash
$ bin/lm-dependencies.sh
```

If all went well, the alignment tool will find and use it to automatically create individual
language models for each document.