#!/usr/bin/env bash

version="0.6.0"
dir="deepspeech-${version}-models"
archive="${dir}.tar.gz"

mkdir -p models
cd models
if [[ ! -f $archive ]] ; then
    wget "https://github.com/mozilla/DeepSpeech/releases/download/v${version}/${archive}"
fi

tar -xzvf $archive
mv $dir en
