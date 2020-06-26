#!/usr/bin/env bash

version="0.7.1"
dir="deepspeech-${version}-models"
am="${dir}.pbmm"
scorer="${dir}.scorer"

mkdir -p models/en
cd models/en

if [[ ! -f $am ]] ; then
    wget "https://github.com/mozilla/DeepSpeech/releases/download/v${version}/${am}"
fi

if [[ ! -f $scorer ]] ; then
    wget "https://github.com/mozilla/DeepSpeech/releases/download/v${version}/${scorer}"
fi

if [[ ! -f "alphabet.txt" ]] ; then
    wget "https://raw.githubusercontent.com/mozilla/DeepSpeech/master/data/alphabet.txt"
fi
