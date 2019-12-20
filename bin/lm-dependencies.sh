#!/usr/bin/env bash

basedir="$(pwd)"

mkdir -p dependencies
pushd dependencies


wget -N https://kheafield.com/code/kenlm.tar.gz
tar -xzvf kenlm.tar.gz
pushd kenlm

mkdir -p build
pushd build
cmake ..
make -j 4
popd

popd


source $basedir/venv/bin/activate
mkdir -p deepspeech
pushd deepspeech
python $basedir/bin/taskcluster.py --target . --branch v0.6.0