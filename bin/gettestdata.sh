#!/usr/bin/env bash

mkpush () {
    mkdir -p data/$1
    pushd data/$1
    $1
    popd
}

cwget () {
    url=$1
    file="${url##*/}"
    if [ ! -f $file ]; then
        wget $url
    fi
}

test1 () {
    cwget https://ia802607.us.archive.org/14/items/artfiction00jamegoog/artfiction00jamegoog_djvu.txt
    cp artfiction00jamegoog_djvu.txt transcript.txt
    cwget http://www.archive.org/download/art_of_fiction_jvw_librivox/art_of_fiction_jvw_librivox_64kb_mp3.zip
    unzip -o art_of_fiction_jvw_librivox_64kb_mp3.zip
    cat *.mp3 >joined.mp3
    ffmpeg -y -i joined.mp3 -ar 16000 -ac 1 audio.wav
}

test2 () {
    cwget http://shakespeare.mit.edu/asyoulikeit/full.html
    html2text full.html >transcript.txt
    cwget http://www.archive.org/download/as_you_like_it_0902_librivox/as_you_like_it_0902_librivox_64kb_mp3.zip
    unzip -o as_you_like_it_0902_librivox_64kb_mp3.zip
    cat *.mp3 >joined.mp3
    ffmpeg -y -i joined.mp3 -ar 16000 -ac 1 audio.wav
}

mkpush test1
mkpush test2
