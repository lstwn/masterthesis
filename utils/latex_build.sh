#!/bin/bash

arg=$1
# without extension
file=$2
row=$3
source_dir=$(dirname $file)
build_dir="$source_dir/build"

echo $source_dir $build_dir

function build() {
  /opt/homebrew/bin/tectonic -X compile $file.tex \
    --untrusted --synctex --keep-logs --keep-intermediates \
    --outdir "$build_dir"
}

function forward_search() {
  /Applications/Skim.app/Contents/SharedSupport/displayline \
    -g -r "$row" "$build_dir/$file.pdf"
}

function clean() {
  rm \
    $file.aux $file.bbl $file.bcf $file.blg $file.log $file.out $file.pdf \
    $file.run.xml $file.synctex.gz
}

if [[ "$1" = "build" ]]; then
  echo "Building"
  build || exit 1
  exit
fi

if [[ "$1" = "build_and_forward_search" ]]; then
  echo "Building and forward searching"
  build && forward_search || exit 1
  exit
fi

if [[ "$1" = "forward_search" ]]; then
  echo "Forward searching"
  forward_search
  exit
fi

if [[ "$1" = "clean" ]]; then
  echo "Cleaning"
  clean || exit 1
  exit
fi

echo "Unknown argument '$@'"
exit 1
