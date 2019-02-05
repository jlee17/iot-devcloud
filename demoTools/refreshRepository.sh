#!/bin/bash
rem=`git config --get remote.origin.url`
wd=`basename "$PWD"`
cd ..
rm -rf "$wd"
git clone "$rem" "$wd"
pwd
