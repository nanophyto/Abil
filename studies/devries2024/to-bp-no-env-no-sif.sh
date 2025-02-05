#!/bin/bash

#make a clean temp folder
mkdir /tmp/Abil
#copy folder but exclude git, docs and dis
rsync -av --progress /home/phyto-2/Abil/ /tmp/Abil/ --exclude .git --exclude dist \
--exclude docs --exclude ModelOutput --exclude examples --exclude tests --exclude studies/devries2024/data --exclude '*.sif'
#copy temp folder to bp
scp -r /tmp/Abil/ ba18321@bp1-login.acrc.bris.ac.uk:/user/work/ba18321/
#if copy ok, remove temp folder
rm -rf /tmp/Abil
