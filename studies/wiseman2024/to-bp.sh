#!/bin/bash

#make a clean temp folder
mkdir /tmp/Abil
#copy folder but exclude git, docs and dis
rsync -av --progress /home/mv23682/Documents/Abil/ /tmp/Abil/ --exclude .git --exclude dist --exclude docs --exclude /studies/wiseman2024/ModelOutput --exclude /studies/wiseman2024/data --exclude /studies/wiseman2024/env_data_processing --exclude /studies/lim2024 --exclude /studies/devries2024 --exclude tests --exclude examples
#copy temp folder to bp
scp -r /tmp/Abil/ mv23682@bp1-login.acrc.bris.ac.uk:/user/work/mv23682/
#if copy ok, remove temp folder
rm -rf /tmp/Abil
