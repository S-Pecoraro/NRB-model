#!/bin/bash

# Placement dans le dossier
cd /media/nvidia/B27876FA7876BD23/NRB-model/

# Lancement du fetch et du pull

git fetch models
git checkout main
git pull -f origin main

