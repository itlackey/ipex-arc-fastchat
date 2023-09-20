#!/bin/sh
git clone https://github.com/vladmandic/automatic.git /apps/sd-webui
git config core.filemode false
cd /apps/sd-webui
./webui.sh -f --use-ipex --listen
