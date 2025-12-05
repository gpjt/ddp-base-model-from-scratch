#!/bin/bash
set -a
curl -LsSf https://astral.sh/uv/install.sh | sh
mkdir -p ~/.local/share/fonts
curl -sL https://github.com/ipython/xkcd-font/raw/master/xkcd-script/font/xkcd-script.ttf -o ~/.local/share/fonts/xkcd-script.ttf
fc-cache -f -v
