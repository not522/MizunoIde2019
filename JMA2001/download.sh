#!/bin/bash

curl -O https://www.data.jma.go.jp/svd/eqev/data/bulletin/catalog/appendix/trtime/tjma2001.zip
unzip tjma2001.zip
mv tjma2001 travel_time

curl -O https://www.data.jma.go.jp/svd/eqev/data/bulletin/errata/20230315/old/toff2001.zip
unzip toff2001.zip
mv toff2001 take_off_angle

curl -O https://www.data.jma.go.jp/svd/eqev/data/bulletin/catalog/appendix/trtime/vjma2001.zip
unzip vjma2001.zip
mv vjma2001 velocity_structure
