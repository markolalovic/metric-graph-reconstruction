#!/bin/bash

NAME="sigma-tube"
pdflatex --shell-escape -halt-on-error "${NAME}.tex"
rm *.aux
rm *.log
rm *.pgf-plot.gnuplot
rm *.pgf-plot.table
pdfcrop "${NAME}.pdf" "${NAME}.pdf"
