# harvard family of bibliographic styles
# Copyright 1994 - Peter Williams pwil3058@bigpond.net.au
#
# This work may be distributed and/or modified under the
# conditions of the LaTeX Project Public License, either version 1.3
# of this license or (at your option) any later version.
# The latest version of this license is in
#   http://www.latex-project.org/lppl.txt
# and version 1.3 or later is part of all distributions of LaTeX
# version 2005/12/01 or later.
#
# This work has the LPPL maintenance status `maintained'.
# 
# The Current Maintainers of this work are Peter Williams and Thorsten Schnier.
#
# This work consists of all files listed in manifest.txt.
#
# Licence notice added on behalf of Peter Williams and Thorsten Schnier
# by Clea F. Rees 2009/01/30.

prefix=/opt/TeX
texlib=$(prefix)/lib/texmf
bstdir=$(texlib)/bibtex/bst/harvard
stydir=$(texlib)/tex/latex/local
docdir=$(texlib)/bibtex/doc/harvard
htmldir=/people/archsci/archsci-www/utils/latex2html/styles

BSTS=agsm.bst dcu.bst jmr.bst jphysicsB.bst kluwer.bst nederlands.bst apsr.bst

all:

harvard.ps: harvard.dvi
	dvips harvard

harvard.dvi: harvard.tex harvard.bbl
	latex harvard
	latex harvard

harvard.bbl: harvard.aux harvard.bib
	bibtex harvard

harvard.aux: harvard.tex
	latex harvard

install: harvard.sty harvard.perl $(BSTS)
	mkdir -p $(bstdir)
	mkdir -p $(stydir)
	mkdir -p $(htmldir)
	cp $(BSTS) $(bstdir)
	cp harvard.sty $(stydir)
	cp harvard.perl $(htmldir)

install_doc: harvard.tex harvard.bib
	mkdir -p $(docdir)
	cp doc_Makefile $(docdir)/Makefile
	cp harvard.tex $(docdir)
	cp harvard.bib $(docdir)

clean:
	rm -f harvard.dvi harvard.aux harvard.bbl harvard.log harvard.blg
