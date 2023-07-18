LATEXMK_FLAGS := "-jobname=cad1_23_hanssian"

texpdf:
	latexmk $(tex_fname) ${LATEXMK_FLAGS} 

paper-dev: LATEXMK_FLAGS += -pvc
paper-dev: paper
paper: tex_fname="template.tex"
paper: texpdf

all: paper

clean:
	latexmk -C ${LATEXMK_FLAGS}
	-rm *.lol
	-rm *.run.xml
	-rm -r _minted*
	-rm *.nav
	-rm *.pyg
	-rm *.snm
	-rm *.vrb
	-rm *.fls
	-rm *.toc
	-rm *.log
