all:
	rubber --pdf ./main.tex
clean:
	rubber --clean main.tex
	rm -f main.pdf
