.PHONY: pdf clean

pdf:
	env LC_ALL=C LANG=C latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

clean:
	env LC_ALL=C LANG=C latexmk -c
