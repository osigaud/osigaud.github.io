################################################################################
# SVG files cannot be directly included in LaTeX. But pdflatex does accept PDF
# and PNG files. Therefore, this makefile converts SVG into PDF and PNG.
#
# Why both? Because very large SVG files (>1Mb, e.g. a graph with many data
# points) might lead to large PDF files. In this case, the PNG is often smaller
# (at the cost of being a rasterized bitmap image instead of a vector graphic).
#
# In the LaTeX file, the PDF/PNG are included as follows:
#   \includegraphics{filename-svg}      (PDF file generated from SVG)
#   \includegraphics{filename-svg-png}  (PNG file generated from SVG)
#   \includegraphics{filename-png}      (PNG file)
#   \includegraphics{filename-pdf}      (PDF file)


################################################################################
# RELEVANT FILES

# Get list of all SVG files
SOURCE_SVG := $(shell find -name '*.svg')
TARGET_SVG_PDF = $(SOURCE_SVG:.svg=-svg.pdf)

# Add output directory
BIN_DIR = ../images
TARGET_SVG_PDF := $(addprefix ${BIN_DIR}/,${TARGET_SVG_PDF})

################################################################################
# TARGETS

all: ${BIN_DIR} ${TARGET_SVG_PDF}

clean:
	rm ${TARGET_SVG_PDF}

${BIN_DIR}:
	mkdir ${BIN_DIR}

copy:
	cp ${BIN_DIR}/*.pdf ../slides/images/

################################################################################
# PATTERN RULES

# Command to convert SVG file into a PNG/PDF file
#   --without-gui         : Don't start inkscape GUI
#   --export-area-drawing : Bounding box tight around the actual drawing
SVG_EXPORT=inkscape --without-gui --export-area-drawing 

#   --export-text-to-path : Convert text to paths before exporting
${BIN_DIR}/%-svg.pdf:%.svg
	@echo "$< -> $@"
	@${SVG_EXPORT} --export-text-to-path --file=$<  --export-pdf=$@ 2> /dev/null > /dev/null

