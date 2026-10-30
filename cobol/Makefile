# COBOL walker kernel — GnuCOBOL 3.2 (apt gnucobol3) on ubuntu-latest and dev
# sandboxes. Builds rivgraph/lib/rivgraph_cobol.so, the shared object the
# Python loader (rivgraph/_cobol.py) binds via ctypes.
#
#   make            build (or rebuild) the backend
#   make clean      remove generated C and the shared object

COB_SRC := walker.cob
SHIM_SRC := shim.c
GEN_C := $(COB_SRC:.cob=.c)
LIB := ../rivgraph/lib/rivgraph_cobol.so

.PHONY: all clean
all: $(LIB)

$(LIB): $(COB_SRC) $(SHIM_SRC)
	cobc -C -O2 -free $(COB_SRC)
	mkdir -p $(dir $(LIB))
	gcc -shared -fPIC -O2 -o $@ $(SHIM_SRC) $(GEN_C) \
		$$(cob-config --cflags) $$(cob-config --libs)

clean:
	rm -f $(LIB) $(GEN_C) $(COB_SRC:.cob=.so.h) cobol.c.h
