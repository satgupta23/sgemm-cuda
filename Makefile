NVCC     := nvcc
ARCH     ?= sm_89
NVCCFLAGS := -arch=$(ARCH) -O3 -Iinclude -lineinfo

SOURCES := $(wildcard src/*.cu)
OBJECTS := $(patsubst src/%.cu,build/%.o,$(SOURCES))
HEADERS := $(wildcard include/*.h)

sgemm: $(OBJECTS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

build/%.o: src/%.cu $(HEADERS)
	@mkdir -p build
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

.PHONY: all
all: sgemm
