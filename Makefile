.RECIPEPREFIX := >

NVCC := nvcc
ARCH ?= sm_89
NVCCFLAGS := -arch=$(ARCH) -O3 -Iinclude -lineinfo

LADDER_SOURCES := src/main.cu src/reference.cu $(wildcard src/sgemm_v*.cu)
LADDER_OBJECTS := $(patsubst src/%.cu,build/%.o,$(LADDER_SOURCES))
HEADERS := $(wildcard include/*.h)

all: sgemm bank_conflicts tune_v4 tune_v5

sgemm: $(LADDER_OBJECTS)
>$(NVCC) $(NVCCFLAGS) -o $@ $^

bank_conflicts: build/bank_conflicts.o
>$(NVCC) $(NVCCFLAGS) -o $@ $^

tune_v4: build/tune_v4.o build/reference.o
>$(NVCC) $(NVCCFLAGS) -o $@ $^

tune_v5: build/tune_v5.o build/reference.o
>$(NVCC) $(NVCCFLAGS) -o $@ $^

build/%.o: src/%.cu $(HEADERS)
>@mkdir -p build
>$(NVCC) $(NVCCFLAGS) -c -o $@ $<

.PHONY: all