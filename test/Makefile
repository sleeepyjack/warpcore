INCDIRS := include ../include ../ext/cudahelpers ../ext/packed_types/include

CC := g++
NVCC := nvcc
STD := c++14
OPT := 3
CCFLAGS := -O$(OPT) -std=$(STD) -Wall -Wextra -fopenmp
XCCFLAGS := $(addprefix -Xcompiler ,$(CCFLAGS))
NVCCGENCODE = -gencode arch=compute_70,code=sm_70 \
			  -gencode arch=compute_61,code=sm_61 \
              -gencode arch=compute_70,code=sm_70
NVCCFLAGS := -O$(OPT) -std=$(STD) -ccbin $(CC) $(XCCFLAGS) $(NVCCGENCODE) --expt-extended-lambda

SRCS := $(wildcard src/*.cu src/*.cpp)
SRCNAMES := $(basename $(notdir $(SRCS)))
OBJS := $(SRCNAMES:%=build/%.o)
INCS := $(foreach dir, $(INCDIRS), $(wildcard $(dir)/*.cuh $(dir)/*.h $(dir)/*.hpp))
INCPARAMS := $(addprefix -I, $(INCDIRS))

#.PHONY: all
#all: ; $(info $$var is [${var}])$(OBJS)

all: bin/tester

info:
	@echo " INFO:"
	@echo " CC = $(CC)"
	@echo " CCFLAGS = $(CCFLAGS)"
	@echo " NVCC = $(NVCC)"
	@echo " NVCCFLAGS = $(NVCCFLAGS)"
	@echo " SOURCES = $(SRCS)"
	@echo " OBJECTS = $(OBJS)"
	@echo " INCLUDES = $(INCS)"

bin/tester: $(OBJS) | bin
	$(NVCC) $(NVCCFLAGS) $(INCPARAMS) $(OBJS) -o $@

build/catch_main.o: src/catch_main.cu include/catch.hpp | build
	$(NVCC) $(NVCCFLAGS) -Iinclude -c $< -o $@

build/%.o: src/%.cu $(INCS) | build
	$(NVCC) $(NVCCFLAGS) $(INCPARAMS) -c $< -o $@

build/%.o: src/%.cpp $(INCS) | build
	$(CC) $(CCFLAGS) $(INCPARAMS)  -c $< -o $@

debug: OPT := 0
debug: CCFLAGS := -O$(OPT) -std=$(STD) -Wall -Wextra -fopenmp
debug: XCCFLAGS := $(addprefix -Xcompiler ,$(CCFLAGS))
debug: NVCCFLAGS := -O$(OPT) -std=$(STD) -ccbin $(CC) $(XCCFLAGS) $(NVCCGENCODE) --expt-extended-lambda -g -G -lineinfo -rdynamic -Xptxas -v -UNDEBUG -DDEBUG
debug: all

profile: NVCCFLAGS += -lineinfo -g -Xptxas -v -UNDEBUG
profile: all

clean:
	$(RM) -r build bin

build bin:
	@mkdir -p $@

.PHONY: clean all build bin
