include ${SADIR}/Makefile.config

CPPFLAGS := -I. -I${SRCDIR} -I../mkFit-external ${CPPFLAGS}

CMS_DIR := ${SRCDIR}/RecoTracker/MkFitCMS

LIB_CMS := ../libMicCMS.so
MAIN    := ../mkFit
WRMEMF  := ../writeMemoryFile

TGTS := ${LIB_CMS} ${MAIN} ${WRMEMF}

.PHONY: all clean distclean

all: ${TGTS}

SRCS := $(wildcard ${CMS_DIR}/src/*.cc) $(wildcard ${SACMS}/*.cc) ${SACMS}/tkNtuple/WriteMemoryFile.cc
SRCB := $(notdir ${SRCS})
DEPS := $(SRCB:.cc=.d)
OBJS := $(SRCB:.cc=.o)

CMS_OBJS := $(filter-out mkFit.o WriteMemoryFile.o, ${OBJS})

vpath %.cc ${CMS_DIR}/src ${SACMS} ${SACMS}/tkNtuple

ifeq ($(filter clean-local clean distclean, ${MAKECMDGOALS}),)
include ${DEPS}
endif

clean-local:
	-rm -f ${TGTS} *.d *.o *.om *.so
	-rm -rf main.dSYM
	-rm -rf plotting/*.so plotting/*.d plotting/*.pcm

clean: clean-local

distclean: clean-local
	-rm -f *.optrpt
	-rm -f ${TGTS}

${LIB_CMS}: ${CMS_OBJS}
	@mkdir -p $(@D)
	${CXX} ${CXXFLAGS} ${VEC_HOST} ${CMS_OBJS} -shared -o $@ ${LDFLAGS_HOST} ${LDFLAGS}

${MAIN}: ${LIB_CMS} mkFit.o
	${CXX} ${CXXFLAGS} ${VEC_HOST} ${LDFLAGS} mkFit.o -o $@ ${LDFLAGS_HOST} -ltbb -L.. -lMicCore -lMicCMS -Wl,-rpath=.

${WRMEMF}: WriteMemoryFile.o
	${CXX} ${CXXFLAGS} ${LDFLAGS} $^ -o $@ ${LDFLAGS_HOST} -ltbb -L.. -lMicCore -Wl,-rpath=.

${OBJS}: %.o: %.cc %.d
	${CXX} ${CPPFLAGS} ${CXXFLAGS} ${VEC_HOST} -c -o $@ $<

%.d: %.cc
	${MAKEDEPEND} -o $@ $<

echo:
	@echo "CXX=${CXX}"
	@echo SRCS = ${SRCS}
	@echo DEPS = ${DEPS}
	@echo OBJS = ${OBJS}

echo_cc_defs:
	${CXX} -dM -E -mavx2 - < /dev/null

echo-srcs:
	@echo ${SRCS}

echo-flags:
	@echo "CPPFLAGS=${CPPFLAGS}"

echo-tbb:
	@echo "TBB_GCC=${TBB_GCC}, TBB_PREFIX=${TBB_PREFIX}, TBB_ROOT=${TBB_ROOT}"
