#ifndef MMSEQS_INDEXTYPES_H
#define MMSEQS_INDEXTYPES_H

#include <cstdint>

#ifdef MMSEQS_INT64_IDS
using DBKeyType = uint64_t;
#else
using DBKeyType = uint32_t;
#endif

#endif
