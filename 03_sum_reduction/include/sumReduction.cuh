// This program performs sum reduction with an optimization
// removing warp bank conflicts
// By: Nick from CoffeeBeforeArch
#ifndef _SUM_REDUCTION_INC_H_
#define _SUM_REDUCTION_INC_H_

#include "../util/appUtil.h"
#include "../util/AppTicToc.h"

int test_sumReduction1a(const int n);  //only for n=2^k
int test_sumReduction1b(const int n);  //for any integer

int test_sumReduction2a(const int n); //only for n=2^k
int test_sumReduction2b(const int n); //for any integer


int test_sumReduction6(const int n); //for any integer

#endif


