#ifndef __APP_DEFS_H__
#define __APP_DEFS_H__
#include <random>
#include <chrono>
#include <limits>

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>  
#include <math.h>
#include <inttypes.h>
#include <assert.h>
#include <time.h>

//#include <cinttypes>
#include <stdexcept>
#include <cstddef>  
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <deque>
#include <locale>
#include <algorithm>
#include <atomic>
#include <fstream>


//put linux headers here
#ifdef  __linux__
#include <sys/resource.h>
#include <unistd.h>
#endif

#define TS_SINCE_EPOCH_MS     (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count())
#define TS_SINCE_EPOCH_US     (std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count())
#define TS_SINCE_EPOCH_NS     (std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count())


#define APP_EPS_64F		(1e-15)
#define APP_EPS_32F 	(1e-7)

#define APP_REALMIN_32F (1e-38f)
#define APP_REALMAX_32F (1e+38f)
#define APP_REALMIN_64F (1e-308)
#define APP_REALMAX_64F (1e+308)

#define APP_MAX_UINT16 (0xFFFF)
#define APP_MAX_UINT32 (0xFFFFFFFF)
#define APP_MAX_UINT64 (0xFFFFFFFFFFFFFFFF)
#define APP_NAN_UINT16 (0xFFFF)
#define APP_NAN_UINT32 (0xFFFFFFFF)
#define APP_NAN_UINT64 (0xFFFFFFFFFFFFFFFF)

#define APP_HALF_PI       (1.57079632679490)
#define APP_PI            (3.14159265358979)
#define APP_TWO_PI        (6.28318530717959)
#define APP_D2R(x)        (0.01745329251994*(x))
#define APP_R2D(x)        (57.29577951308232*(x))

#define APP_ROUND(x)	( (int) floor( (x) + 0.500 ) )
#define APP_NAN			( sqrt(-1.0) )
#define APP_ISNAN(x)	( (x) != (x) )

#define APP_MAX(a,b)	( (a) > (b) ? (a) : (b) )
#define APP_MIN(a,b)	( (a) > (b) ? (b) : (a) )
#define APP_INT_RAND_IN_RANGE(i1,i2) ( (i1) + rand() % ((i2) + 1 - (i1)) )
#define APP_IN_RANGE_CLOSE(x, x1, x2) ( ((x) >= (x1)) && ((x) <= (x2))  )
#define APP_IN_RANGE_OPEN(x, x1, x2)  ( ((x) >  (x1)) && ((x) <  (x2))  )

#define APP_USED_TIME_MS(t0)  ( 1000 * (clock() - (t0)) / CLOCKS_PER_SEC )

#define USLEEP_1_SEC		(1000000)
#define USLEEP_1_MILSEC	(1000)


#ifdef __GNUC__
#define  sscanf_s  sscanf
#define  swscanf_s swscanf
   #define  GNU_PACK  __attribute__ ((packed))
#else
   #define  GNU_PACK
#endif

#define APP_DISK_GB (1000000000)
#define APP_DISK_MB (1000000)

#define APP_FRM_CNT			      uint64_t
#define APP_TIME_MS           int64_t    //milli second
#define APP_TIME_US           uint64_t   //micro second
#define APP_TIME_US2MS( t_us )  ( (int64_t)(t_us/1000) )
#define APP_TIME_MS2US( t_ms )  ( ((uint64_t)t_ms) * 1000 )
#endif
