#ifndef _APP_MEAN_STD_H_
#define _APP_MEAN_STD_H_

#include "AppDefs.h"

namespace cge
{
    template<typename T>
    class AppMeanStd
    {
    public:
        AppMeanStd(const std::string &unit_ = "")
            : unit(unit_), v()
        {
        }

        void reset()
        {
            v.clear();
        }

        size_t calMeanStd()
        {
            if (v.empty()){
                mu = APP_NAN;
                std = APP_NAN;
                return 0;
            }

            size_t n = v.size();
            double s = 0;
            double sq = 0;
            for( const T e : v){
                s += (double)e;
                sq += (double)e * (double)e;
            }
            mu = (T)(s/n);
            std = (T)sqrt(sq / n - mu * mu);  // sqrt(variance)
            return n;
        }

        void addSample(const T x)
        {
            v.push_back(x);
        }

    public:
        std::string unit{""};
        std::vector<T> v{};
        T mu{0};
        T std{0};
    };
}

#endif
