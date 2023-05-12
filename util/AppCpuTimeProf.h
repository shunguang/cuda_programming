#ifndef _APP_CPU_TIME_PROF_H_
#define _APP_CPU_TIME_PROF_H_

#include "AppDefs.h"
#include "AppMeanStd.h"

#define APP_MONITOR_SYS_CPU_USAGE 0

//all time units are macro second (us)
namespace cge
{
    class AppCpuTimeProf
    {
    public:
        AppCpuTimeProf(const size_t thdCntTpPrintf=100, const std::string myName="" ) 
        : wall("us")
#if APP_MONITOR_SYS_CPU_USAGE
        , ucpu("us"), scpu("us") 
#endif  
        , thdCntToPrintOut(thdCntTpPrintf)
        , name(myName)      
        {
            reset();
        }

        void setThdCntToPrintOut( const size_t nFrms ){
            thdCntToPrintOut = nFrms;
        }

        void reset()
        {
            cnt=0;
            wall.reset();
#if APP_MONITOR_SYS_CPU_USAGE
            ucpu.reset();
            scpu.reset();
#endif            
        }   

        size_t calMeanStd(std::string & out, const std::string &msg = "")
        {
            char buf[256];
            size_t nSmps = wall.calMeanStd();

#if APP_MONITOR_SYS_CPU_USAGE
            nSmps = ucpu.calMeanStd();
            nSmps = scpu.calMeanStd();
            snprintf(buf, 256, "%s nSmps=%zu, wall(mu=%ld, std=%ld) (us), usrCpu(mu=%ld, std=%ld) (us), sysCpu(mu=%ld, std=%ld) (us)",
                                    msg.c_str(), nSmps, wall.mu, wall.std, ucpu.mu, ucpu.std, scpu.mu, scpu.std);
#else                                    
            snprintf(buf, 256, "%s nSmps=%zu, wall(mu=%ld, std=%ld) (us)", msg.c_str(), nSmps, wall.mu, wall.std);
#endif

            out = std::string(buf);

            return nSmps;
        }

        void addSample(const int64_t dt_wall
#if APP_MONITOR_SYS_CPU_USAGE
        , const int64_t dt_ucpu, const int64_t dt_scpu
#endif        
        )
        {
            cnt++;
            wall.addSample(dt_wall);

#if APP_MONITOR_SYS_CPU_USAGE
            ucpu.addSample(dt_ucpu);
            scpu.addSample(dt_scpu);
#endif            
            if( 0 == cnt % thdCntToPrintOut)
            {
                std::string result;
                calMeanStd(result, "perFrameStatistics");
#if APP_MONITOR_SYS_CPU_USAGE
                printf("%s: current(dt_wall=%ld,dt_ucpu=%ld,dt_scpu=%ld), %s\n", name.c_str(), dt_wall, dt_ucpu, dt_scpu, result.c_str());
#else
                printf("%s: current(dt_wall=%ld), %s\n", name.c_str(), dt_wall, result.c_str());
#endif                
            }
        }

    public:
        AppMeanStd<int64_t> wall;   //wall time, micro sec (us)

#if APP_MONITOR_SYS_CPU_USAGE
        AppMeanStd<int64_t> ucpu;    //usr time, micro sec (us)
        AppMeanStd<int64_t> scpu;    //sys time, micro sec (us)
#endif

    private:
        size_t thdCntToPrintOut{100};
        std::string name{"profile"};
        size_t cnt{0};
    };
}

#endif
