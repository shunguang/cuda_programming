@echo off

REM define environment variables, do not change the variable name, just it's value!
REM set GTEST_INC=C:\pkg\googletest\vs2019-install-x64\include
REM set GTEST_LIB=C:\pkg\googletest\vs2019-install-x64\lib

set CUDA_INC="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\include"
set CUDA_LIB="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\lib\x64"

REM set CV_INC=C:\pkg\opencv\opencv-4.1.1\build-vs2019\install\include
REM set CV_LIB=C:\pkg\opencv\opencv-4.1.1\build-vs2019\install\x64\vc16\lib

REM set BOOST_INC=C:\pkg\boost\boost_1_71_0
REM set BOOST_LIB=C:\pkg\boost\boost_1_71_0\lib64-msvc-14.2

set APP_CPP=C:\Users\wus1\Projects\2023\cuda_programming
set APP_BUILD_INT=C:\Users\wus1\Projects\2023\cuda_programming\build-vs2019-x64-cuda\int
set APP_BUILD_BIN=C:\Users\wus1\Projects\2023\cuda_programming\build-vs2019-x64-cuda\bin

"C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\Common7\IDE\devenv.exe" %APP_CPP%\vs2019\app_vs2019.sln

REM ---eof---/Users/wus1/Projects/2020/p803/software/payload-cpu/pyxis-analysis/vs2017/pyxisAnalysis.props
