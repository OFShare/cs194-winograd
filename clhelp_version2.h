#ifndef __CLHELP_H
#define __CLHELP_H
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120

#ifdef __linux__
// #include "CL/cl.h"
#include "./opencl.hpp"
#elif __APPLE__
// #include <OpenCL/opencl.h>
#include "./opencl.hpp"
#else
#error Unsupported OS
#endif

#include <map>
#include <cstdio>
#include <string>
#include <sstream>
#include <list>
#include <vector>
#include <cstdlib>
#include <cstring>

//#define DEBUG 1

typedef struct CLVARS {
  cl_int err;
  cl::Platform platform;
  cl::Device device;
  cl::Context context;
  cl::CommandQueue command;
  cl::Program main_program;
  std::vector<cl::Kernel> kernels;
} cl_vars_t;

void ocl_device_query(cl_vars_t &cv);

std::string reportOCLError(cl_int err);

#define CHK_ERR(err) {\
  if(err != CL_SUCCESS) {\
    printf("Error: %s, File: %s, Line: %d\n", reportOCLError(err).c_str(), __FILE__, __LINE__); \
    exit(-1);\
  }\
}

void initialize_ocl(cl_vars_t& cv);
void uninitialize_ocl(cl_vars_t & clv);
void adjustWorkSize(size_t &global, size_t local);

void compile_ocl_program(std::map<std::string, cl::Kernel> &kernels, 
			 cl_vars_t &cv, const std::string &cl_src, 
			 const std::vector<std::string> &knames);

void readFile(std::string& fileName, std::string &out); 
double timestamp();
#endif
