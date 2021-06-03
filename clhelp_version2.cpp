#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <sys/time.h>
#include <time.h>

#include "clhelp_version2.h"

void initialize_ocl(cl_vars_t& cv) {
  std::vector<cl::Platform> all_platforms;
  cv.err = cl::Platform::get(&all_platforms);
  CHK_ERR(cv.err);
  if (all_platforms.empty()) {
    printf("No OpenCL platform found!");
    exit(-1);
  }
  cv.platform = all_platforms[0];

  std::vector<cl::Device> all_devices;
  cv.err = cv.platform.getDevices(CL_DEVICE_TYPE_GPU, &all_devices);
  CHK_ERR(cv.err);
  if (all_devices.empty()) {
    printf("No available OpenCL GPU device found!");
    exit(-1);
  }
  cv.device = all_devices[0];

  cv.context = cl::Context(std::vector<cl::Device>{cv.device}, nullptr, nullptr, nullptr, &cv.err);
  CHK_ERR(cv.err);

  cv.command = cl::CommandQueue(cv.context, cv.device, 0, &cv.err);
  CHK_ERR(cv.err);
}

void uninitialize_ocl(cl_vars_t & clv) {
}

void compile_ocl_program(std::map<std::string, cl::Kernel> &kernels, 
			 cl_vars_t &cv, const std::string &cl_src, 
			 const std::vector<std::string> &knames) {
  cl::Program::Sources sources;
  sources.push_back(cl_src);
  cv.main_program = cl::Program(cv.context, sources, &cv.err);
  CHK_ERR(cv.err);
  
  cv.err = cv.main_program.build({cv.device}, nullptr);
  if (cv.err != CL_SUCCESS) {
    std::cout << "Program build error: ";
    if (cv.main_program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(cv.device) == CL_BUILD_ERROR) {
      std::string log = cv.main_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(cv.device);
      std::cout << log;
    }
    exit(-1);
  }

  for (int i = 0; i < knames.size(); ++i) {
    cl::Kernel kernel = cl::Kernel(cv.main_program, knames[i].c_str(), &cv.err);
    CHK_ERR(cv.err);
    kernels[knames[i]] = kernel;
    cv.kernels.push_back(std::move(kernel));
  }
}

void readFile(std::string& fileName, std::string &out) {
  std::ifstream in(fileName.c_str(), std::ios::in | std::ios::binary);
  if(in) {
    in.seekg(0, std::ios::end);
    out.resize(in.tellg());
    in.seekg(0, std::ios::beg);
    in.read(&out[0], out.size());
    in.close();
  } else {
    std::cout << "Failed to open " << fileName << std::endl;
    exit(-1);
  }
}

double timestamp() {
  struct timeval tv;
  gettimeofday (&tv, 0);
  return tv.tv_sec + 1e-6*tv.tv_usec;
}

void adjustWorkSize(size_t &global, size_t local) {
  if(global % local != 0) {
    global = ((global/local) + 1) * local;  
  }
}

std::string reportOCLError(cl_int err) {
  std::stringstream stream;
  switch (err) {
    case CL_DEVICE_NOT_FOUND:          
      stream << "Device not found.";
      break;
    case CL_DEVICE_NOT_AVAILABLE:           
      stream << "Device not available";
      break;
    case CL_COMPILER_NOT_AVAILABLE:     
      stream << "Compiler not available";
      break;
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:   
      stream << "Memory object allocation failure";
      break;
    case CL_OUT_OF_RESOURCES:       
      stream << "Out of resources";
      break;
    case CL_OUT_OF_HOST_MEMORY:     
      stream << "Out of host memory";
      break;
    case CL_PROFILING_INFO_NOT_AVAILABLE:  
      stream << "Profiling information not available";
      break;
    case CL_MEM_COPY_OVERLAP:        
      stream << "Memory copy overlap";
      break;
    case CL_IMAGE_FORMAT_MISMATCH:   
      stream << "Image format mismatch";
      break;
    case CL_IMAGE_FORMAT_NOT_SUPPORTED:         
      stream << "Image format not supported";    break;
    case CL_BUILD_PROGRAM_FAILURE:     
      stream << "Program build failure";    break;
    case CL_MAP_FAILURE:         
      stream << "Map failure";    break;
    case CL_INVALID_VALUE:
      stream << "Invalid value";    break;
    case CL_INVALID_DEVICE_TYPE:
      stream << "Invalid device type";    break;
    case CL_INVALID_PLATFORM:        
      stream << "Invalid platform";    break;
    case CL_INVALID_DEVICE:     
      stream << "Invalid device";    break;
    case CL_INVALID_CONTEXT:        
      stream << "Invalid context";    break;
    case CL_INVALID_QUEUE_PROPERTIES: 
      stream << "Invalid queue properties";    break;
    case CL_INVALID_COMMAND_QUEUE:          
      stream << "Invalid command queue";    break;
    case CL_INVALID_HOST_PTR:            
      stream << "Invalid host pointer";    break;
    case CL_INVALID_MEM_OBJECT:              
      stream << "Invalid memory object";    break;
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:  
      stream << "Invalid image format descriptor";    break;
    case CL_INVALID_IMAGE_SIZE:           
      stream << "Invalid image size";    break;
    case CL_INVALID_SAMPLER:     
      stream << "Invalid sampler";    break;
    case CL_INVALID_BINARY:                    
      stream << "Invalid binary";    break;
    case CL_INVALID_BUILD_OPTIONS:           
      stream << "Invalid build options";    break;
    case CL_INVALID_PROGRAM:               
      stream << "Invalid program";
    case CL_INVALID_PROGRAM_EXECUTABLE:  
      stream << "Invalid program executable";    break;
    case CL_INVALID_KERNEL_NAME:         
      stream << "Invalid kernel name";    break;
    case CL_INVALID_KERNEL_DEFINITION:      
      stream << "Invalid kernel definition";    break;
    case CL_INVALID_KERNEL:               
      stream << "Invalid kernel";    break;
    case CL_INVALID_ARG_INDEX:           
      stream << "Invalid argument index";    break;
    case CL_INVALID_ARG_VALUE:               
      stream << "Invalid argument value";    break;
    case CL_INVALID_ARG_SIZE:              
      stream << "Invalid argument size";    break;
    case CL_INVALID_KERNEL_ARGS:           
      stream << "Invalid kernel arguments";    break;
    case CL_INVALID_WORK_DIMENSION:       
      stream << "Invalid work dimension";    break;
      break;
    case CL_INVALID_WORK_GROUP_SIZE:          
      stream << "Invalid work group size";    break;
      break;
    case CL_INVALID_WORK_ITEM_SIZE:      
      stream << "Invalid work item size";    break;
      break;
    case CL_INVALID_GLOBAL_OFFSET: 
      stream << "Invalid global offset";    break;
      break;
    case CL_INVALID_EVENT_WAIT_LIST: 
      stream << "Invalid event wait list";    break;
      break;
    case CL_INVALID_EVENT:                
      stream << "Invalid event";    break;
      break;
    case CL_INVALID_OPERATION:       
      stream << "Invalid operation";    break;
      break;
    case CL_INVALID_GL_OBJECT:              
      stream << "Invalid OpenGL object";    break;
      break;
    case CL_INVALID_BUFFER_SIZE:          
      stream << "Invalid buffer size";    break;
      break;
    case CL_INVALID_MIP_LEVEL:             
      stream << "Invalid mip-map level";   
      break;  
    default: 
      stream << "Unknown";
      break;
    }
  return stream.str();
 }
