/**
 * 
 * Implement "winograd_gpu.cpp" with OpenCL API C++ bindings
 * Implemented by OFShare on 2021-06-03
 * Feel free to use
 * 
 * Usage:
 * g++ winograd_gpu_version2.cpp clhelp_version2.cpp -std=c++11 -framework OpenCL -o winograd_gpu
 * 
 */
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <math.h>
#include <sys/time.h>
#include "clhelp_version2.h"

using namespace std;

/* We are using 3 x 3 filters and an output tile size of 2 x 2. 
 * alpha = m + r - 1 = 4 */
#define m 2
#define r 3
#define alpha 4

/* Returns the next number greater than or equal to global_size that is a 
 * multiple of local_size.*/
cl::size_type gws(int global_size, int local_size) {
  if (global_size % local_size != 0)    
    return (global_size + local_size) / local_size * local_size;
  else
    return global_size;
}

void report_winograd_statistics(int K, int C, int P, double time) {
  int flop = (K * C * (4 * 3 * 5) * 2 +
              C * P * (4 * 4 * 7) * 2 + 
              16 * K * P * (2 * C - 1) + 
              K * P * (2 * 4 * 7) * 2);
  double mflops = flop / (1024.0 * 1024.0 * time);
  cout << "Floating point operations: " << flop << "\n";
  cout << "Time Elapsed: " << time << "\n";
  cout << "MFlop/s: " << mflops << "\n";
}

int main(int argc, char *argv[]) {
  /* Check that program arguments are properly specified. */
  if (argc != 3) {
    cout << "Usage: ./winograd_gpu <input filename> <output filename>\n";
    return 0;
  }

  ifstream file;
  file.open(argv[1]);

  /* Parse problem size. */
  int K, C, H, W;
  file >> K >> C >> H >> W;

  /* Check that sizes are appropriate. */
  bool valid = true;
  if(H % 2 != 0)
    valid = false;
  if( W % 2 != 0)
    valid = false;
  if (!valid) {
    cout << "Please make sure that:\n";
    cout << "H (height of image) is even \n";
    cout << "W (width of image) is even \n";
    file.close();
    return 0;
  }

  int out_H = H - r + 1;
  int out_W = W - r + 1;
  int num_h_tiles = ceil(out_H/m);
  int num_w_tiles = ceil(out_W/m);
  int P = num_h_tiles * num_w_tiles;

  /* Read in filters. */
  float *filters = new float[K*C*r*r];
  for (int k = 0; k < K; k++) {
    for (int c = 0; c < C; c++) {
      for (int i = 0; i < r; i++) {
        for (int j = 0; j < r; j++) {
          file >> filters[k*(C*r*r) + c*(r*r) + i*r + j];
        }
      }
    }
  }

  /* Read in image. */
  float *data = new float[C*H*W];
  for (int c = 0; c < C; c++) {
    for (int i = 0; i < H; i++) {
      for (int j = 0; j < W; j++) {
        file >> data[c*(H*W) + i*H + j];
      }
    }
  }
  file.close();

  /* Filter transform matrix. */
  float G[12] = {1.0, 0.0, 0.0,
                 0.5, 0.5, 0.5,
                 0.5, -0.5, 0.5,
                 0.0, 0.0, 1.0};

  /* Data transform matrix. */
  float B[16] = {1.0, 0.0, 0.0, 0.0,
                 0.0, 1.0, -1.0, 1.0,
                 -1.0, 1.0, 1.0, 0.0,
                 0.0, 0.0, 0.0, -1.0};

  /* Inverse transform matrix (to transform the output after it is computed).*/
  float A[8] = {1.0, 0.0,
                1.0, 1.0,
                1.0, -1.0,
                0.0, -1.0};
  
  /* Array to hold the output. */
  float *Y = new float[K*out_H*out_W];

  /* Provide names of the OpenCL kernels
   * and cl file that they're kept in. */
  std::string arraycompact_kernel_file = std::string("winograd.cl");

  std::vector<std::string> kernel_names;
  std::string filter_transform_name_str = std::string("filter_transform");
  std::string data_transform_name_str = std::string("data_transform");
  std::string calc_M_name_str = std::string("calc_M");
  std::string calc_Y_name_str = std::string("calc_Y");

  kernel_names.push_back(filter_transform_name_str);
  kernel_names.push_back(data_transform_name_str);
  kernel_names.push_back(calc_M_name_str);
  kernel_names.push_back(calc_Y_name_str);
 
   /* OpenCL setup. */
  std::string kernel_source_str;
  std::map<std::string, cl::Kernel> kernel_map;
  readFile(arraycompact_kernel_file, kernel_source_str);

  /* Intialize OpenCL runtime. */
  cl_vars_t cv;
  initialize_ocl(cv);

  /* Compile kernels. */
  compile_ocl_program(kernel_map, cv, kernel_source_str, kernel_names);
    
  // /* Create buffers on GPU. */
  // cl_mem g_filters, g_data, g_G, g_B, g_A, g_U, g_V, g_M, g_Y;
  cl::Buffer g_filters(cv.context, CL_MEM_READ_WRITE, sizeof(float)*K*C*3*3);
  cl::Buffer g_data(cv.context, CL_MEM_READ_WRITE, sizeof(float)*C*H*W);
  cl::Buffer g_G(cv.context, CL_MEM_READ_ONLY, sizeof(float)*alpha*r);
  cl::Buffer g_B(cv.context, CL_MEM_READ_ONLY, sizeof(float)*alpha*alpha);
  cl::Buffer g_A(cv.context, CL_MEM_READ_ONLY, sizeof(float)*alpha*m,NULL);
  
  // /* Will hold output of the filter transform. */
  cl::Buffer g_U(cv.context, CL_MEM_READ_WRITE, sizeof(float)*K*C*alpha*alpha);
  // /* Will hold output of the data transform. */
  cl::Buffer g_V(cv.context, CL_MEM_READ_WRITE, sizeof(float)*C*P*alpha*alpha);
  // /* Will hold the pre-transformed output. */
  cl::Buffer g_M(cv.context, CL_MEM_READ_WRITE, sizeof(float)*K*P*alpha*alpha);
  // /* Will hold the final (transformed) output. */
  cl::Buffer g_Y(cv.context, CL_MEM_READ_WRITE, sizeof(float)*K*out_H*out_W,NULL);

  // /* Copy data into buffers. */
  cv.command.enqueueWriteBuffer(g_filters, CL_TRUE, 0, sizeof(float)*K*C*r*r, filters);
  cv.command.enqueueWriteBuffer(g_data, CL_TRUE, 0, sizeof(float)*C*H*W, data);
  cv.command.enqueueWriteBuffer(g_G, CL_TRUE, 0, sizeof(float)*alpha*r, G);
  cv.command.enqueueWriteBuffer(g_B, CL_TRUE, 0, sizeof(float)*alpha*alpha, B);
  cv.command.enqueueWriteBuffer(g_A, CL_TRUE, 0, sizeof(float)*alpha*m, A);

  /* Compute global and local work sizes for the following: */

  /* Filter transform, which calculates U. */
  cl::NDRange global_work_size_U = {gws(K,8), gws(C,4)};
  cl::NDRange local_work_size_U = {8, 4};

  /* Data transform, which calculates V. */
  cl::NDRange global_work_size_V = {gws(C, 4), gws(num_h_tiles, 4), gws(num_w_tiles, 4)};
  cl::NDRange local_work_size_V = {4, 4, 4};

  /* Calculating M. */
  cl::size_type local_M = 8;
  cl::NDRange global_work_size_M = {gws(K, local_M), gws(P, local_M)};
  cl::NDRange local_work_size_M = {local_M, local_M};

  /* Calculating Y. */
  cl::NDRange global_work_size_Y = {gws(K, 2), gws(num_h_tiles, 8), gws(num_w_tiles, 8)};
  cl::NDRange local_work_size_Y = {2, 8, 8};

  /* Get the compiled kernels. */
  cl::Kernel &filter_transform_kern = kernel_map[filter_transform_name_str];
  cl::Kernel &data_transform_kern = kernel_map[data_transform_name_str];
  cl::Kernel &calc_M_kern = kernel_map[calc_M_name_str];
  cl::Kernel &calc_Y_kern = kernel_map[calc_Y_name_str];

  // /* Set the arguments for each kernel. */
  filter_transform_kern.setArg(0, g_filters);
  filter_transform_kern.setArg(1, g_G);
  filter_transform_kern.setArg(2, g_U);
  filter_transform_kern.setArg(3, K);
  filter_transform_kern.setArg(4, C);

  data_transform_kern.setArg(0, g_data);
  data_transform_kern.setArg(1, g_B);
  data_transform_kern.setArg(2, g_V);
  data_transform_kern.setArg(3, C);
  data_transform_kern.setArg(4, P);
  data_transform_kern.setArg(5, H);
  data_transform_kern.setArg(6, W);
  data_transform_kern.setArg(7, num_h_tiles);
  data_transform_kern.setArg(8, num_w_tiles);

  calc_M_kern.setArg(0, g_U);
  calc_M_kern.setArg(1, g_V);
  calc_M_kern.setArg(2, g_M);
  calc_M_kern.setArg(3, K);
  calc_M_kern.setArg(4, P);
  calc_M_kern.setArg(5, C);

  calc_Y_kern.setArg(0, g_M);
  calc_Y_kern.setArg(1, g_A);
  calc_Y_kern.setArg(2, g_Y);
  calc_Y_kern.setArg(3, out_H);
  calc_Y_kern.setArg(4, out_W);
  calc_Y_kern.setArg(5, K);
  calc_Y_kern.setArg(6, P);
  calc_Y_kern.setArg(7, num_h_tiles);
  calc_Y_kern.setArg(8, num_w_tiles);

  /* Start recording time for benchmarking. */
  double time = timestamp();

  // /* Compute filter transform. */
  cv.command.enqueueNDRangeKernel(filter_transform_kern, cl::NullRange, global_work_size_U, local_work_size_U, 0, 0);
  // /* Compute data transform. */
  cv.command.enqueueNDRangeKernel(data_transform_kern, cl::NullRange, global_work_size_V, local_work_size_V, 0, 0);
  // /* Compute the pre-transformed output. */
  cv.command.enqueueNDRangeKernel(calc_M_kern, cl::NullRange, global_work_size_M, local_work_size_M, 0, 0);
  // /* Transform the output. */
  cv.command.enqueueNDRangeKernel(calc_Y_kern, cl::NullRange, global_work_size_Y, local_work_size_Y, 0, 0);
  
  cv.command.finish();
  time = timestamp() - time;

  /* Report timing and Mflop/s */
  report_winograd_statistics(K, C, P, time);

  cv.command.enqueueReadBuffer(g_Y, CL_TRUE, 0, sizeof(float)*K*out_H*out_W, Y);
  /* Write output Y to the specified file. */
  ofstream fileout;
  fileout.open(argv[2], ofstream::out | ofstream::trunc);
  fileout << K << " " << C << " " << H - 2 << " " << W - 2 << endl;
  for(int k = 0; k < K; k++) {
    fileout << "\n";
    for(int i = 0; i < out_H; i++) {
      for(int j = 0; j < out_W; j++) {
        int index = k*(out_H*out_W) + i*out_W + j;
        fileout << "   " << std::fixed << std::setw(5) << std::setprecision(4) << Y[index];
      }
      fileout << "\n";
    }
  }
  fileout.close();
  uninitialize_ocl(cv);
  delete[] filters;
  delete[] data;
  delete[] Y;

  return 0;
}
