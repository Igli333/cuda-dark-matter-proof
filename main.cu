#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>

#define NUM_BINS 720 // Bins of width 0.25 on a range of 0-180 degrees.
#define NUM_GALAXIES 100000

typedef struct {
    float right_ascension;
    float declination;
} Galaxy;

__device__ float to_degrees(float const radians) {
    return (radians * 180.0f) / 3.141592654f;
}

__global__ void calculate_angles(unsigned int *hist, Galaxy *first_set, Galaxy *second_set, bool are_same) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;

    // Using a histogram in shared memory to save values in block-level.
    __shared__ unsigned int shared_hist[NUM_BINS];

    int tx = threadIdx.x;
    if (tx < NUM_BINS) {
        shared_hist[tx] = 0;
        tx += blockDim.x;
    }

    __syncthreads();

    if (i < NUM_GALAXIES){
        for (int j = 0; j < NUM_GALAXIES; j++){
            // Skip calculation for pairs of the same galaxy.
            if (i == j && are_same) {
                atomicAdd(&shared_hist[0], 1);
                continue;
            }

            float first_ascension = first_set[i].right_ascension;
            float first_declination = first_set[i].declination;
            float second_ascension = second_set[j].right_ascension;
            float second_declination = second_set[j].declination;

            float angle = sinf(first_declination) * sinf(second_declination)
                                    + cosf(first_declination) * cosf(second_declination)
                                    * cosf(first_ascension - second_ascension);
            
            angle = fmax(-1.0f, fmin(1.0f, angle));
            angle = acosf(angle);

            int k = to_degrees(angle) * 4.0f;
            atomicAdd(&shared_hist[k], 1);
        }
    }

    __syncthreads();

    int tx_copy = threadIdx.x;

    // Saving all calculated values in the histogram.
    if (tx_copy < NUM_BINS) {
        atomicAdd(&hist[tx_copy], shared_hist[tx_copy]);
        tx_copy += blockDim.x;
    }
}

__global__ void calculate_omegas(float *omega, unsigned int *DD, unsigned int *DR, unsigned int *RR) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(i < NUM_BINS){
        float dd = DD[i];
        float dr = DR[i];
        float rr = RR[i];

        // In case any of the bins happen to be 0, to avoid division by 0.
        if (rr == 0) {
            omega[i] = 0;
            return;
        }
        
        omega[i] = (dd - (2 * dr) + rr) / rr;
    }
}

float to_radians(float const arc_minutes) {
    return (arc_minutes * M_PI) / 10800;
}

Galaxy *read_galaxy_information(const char *sourcefile) {
    FILE *f = fopen(sourcefile, "rb");

    if (f == NULL) {
        printf("Could not open file %s\n", sourcefile);
        exit(1);
    }

    // First value of the file is its size.
    int *size;
    if (fscanf(f, "%d", &size) != 1) {
        fprintf(stderr, "Error reading count from file\n");
        fclose(f);
        return NULL;
    }

    // Initializing the array of structs.
    Galaxy *galaxy_list = (Galaxy*) malloc(NUM_GALAXIES * sizeof(Galaxy));
    if (galaxy_list == NULL) {
        perror("Memory allocation for galaxies failed");
        fclose(f);
        return NULL;
    }

    // Reading the file into the array.
    for (int i = 0; i < NUM_GALAXIES; i++) {
        float right_ascension_am, declination_am;
        if (fscanf(f, "%f %f", &right_ascension_am, &declination_am, f) != 2) {
            fprintf(stderr, "Error reading right ascension and declination from file\n");
            free(galaxy_list);
            fclose(f);
            return NULL;
        }

        galaxy_list[i].right_ascension = to_radians(right_ascension_am);
        galaxy_list[i].declination = to_radians(declination_am);
    }

    fclose(f);
    return galaxy_list;
}

void print_histogram_file(unsigned int *hist, const char *output_file_name){
    FILE *out_file = fopen(output_file_name, "w");

    if (out_file == NULL) {  
        printf("Error! Could not open file\n");
        return;
    }

    for (int i = 0; i < NUM_BINS; i++) {
        fprintf(out_file, "%u\n", hist[i]);
    }

    fclose(out_file);
}

void print_histogram(unsigned int *hist) {
    for (int i = 0; i < NUM_BINS; i++) {
        printf("%-15u ", hist[i]);
    }
}

void print_omegas(float *omega){
    for (int i = 0; i < NUM_BINS; i++) {
        printf("%-15f ", omega[i]);
    }
}

void print_total_angles_of_histogram(unsigned int *hist, const char *name){
    long int sum = 0L;
    for (int i = 0; i < NUM_BINS; i++){
        sum += (long) hist[i];
    }

    printf("Total angles in histogram %s are %ld.\n", name, sum);
}

int main(void) {
    clock_t begin = clock();

    int getDevice(int device_number);
    if (getDevice(0) != 0) {
        return(-1);
    }

    // Declaring all arrays and histograms for CPU and GPU.
    Galaxy *real_galaxy_data, *rand_galaxy_data, *d_real_galaxy_data, *d_rand_galaxy_data; 
    unsigned int *DD, *DR, *RR, *d_DD, *d_DR, *d_RR;
    float *omega, *d_omega;

    // Declaring the sizes of the declared arrays for memory allocation.
    int size_of_initial_vectors = NUM_GALAXIES * sizeof(Galaxy);
    int size_hist = NUM_BINS * sizeof(unsigned int);
    int size_omega = NUM_BINS * sizeof(float);

    // Reading data from the files
    real_galaxy_data = read_galaxy_information("data_100k_arcmin.dat");
    if (real_galaxy_data == NULL) {
        return -1;
    }

    rand_galaxy_data = read_galaxy_information("rand_100k_arcmin.dat");
    if (rand_galaxy_data == NULL) {
        return -1;
    }

    // Allocating all the GPU data arrays and copying the values into them from the CPU arrays.
    cudaMalloc(&d_real_galaxy_data, size_of_initial_vectors);
    cudaMalloc(&d_rand_galaxy_data, size_of_initial_vectors);
    cudaMemcpy(d_real_galaxy_data, real_galaxy_data, size_of_initial_vectors, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rand_galaxy_data, rand_galaxy_data, size_of_initial_vectors, cudaMemcpyHostToDevice);

    // Allocating memory in CPU for the histograms with initial values set as 0.
    DD = (unsigned int *) calloc(NUM_BINS, sizeof(unsigned int));
    DR = (unsigned int *) calloc(NUM_BINS, sizeof(unsigned int));
    RR = (unsigned int *) calloc(NUM_BINS, sizeof(unsigned int));
    omega = (float *) calloc(NUM_BINS, sizeof(float));

    // Allocating the histograms in GPU.
    cudaMalloc(&d_DD, size_hist);
    cudaMalloc(&d_DR, size_hist);
    cudaMalloc(&d_RR, size_hist);
    cudaMalloc(&d_omega, size_omega);

    // Setting number of threads for block and number of blocks.
    int number_of_threads = 512;
    int number_of_blocks = ceil((NUM_GALAXIES + number_of_threads) / number_of_threads);
       
    calculate_angles<<<number_of_blocks,number_of_threads>>>(d_DD, d_real_galaxy_data, d_real_galaxy_data, true);
    cudaDeviceSynchronize();
    calculate_angles<<<number_of_blocks,number_of_threads>>>(d_DR, d_real_galaxy_data, d_rand_galaxy_data, false);
    cudaDeviceSynchronize();
    calculate_angles<<<number_of_blocks,number_of_threads>>>(d_RR, d_rand_galaxy_data, d_rand_galaxy_data, true);
    cudaDeviceSynchronize();

    // Setting number of blocks for the omega calculation.
    int number_of_block_omega = (NUM_BINS + number_of_threads) / number_of_threads;
   
    calculate_omegas<<<number_of_block_omega,number_of_threads>>>(d_omega, d_DD, d_DR, d_RR);
    cudaDeviceSynchronize();

    // Copying histogram values from GPU to CPU.
    cudaMemcpy(DD, d_DD, size_hist, cudaMemcpyDeviceToHost);
    cudaMemcpy(DR, d_DR, size_hist, cudaMemcpyDeviceToHost);
    cudaMemcpy(RR, d_RR, size_hist, cudaMemcpyDeviceToHost);
    cudaMemcpy(omega, d_omega, size_omega, cudaMemcpyDeviceToHost);

    // Printing all results out.
    printf("\nDD: ");
    print_histogram(DD);
    printf("\nDR: ");
    print_histogram(DR);
    printf("\nRR: ");
    print_histogram(RR);
    printf("\nO:  ");
    print_omegas(omega);

    printf("\n\n");

    print_total_angles_of_histogram(DD, "DD");
    print_total_angles_of_histogram(DR, "DR");
    print_total_angles_of_histogram(RR, "RR");

    // Outputing into files the results of the histograms.
    print_histogram_file(DD, "DD.txt");
    print_histogram_file(DR, "DR.txt");
    print_histogram_file(RR, "RR.txt");

    // Freeing memory in GPU and CPU.
    cudaFree(d_real_galaxy_data);
    cudaFree(d_rand_galaxy_data);
    cudaFree(d_DD);
    cudaFree(d_DR);
    cudaFree(d_RR);
    cudaFree(d_omega);

    free(real_galaxy_data);
    free(rand_galaxy_data);
    free(DD);
    free(DR);
    free(RR);
    free(omega);

    // Calculating total time of execution.
    clock_t end = clock();

    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("\n\nTotal execution time is %.4f seconds", time_spent);

    return 0;
}

// Printing out the GPUs information.
int getDevice(int device_number) {
  int deviceCount;

  cudaGetDeviceCount(&deviceCount);
  printf("   Found %d CUDA devices\n",deviceCount);

  if (deviceCount < 0 || deviceCount > 128) {
    return(-1);
  }

  int device;
  for (device = 0; device < deviceCount; ++device) {
       cudaDeviceProp deviceProp;
       cudaGetDeviceProperties(&deviceProp, device);
       printf("      Device %s                  device %d\n", deviceProp.name,device);
       printf("         compute capability            =        %d.%d\n", deviceProp.major, deviceProp.minor);
       printf("         totalGlobalMemory             =       %.2lf GB\n", deviceProp.totalGlobalMem/1000000000.0);
       printf("         l2CacheSize                   =   %8d B\n", deviceProp.l2CacheSize);
       printf("         regsPerBlock                  =   %8d\n", deviceProp.regsPerBlock);
       printf("         multiProcessorCount           =   %8d\n", deviceProp.multiProcessorCount);
       printf("         maxThreadsPerMultiprocessor   =   %8d\n", deviceProp.maxThreadsPerMultiProcessor);
       printf("         sharedMemPerBlock             =   %8d B\n", (int)deviceProp.sharedMemPerBlock);
       printf("         warpSize                      =   %8d\n", deviceProp.warpSize);
       printf("         clockRate                     =   %8.2lf MHz\n", deviceProp.clockRate/1000.0);
       printf("         maxThreadsPerBlock            =   %8d\n", deviceProp.maxThreadsPerBlock);
       printf("         asyncEngineCount              =   %8d\n", deviceProp.asyncEngineCount);
       printf("         f to lf performance ratio     =   %8d\n", deviceProp.singleToDoublePrecisionPerfRatio);
       printf("         maxGridSize                   =   %d x %d x %d\n",
                          deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
       printf("         maxThreadsDim in thread block =   %d x %d x %d\n",
                          deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
       printf("         concurrentKernels             =   ");
       if(deviceProp.concurrentKernels==1) printf("     yes\n"); else printf("    no\n");
       printf("         deviceOverlap                 =   %8d\n", deviceProp.deviceOverlap);
       if(deviceProp.deviceOverlap == 1)
       printf("            Concurrently copy memory/execute kernel\n");
    }

    cudaSetDevice(device_number);
    cudaGetDevice(&device);

    if (device != device_number) {
        printf("   Unable to set device %d, using device %d instead", device_number, device);
    } else {
        printf("   Using CUDA device %d\n\n", device);
    }

    return(0);
}
