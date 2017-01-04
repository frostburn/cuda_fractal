#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <limits.h>

#include <cuda_runtime.h>

#include <omp.h>

// Compile with
// nvcc main.cu -Wno-deprecated-gpu-targets  -Xcompiler -fopenmp

// #define DOUBLE_PREC
// Double precission seems to be about 4 times slower.
#ifdef DOUBLE_PREC
typedef double real;
#else
typedef float real;
#endif

#if __CUDA_ARCH__ < 600
__device__ double atomicAddDouble(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

/* Public domain code for JKISS RNG */
// http://www0.cs.ucl.ac.uk/staff/d.jones/GoodPracticeRNG.pdf

typedef struct jkiss
{
    unsigned int x, y, z, c;
} jkiss;

#define JKISS_DEFAULT_SEED ({123456789, 987654321, 43219876, 6543217})

unsigned int devrand(void)
{
    int fn;
    unsigned int r;
    fn = open("/dev/urandom", O_RDONLY);
    if (fn == -1)
        exit(-1); /* Failed! */
    if (read(fn, &r, 4) != 4)
        exit(-1); /* Failed! */
    close(fn);
    return r;
}

real devrand_uniform()
{
    return devrand() / (real)UINT_MAX;
}

real devrand_gauss()
{
    return devrand_uniform() + devrand_uniform() + devrand_uniform() - devrand_uniform() - devrand_uniform() - devrand_uniform();
}

/* Initialise KISS generator using thread indices and a devrand base seed */
__host__ __device__
void jkiss_init(jkiss *j, unsigned int tid, unsigned int bid, unsigned int base_seed)
{
    // This is probably a horrible way to initialize, but whatever.
    tid += base_seed * 1233;
    bid -= base_seed * 123239;
    j->x = 123456789 + tid * tid * (tid + 2345) * 238461 * (bid + 2123123);
    j->y = 987654321 + 23123 * bid * bid + tid * (12323 + bid);  // Must not be zero.
    j->z = 43219876 + bid * bid * (bid + 123) * 923873 * (tid + 32234);  // Avoid z=c=0.
    j->c = base_seed;

    if (j->y == 0) {
        j->y = 12345;
    }
    if (j->c == 0) {
        j->c = 789;
    }
}

__host__ __device__
unsigned int jkiss_step(jkiss *j)
{
    unsigned long long t;
    j->x = 314527869 * j->x + 1234567;
    j->y ^= j->y << 5; j->y ^= j->y >> 7; j->y ^= j->y << 22;
    t = 4294584393ULL * j->z + j->c;
    j->c = t >> 32;
    j->z = t;
    return j->x + j->y + j->z;
}

__host__ __device__
real jkiss_uniform(jkiss *j)
{
    return jkiss_step(j) / (real)UINT_MAX;
}

__host__ __device__
real jkiss_gauss(jkiss *j)
{
    return jkiss_uniform(j) + jkiss_uniform(j) + jkiss_uniform(j) - jkiss_uniform(j) - jkiss_uniform(j) - jkiss_uniform(j);
}

const size_t BLOCKS_PER_LAUNCH = 13;
const size_t THREADS_PER_BLOCK = 1024;
const size_t SAMPLES_PER_THREAD = 16;

void accumulate_cpu(
    real *h_img,
    int width, int height,
    size_t iterations,
    real spot_x, real spot_y,
    real spot_scale_x, real spot_scale_y,
    real center_x, real center_y,
    real proj_xx, real proj_xy, real proj_yx, real proj_yy,
    size_t samples,
    unsigned int base_seed
) {
    jkiss jk_;
    jkiss *jk = &jk_;
    jkiss_init(jk, 1234, 5678, base_seed);
    for (size_t j = 0; j < samples; j++) {
        real x = 0;
        real y = 0;
        real cx = jkiss_gauss(jk) * spot_scale_x + spot_x;
        real cy = jkiss_gauss(jk) * spot_scale_y + spot_y;
        size_t i = 0;
        for (i = 0; i < iterations; i++) {
            real x_sqr = x * x;
            real y_sqr = y * y;
            if (x_sqr + y_sqr > 128) {
                break;
            }
            y = 2*x*y + cy;
            x = x_sqr - y_sqr + cx;
        }
        if (i == iterations) {
            continue;
        }
        // Only escaping samples are used in a buddhabrot.
        x = 0;
        y = 0;
        for (size_t k = 0; k < i; k++) {
            real x_sqr = x * x;
            real y_sqr = y * y;
            y = 2*x*y + cy;
            x = x_sqr - y_sqr + cx;

            if (k == 0) {
                continue;
            }
            x_sqr = x - center_x;
            y_sqr = y - center_y;
            real temp = x_sqr;
            // After translation the origin projects to the middle of the image.
            x_sqr = temp * proj_xx + y_sqr * proj_yx;
            y_sqr = temp * proj_xy + y_sqr * proj_yy;

            x_sqr += 0.5;
            y_sqr = 0.5 - y_sqr;

            int x_idx = x_sqr * width;
            int y_idx = y_sqr * height;
            if (x_sqr >= 0 && x_idx < width && y_sqr >= 0 && y_idx < height) {
                #pragma omp atomic
                h_img[x_idx + width * y_idx]++;
            }
        }
    }
}


__global__
void accumulate(
    real *d_img,
    int width, int height,
    size_t iterations,
    real spot_x, real spot_y,
    real spot_scale_x, real spot_scale_y,
    real center_x, real center_y,
    real proj_xx, real proj_xy, real proj_yx, real proj_yy,
    size_t samples,
    unsigned int base_seed
) {
    jkiss jk_;
    jkiss *jk = &jk_;
    jkiss_init(jk, threadIdx.x, blockIdx.x, base_seed);
    for (size_t j = 0; j < samples; j++) {
        real cx = jkiss_gauss(jk) * spot_scale_x + spot_x;
        real cy = jkiss_gauss(jk) * spot_scale_y + spot_y;
        real x = cx;
        real y = cy;
        size_t i = 0;
        for (i = 0; i < iterations; i++) {
            real x_sqr = x * x;
            real y_sqr = y * y;
            if (x_sqr + y_sqr > 128) {
                break;
            }
            y = 2*x*y + cy;
            x = x_sqr - y_sqr + cx;
        }
        if (i == iterations) {
            continue;
        }
        // Only escaping samples are used in a buddhabrot.
        x = cx;
        y = cy;
        for (size_t k = 0; k < i; k++) {
            real x_sqr = x * x;
            real y_sqr = y * y;
            y = 2*x*y + cy;
            x = x_sqr - y_sqr + cx;

            x_sqr = x - center_x;
            y_sqr = y - center_y;
            real temp = x_sqr;
            // After translation the origin projects to the middle of the image.
            x_sqr = temp * proj_xx + y_sqr * proj_yx;
            y_sqr = temp * proj_xy + y_sqr * proj_yy;

            x_sqr += 0.5;
            y_sqr = 0.5 - y_sqr;

            int x_idx = x_sqr * width;
            int y_idx = y_sqr * height;
            if (x_sqr >= 0 && x_idx < width && y_sqr >= 0 && y_idx < height) {
                real red_val = 0;
                real green_val = 0;
                real blue_val = 0;

                red_val += 1.0 / (k + 1.0);
                green_val += 1000.0 / (k*k + 10000.0);
                blue_val += 10000.0 / (k*k + 200000.0);

                if (k % 3 == 0) {
                    red_val += 0.1;
                    blue_val += 0.1;
                }
                else if (k % 3 == 1) {
                    red_val += 0.1;
                    green_val += 0.1;
                }
                else {
                    green_val += 0.1;
                    blue_val += 0.1;
                }

                real *pixel = d_img + 3 * (x_idx + width * y_idx);
                real *red = pixel;
                real *green = pixel + 1;
                real *blue = pixel + 2;
                #ifdef DOUBLE_PREC
                    atomicAddDouble(d_img + (x_idx + width * y_idx), 1.0);
                #else
                    atomicAdd(red, red_val);
                    atomicAdd(green, green_val);
                    atomicAdd(blue, blue_val);
                #endif
            }
        }
    }
    __syncthreads();
}

#define NUM_COLORS (3)

int main(int argc, char **argv)
{
    // Error code to check return values for CUDA calls
    // cudaError_t err = cudaSuccess;

    size_t launches = 100;
    int width = 200;
    int height = 200;
    size_t iterations = 1000;
    real spot_x = -0.666;
    real spot_y = 0;
    real spot_scale_x = 1;
    real spot_scale_y = 1;
    real center_x = -0.75;
    real center_y = 0;
    real proj_xx = 0.3;
    real proj_xy = 0;
    real proj_yx = 0;
    real proj_yy = 0.3;

    int c = 1;
    if (argc > c) {
        launches = atoi(argv[c]);
    }
    c++;
    if (argc > c) {
        width = atoi(argv[c]);
    }
    c++;
    if (argc > c) {
        height = atoi(argv[c]);
    }
    c++;
    if (argc > c) {
        iterations = atoi(argv[c]);
    }
    c++;
    if (argc > c) {
        spot_x = atof(argv[c]);
    }
    c++;
    if (argc > c) {
        spot_y = atof(argv[c]);
    }
    c++;
    if (argc > c) {
        spot_scale_x = atof(argv[c]);
    }
    c++;
    if (argc > c) {
        spot_scale_y = atof(argv[c]);
    }
    c++;
    if (argc > c) {
        center_x = atof(argv[c]);
    }
    c++;
    if (argc > c) {
        center_y = atof(argv[c]);
    }
    c++;
    if (argc > c) {
        proj_xx = atof(argv[c]);
    }
    c++;
    if (argc > c) {
        proj_xy = atof(argv[c]);
    }
    c++;
    if (argc > c) {
        proj_yx = atof(argv[c]);
    }
    c++;
    if (argc > c) {
        proj_yy = atof(argv[c]);
    }
    c++;

    int num_bins = width * height * NUM_COLORS;
    int img_size = num_bins * sizeof(real);
    real *h_img = (real *)malloc(img_size);
    real *h_img_2 = (real *)malloc(img_size);
    real *d_img;
    cudaMalloc(&d_img, img_size);

    for (int i = 0; i < num_bins; i++) {
        h_img[i] = 0;
        h_img_2[i] = 0;
    }
    cudaMemcpy(d_img, h_img, img_size, cudaMemcpyHostToDevice);

    unsigned int base_seed;
    for (size_t i = 0; i < launches; i++) {
        base_seed = devrand();
        accumulate<<<BLOCKS_PER_LAUNCH, THREADS_PER_BLOCK>>>(
            d_img,
            width, height,
            iterations,
            spot_x, spot_y,
            spot_scale_x, spot_scale_y,
            center_x, center_y,
            proj_xx, proj_xy, proj_yx, proj_yy,
            SAMPLES_PER_THREAD,
            base_seed
        );
    }

// Parallel CPU/GPU disabled for single precission as the sample factor is too large to be useful.
#ifdef DOUBLE_PREC
    Cpu processing and thus double precission deprecated for the time being.
    #pragma omp parallel
    {
        size_t samples = launches * BLOCKS_PER_LAUNCH * THREADS_PER_BLOCK * SAMPLES_PER_THREAD;
        #ifdef DOUBLE_PREC
            samples /= 11;
        #else
            samples /= 200;
        #endif
        base_seed = devrand();
        accumulate_cpu(
            h_img_2,
            width, height,
            iterations,
            spot_x, spot_y,
            spot_scale_x, spot_scale_y,
            center_x, center_y,
            proj_xx, proj_xy, proj_yx, proj_yy,
            samples,
            base_seed
        );
    }
#endif

    cudaDeviceSynchronize();
    cudaMemcpy(h_img, d_img, img_size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < num_bins; i++) {
        h_img[i] += h_img_2[i];
    }

    // fprintf(stderr, "Dumbing to stdout.\n");
#ifdef EVAL_API
    // Text based eval API is bestest.
    printf("array([");
    for (int i = 0; i < num_bins; i++) {
        printf("%g,", h_img[i]);
    }
    printf("])\n");
#else
    write(fileno(stdout), h_img, img_size);
#endif
    //fprintf(stderr, "Dump complete.\n");

    free(h_img);
    free(h_img_2);
    cudaFree(d_img);
    exit(EXIT_SUCCESS);
}
