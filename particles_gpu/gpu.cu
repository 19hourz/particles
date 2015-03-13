#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <cuda.h>
#include "common.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <vector>
#include <algorithm>


#define NUM_THREADS 256
//#define _cutoff 0.01    //Value copied from common.cpp
//#define _density 0.0005
#define MAXITEM 4 //Assume at most MAXITEM particles in one bin. Change depends on binSize


extern double size;

double binSize;
int binNum;
//
//  benchmarking program
//


__device__ void apply_force_gpu(particle_t &particle, particle_t &neighbor)
{
  double dx = neighbor.x - particle.x;
  double dy = neighbor.y - particle.y;
  double r2 = dx * dx + dy * dy;
  if( r2 > cutoff*cutoff )
      return;
  //r2 = fmax( r2, min_r*min_r );
  r2 = (r2 > min_r*min_r) ? r2 : min_r*min_r;
  double r = sqrt( r2 );

  //
  //  very simple short-range repulsive force
  //
  double coef = ( 1 - cutoff / r ) / r2 / mass;
  particle.ax += coef * dx;
  particle.ay += coef * dy;

}

__global__ void compute_forces_gpu(particle_t * particles, int n)
{
  // Get thread (particle) ID
  __shared__ int a[1024*1024*1024];
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid >= n) return;

  particles[tid].ax = particles[tid].ay = 0;
  for(int j = 0 ; j < n ; j++)
    a[j]++;
    //apply_force_gpu(particles[tid], particles[j]);

}

__global__ void move_gpu (particle_t * particles, int n, double size)
{

  // Get thread (particle) ID
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid >= n) return;

  particle_t * p = &particles[tid];
    //
    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    //
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x  += p->vx * dt;
    p->y  += p->vy * dt;

    //
    //  bounce from walls
    //
    while( p->x < 0 || p->x > size )
    {
        p->x  = p->x < 0 ? -(p->x) : 2*size-p->x;
        p->vx = -(p->vx);
    }
    while( p->y < 0 || p->y > size )
    {
        p->y  = p->y < 0 ? -(p->y) : 2*size-p->y;
        p->vy = -(p->vy);
    }

}

__global__ void getCount(particle_t* particles, int* count,int n,double binSize,int binNum)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = gridDim.x*blockDim.x;
    for(int i=tid;i<n;i+=offset)
    {
        int x = int(particles[i].x / binSize);
        int y = int(particles[i].y / binSize);
        atomicAdd(count+x*binNum+y,1);
    }
}

__global__ void buildBins(particle_t* particles,particle_t* tmp,int* count,int n,double binSize,int binNum)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = gridDim.x*blockDim.x;
    for(int i=tid;i<n;i+=offset)
    {
        int x = int(particles[i].x / binSize);
        int y = int(particles[i].y / binSize);
        int id = atomicSub(count+x*binNum+y,1);
        tmp[id-1] = particles[i];
    }
}

int main( int argc, char **argv )
{    
    // This takes a few seconds to initialize the runtime
    cudaThreadSynchronize(); 

    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set the number of particles\n" );
        printf( "-o <filename> to specify the output file name\n" );
        return 0;
    }
    
    int n = read_int( argc, argv, "-n", 1000 );

    char *savename = read_string( argc, argv, "-o", NULL );
    
    FILE *fsave = savename ? fopen( savename, "w" ) : NULL;
    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );

    // GPU particle data structure
    particle_t * d_particles,*tmp;
    cudaMalloc((void **) &d_particles, n * sizeof(particle_t));
    cudaMalloc((void **) &tmp, n * sizeof(particle_t));
    
    std::vector<thrust::device_vector<int> > H(4);
    set_size( n );
    binSize = cutoff*4;  
    binNum = int(size / binSize)+1; // Should be around sqrt(N/2)
    printf("Grid Size: %.4lf\n",size);
    printf("Number of Bins: %d*%d\n",binNum,binNum);
    printf("Bin Size: %.2lf\n",binSize);
    
    
    int* cnt;
    cudaMalloc((void **) &cnt, (binNum*binNum+1) * sizeof(int));
    cudaMemset(cnt,0,(binNum*binNum+1)*sizeof(int));
    cnt+=1; //Add one therefore cnt[-1]==0
    int* count = (int*) malloc(binNum*binNum * sizeof(int));
    	
    init_particles( n, particles );

    cudaThreadSynchronize();
    double copy_time = read_timer( );

    // Copy the particles to the GPU
    cudaMemcpy(d_particles, particles, n * sizeof(particle_t), cudaMemcpyHostToDevice);

    cudaThreadSynchronize();
    copy_time = read_timer( ) - copy_time;
    
    //
    //  simulate a number of time steps
    //
    cudaThreadSynchronize();
    double simulation_time = read_timer( );

    for( int step = 0; step < NSTEPS; step++ )
    {
        //
        //  compute forces
        //
        int threadNum = 512;
        int blockNum = min(512,(n+threadNum-1)/threadNum);
        
        
        //Old methods that don't assume maxitems for each bin
        cudaMemset(cnt,0,binNum*binNum*sizeof(int));
        getCount<<<blockNum,threadNum>>>(d_particles,cnt,n,binSize,binNum);
        
        cudaMemcpy(count, cnt, binNum*binNum * sizeof(int), cudaMemcpyDeviceToHost);
        for(int i=1;i<binNum*binNum;i++)  //Prefix sum  could be faster using parallel....
            count[i]+=count[i-1];
        cudaMemcpy(cnt, count, binNum*binNum * sizeof(int), cudaMemcpyHostToDevice);
        buildBins<<<blockNum,threadNum>>>(d_particles,tmp,cnt,n,binSize,binNum);
        std::swap(d_particles,tmp);
        cudaMemcpy(cnt, count, binNum*binNum * sizeof(int), cudaMemcpyHostToDevice);
        
        //int blks = (n + NUM_THREADS - 1) / NUM_THREADS;
    	//compute_forces_gpu <<< blks, NUM_THREADS >>> (d_particles, n);
    	compute_forces_gpu<<<blockNum,threadNum>>> (d_particles,cnt,n);
        
        //
        //  move particles
        //
    	//move_gpu <<< blks, NUM_THREADS >>> (d_particles, n, size);
        
        //
        //  save if necessary
        //
        if( fsave && (step%SAVEFREQ) == 0 ) {
	    // Copy the particles back to the CPU
            cudaMemcpy(particles, d_particles, n * sizeof(particle_t), cudaMemcpyDeviceToHost);
            save( fsave, n, particles);
	    }
    }
    cudaThreadSynchronize();
    simulation_time = read_timer( ) - simulation_time;
    
    printf( "CPU-GPU copy time = %g seconds\n", copy_time);
    printf( "n = %d, simulation time = %g seconds\n", n, simulation_time );
    
    free( particles );
    cudaFree(d_particles);
    if( fsave )
        fclose( fsave );
    
    return 0;
}
