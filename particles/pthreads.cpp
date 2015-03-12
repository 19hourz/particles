#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <pthread.h>
#include <vector>
#include <map>
#include <set>
#include "common.h"
#ifdef __APPLE__
#include "pthread_barrier.h"
#endif

#define _cutoff 0.01    //Value copied from common.cpp
#define _density 0.0005

using std::vector;
using std::map;
using std::set;

namespace {

    int n, n_threads,no_output=0;
    
    FILE *fsave,*fsum;
    particle_t *particles;
    
    pthread_barrier_t barrier;
    
    pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
    
    double gabsmin=1.0,gabsavg=0.0;

    struct thread_data_t {
        int thread_no;
        int threads;
    };

    vector<bin_t> bins;
    vector<bin_t> temp_bins;

}

double bin_size, grid_size;
int bin_count;

void build_bins(vector<bin_t>& bins, particle_t* particles, int n)
{
    grid_size = sqrt(n * _density);
    bin_size = _cutoff;  
    bin_count = int(grid_size / bin_size) + 1; // Should be around sqrt(N/2)

    printf("Grid Size: %.4lf\n", grid_size);
    printf("Number of Bins: %d*%d\n", bin_count, bin_count);
    printf("Bin Size: %.2lf\n", bin_size);
    // Increase\Decrease bin_count to be something like 2^k?
    
    bins.resize(bin_count * bin_count);

    for (int i = 0; i < n; i++)
    {
        int x = int(particles[i].x / bin_size);
        int y = int(particles[i].y / bin_size);
        bins[x*bin_count + y].push_back(particles[i]);
    }
}

inline void compute_forces_for_bin(vector<bin_t>& bins, int i, int j, double& dmin, double& davg, int& navg)
{
    bin_t& vec = bins[i * bin_count + j];

    for (int k = 0; k < vec.size(); k++)
        vec[k].ax = vec[k].ay = 0;

    for (int dx = -1; dx <= 1; dx++)   //Search over nearby 8 bins and itself
    {
        for (int dy = -1; dy <= 1; dy++)
        {
            if (i + dx >= 0 && i + dx < bin_count && j + dy >= 0 && j + dy < bin_count)
            {
                bin_t& vec2 = bins[(i+dx) * bin_count + j + dy];
                for (int k = 0; k < vec.size(); k++)
                    for (int l = 0; l < vec2.size(); l++)
                        apply_force( vec[k], vec2[l], &dmin, &davg, &navg);
            }
        }
    }
}

inline void bin_particle(particle_t& particle, vector<bin_t>& bins)
{
    int x = particle.x / bin_size;
    int y = particle.y / bin_size;
    bins[x*bin_count + y].push_back(particle);
}

//
//  check that pthreads routine call was successful
//
#define P( condition ) {if( (condition) != 0 ) { printf( "\n FAILURE in %s, line %d\n", __FILE__, __LINE__ );exit( 1 );}}

void *thread_routine(int thread_no, int threads);

void *thread_launch(void *data)
{
    struct thread_data_t *typed = (struct thread_data_t *) data;
    thread_routine(typed->thread_no, typed->threads);
    return NULL;
}
//
//  This is where the action happens
//
void *thread_routine(int thread_no, int threads)
{
    int navg,nabsavg=0;
    double dmin,absmin=1.0,davg,absavg=0.0;

    const int bins_per_thread = bins.size() / threads + 1;
    const int bin_start = thread_no * bins_per_thread;
    const int bin_end = min((thread_no+1) * bins_per_thread, bins.size());
    //
    //  simulate a number of time steps
    //
    for( int step = 0; step < NSTEPS; step++ )
    {
        dmin = 1.0;
        navg = 0;
        davg = 0.0;
        //
        //  compute forces
        //

        // compute local forces
        for (int index = bin_start; index < bin_end; ++index) {
            int i = index / bin_count, j = index % bin_count;
            compute_forces_for_bin(bins, i, j, dmin, davg, navg);
        }
        
        int rc = pthread_barrier_wait( &barrier );
        if (rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD) {
            printf("Could not wait on barrier 1\n");
            exit(-1);
        }
        
        if( no_output == 0)
        {
            if (navg) {
                absavg +=  davg/navg;
                nabsavg++;
            }
            if (dmin < absmin) absmin = dmin;
        }

        // move, but not rebin
        for (int index = bin_start; index < bin_end; ++index) {
            int i = index / bin_count, j = index % bin_count;
            bin_t& bin = bins[index];
            int k = 0, tail = bin.size();
            for (; k < tail; ) {
                move(bin[k]);
                int x = int(bin[k].x / bin_size);  //Check the position
                int y = int(bin[k].y / bin_size);
                if (x == i && y == j) {
                    ++k;
                } else {
                    temp_bins[thread_no].push_back(bin[k]);
                    bin[k] = bin[--tail];
                }
            }
            bin.resize(k);
        }
        
        rc = pthread_barrier_wait( &barrier );
        if (rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD) {
            printf("Could not wait on barrier 2\n");
            exit(-1);
        }

        // rebin
        if (thread_no == 0) {
            for (int i = 0; i < threads; ++i) {
                bin_t& bin = temp_bins[i];
                for (int k = 0; k < bin.size(); ++k) {
                    bin_particle(bin[k], bins);
                }
                bin.clear();
            }
        }

        rc = pthread_barrier_wait( &barrier );
        if (rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD) {
            printf("Could not wait on barrier 3\n");
            exit(-1);
        }

        //
        //  save if necessary
        //
        if (no_output == 0) 
          if( thread_no == 0 && fsave && (step%SAVEFREQ) == 0 )
            save( fsave, n, particles );
        
    }
     
    if (no_output == 0 )
    {
      absavg /= nabsavg; 	
      //printf("Thread %d has absmin = %lf and absavg = %lf\n",thread_id,absmin,absavg);
      pthread_mutex_lock(&mutex);
      gabsavg += absavg;
      if (absmin < gabsmin) gabsmin = absmin;
      pthread_mutex_unlock(&mutex);    
    }

    return NULL;
}

//
//  benchmarking program
//
int main( int argc, char **argv )
{    
    //
    //  process command line
    //
    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set the number of particles\n" );
        printf( "-p <int> to set the number of threads\n" );
        printf( "-o <filename> to specify the output file name\n" );
        printf( "-s <filename> to specify a summary file name\n" );
        printf( "-no turns off all correctness checks and particle output\n");        
        return 0;
    }
    
    n = read_int( argc, argv, "-n", 1000 );
    n_threads = read_int( argc, argv, "-p", 2 );
    char *savename = read_string( argc, argv, "-o", NULL );
    char *sumname = read_string( argc, argv, "-s", NULL );

    fsave = savename ? fopen( savename, "w" ) : NULL;
    fsum = sumname ? fopen ( sumname, "a" ) : NULL;

    if( find_option( argc, argv, "-no" ) != -1 )
      no_output = 1;

    //
    //  allocate resources
    //

    particles = new particle_t[n];
    set_size(n);
    init_particles( n, particles );

    pthread_attr_t attr;
    P( pthread_attr_init( &attr ) );
    P( pthread_barrier_init( &barrier, NULL, n_threads ) );

    build_bins(bins, particles, n);

    temp_bins.resize(n_threads);

    pthread_t* threads = new pthread_t[n_threads];
    
    //
    //  do the parallel work
    //
    double simulation_time = read_timer( );
    for (int i = 1; i < n_threads; i++) {
        thread_data_t *data = new thread_data_t();
        data->thread_no = i;
        data->threads = n_threads;
        P(pthread_create( &threads[i], &attr, thread_launch, (void *)data));
    }

    thread_data_t *data = new thread_data_t();
    data->thread_no = 0;
    data->threads = n_threads;
    thread_launch(data);
    
    for( int i = 1; i < n_threads; i++ ) 
        P( pthread_join( threads[i], NULL ) );


    simulation_time = read_timer( ) - simulation_time;
   
    printf( "n = %d, simulation time = %g seconds", n, simulation_time);

    if( find_option( argc, argv, "-no" ) == -1 )
    {
      gabsavg /= (n_threads*1.0);
      // 
      //  -The minimum distance absmin between 2 particles during the run of the simulation
      //  -A Correct simulation will have particles stay at greater than 0.4 (of cutoff) with typical values between .7-.8
      //  -A simulation where particles don't interact correctly will be less than 0.4 (of cutoff) with typical values between .01-.05
      //
      //  -The average distance absavg is ~.95 when most particles are interacting correctly and ~.66 when no particles are interacting
      //
      printf( ", absmin = %lf, absavg = %lf", gabsmin, gabsavg);
      if (gabsmin < 0.4) printf ("\nThe minimum distance is below 0.4 meaning that some particle is not interacting ");
      if (gabsavg < 0.8) printf ("\nThe average distance is below 0.8 meaning that most particles are not interacting ");
    }
    printf("\n");

    //
    // Printing summary data
    //
    if( fsum)
        fprintf(fsum,"%d %d %g\n",n,n_threads,simulation_time); 
    
    //
    //  release resources
    //
    P( pthread_barrier_destroy( &barrier ) );
    P( pthread_attr_destroy( &attr ) );
    delete[] particles;
    free( threads );
    
    if( fsave )
        fclose( fsave );
    if( fsum )
        fclose ( fsum );
    
    return 0;
}
