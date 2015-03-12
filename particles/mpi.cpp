#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <vector>
#include <map>
#include <set>
#include <cmath>
#include "common.h"

using std::vector;
using std::map;
using std::set;

#define _cutoff 0.01    //Value copied from common.cpp
#define _density 0.0005

double bin_size, grid_size;
int bin_count;

inline void build_bins(vector<bin_t>& bins, particle_t* particles, int n)
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


inline void get_neighbors(int i, int j, vector<int>& neighbors)
{
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            if (dx == 0 && dy == 0)
                continue;
            if (i + dx >= 0 && i + dx < bin_count && j + dy >= 0 && j + dy < bin_count) {
                int index = (i + dx) * bin_count + j + dy;
                neighbors.push_back(index);
            }
        }
    }
}


//
//  benchmarking program
//
int main( int argc, char **argv )
{    
    int navg, nabsavg=0;
    double dmin, absmin=1.0,davg,absavg=0.0;
    double rdavg,rdmin;
    int rnavg; 
 
    //
    //  process command line parameters
    //
    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set the number of particles\n" );
        printf( "-o <filename> to specify the output file name\n" );
        printf( "-s <filename> to specify a summary file name\n" );
        printf( "-no turns off all correctness checks and particle output\n");
        return 0;
    }
    
    int n = read_int( argc, argv, "-n", 1000 );
    char *savename = read_string( argc, argv, "-o", NULL );
    char *sumname = read_string( argc, argv, "-s", NULL );
    
    //
    //  set up MPI
    //
    int n_proc, rank;
    MPI_Init( &argc, &argv );
    MPI_Comm_size( MPI_COMM_WORLD, &n_proc );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    
    //
    //  allocate generic resources
    //
    FILE *fsave = savename && rank == 0 ? fopen( savename, "w" ) : NULL;
    FILE *fsum = sumname && rank == 0 ? fopen ( sumname, "a" ) : NULL;


    particle_t *particles = new particle_t[n];
    
    MPI_Datatype PARTICLE;
    MPI_Type_contiguous( 6, MPI_DOUBLE, &PARTICLE );
    MPI_Type_commit( &PARTICLE );
    
    //
    //  initialize and distribute the particles (that's fine to leave it unoptimized)
    //
    set_size( n );
    if( rank == 0 )
        init_particles( n, particles );

    MPI_Bcast(particles, n, PARTICLE, 0, MPI_COMM_WORLD);

    vector<bin_t> bins;
    build_bins(bins, particles, n);

    delete[] particles;
    particles = NULL;

    int x_bins_per_proc = bin_count / n_proc + 1;
    int my_bins_start = x_bins_per_proc * rank;
    int my_bins_end = min(bin_count, x_bins_per_proc * (rank + 1));
    printf("worker %d: from %d to %d.\n", rank, my_bins_start, my_bins_end);
    //
    //  simulate a number of time steps
    //
    double simulation_time = read_timer( );
    for( int step = 0; step < NSTEPS; step++ )
    {
        navg = 0;
        dmin = 1.0;
        davg = 0.0;

        if( find_option(argc, argv, "-no" ) == -1 )
            if( fsave && (step%SAVEFREQ) == 0 )
                save( fsave, n, particles );

        // compute local forces
        for (int i = my_bins_start; i < my_bins_end; ++i) {
            for (int j = 0; j < bin_count; ++j) {
                compute_forces_for_bin(bins, i, j, dmin, davg, navg);
            }
        }

        if (find_option( argc, argv, "-no" ) == -1) {
          MPI_Reduce(&davg,&rdavg,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
          MPI_Reduce(&navg,&rnavg,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
          MPI_Reduce(&dmin,&rdmin,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
          if (rank == 0){
            if (rnavg) {
              absavg +=  rdavg/rnavg;
              nabsavg++;
            }
            if (rdmin < absmin) 
                absmin = rdmin;
          }
        }

        // move, but not rebin
        for (int i = my_bins_start; i < my_bins_end; ++i) {
            for (int j = 0; j < bin_count; ++j) {
                move(bins[i * bin_count + j]);
            }
        }
        for (int index = my_bins_start; index < my_bins_end; ++index) {
            for (int k = 0; k < bins[index].size(); ++k) {
                move(bins[index][k]);
            }
        }

        // send
        for (int count = 0; count < 2; ++count) {
            vector<MPI_Request> requests;
            std::map<int, bin_t> old_bins;
            for (int index = my_bins_start; index < my_bins_end; ++index) {
                int i = index / bin_count, j = index % bin_count;
                vector<int> neighbors;
                set<int> neighbor_proc;
                get_neighbors(i, j, neighbors);

                // copy old data until completion
                old_bins[index] = bins[index];

                // send
                for (int k = 0; k < neighbors.size(); ++k) {
                    int who = neighbors[k] / bins_per_proc;
                    if (who != rank)
                        neighbor_proc.insert(who);
                }

                for (set<int>::iterator it = neighbor_proc.begin(); it != neighbor_proc.end(); ++it) {
                    requests.push_back(MPI_Request());
                    MPI_Isend(old_bins[index].data(), old_bins[index].size(), PARTICLE, *it, index, MPI_COMM_WORLD, &(requests.back()));
                    //printf("worker %d: sending bin %d with %ld particles to worker %d.\n",
                    //    rank, index, old_bins[index].size(), *it);
                }
            }

            //printf("worker %d: barrier.\n", rank);
            //MPI_Barrier(MPI_COMM_WORLD);

            // recv
            //printf("worker %d: recv.\n", rank);
            for (int index = my_bins_start; index < my_bins_end; ++index) {
                int i = index / bin_count, j = index % bin_count;

                vector<int> neighbors;
                get_neighbors(i, j, neighbors);
                set<int> neighbor_proc;

                for (int k = 0; k < neighbors.size(); ++k) {
                    int who = neighbors[k] / bins_per_proc;
                    if (who != rank)
                        neighbor_proc.insert(who);
                }

                for (set<int>::iterator it = neighbor_proc.begin(); it != neighbor_proc.end(); ++it) {
                    int who = *it;
                    MPI_Status status;
                    MPI_Probe(who, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

                    int len = 0;
                    MPI_Get_count(&status, PARTICLE, &len);
                    assert(len >= 0);
                    assert(len <= n);

                    int bin = status.MPI_TAG;
                    assert(bin < bin_count * bin_count);
                    assert(0 <= bin);

                    bins[bin].resize(len);

                    MPI_Recv(bins[bin].data(), bins[bin].size(), PARTICLE, who, bin, MPI_COMM_WORLD, &status);
                    //printf("worker %d: recv bin %d with %d particles from worker %d.\n", 
                    //    rank, bin, len, who);
                }
            }

            // rebin
            vector<int> local_bin_indices;
            for (int index = my_bins_start; index < my_bins_end; ++index) {
                int i = index / bin_count, j = index % bin_count;
                vector<int> neighbors;
                get_neighbors(i, j, neighbors);
                local_bin_indices.insert(local_bin_indices.end(), neighbors.begin(), neighbors.end());
            }
            // dedup
            std::sort(local_bin_indices.begin(), local_bin_indices.end());
            local_bin_indices.erase(std::unique(local_bin_indices.begin(), local_bin_indices.end()), local_bin_indices.end());

            bin_t local_bin;
            for (int b = 0; b < local_bin_indices.size(); ++b) {
                int i = local_bin_indices[b];
                local_bin.insert(local_bin.end(), bins[i].begin(), bins[i].end());
                bins[i].clear();
            }

            for (int i = 0; i < local_bin.size(); ++i) {
                bin_particle(local_bin[i], bins);
            }

            //printf("worker %d: barrier 2.\n", rank);
            //MPI_Barrier(MPI_COMM_WORLD);
        }
    }
    simulation_time = read_timer( ) - simulation_time;
  
    if (rank == 0) {  
      printf( "n = %d, simulation time = %g seconds", n, simulation_time);

      if( find_option( argc, argv, "-no" ) == -1 )
      {
        if (nabsavg) absavg /= nabsavg;
          // 
          //  -The minimum distance absmin between 2 particles during the run of the simulation
          //  -A Correct simulation will have particles stay at greater than 0.4 (of cutoff) with typical values between .7-.8
          //  -A simulation where particles don't interact correctly will be less than 0.4 (of cutoff) with typical values between .01-.05
          //
          //  -The average distance absavg is ~.95 when most particles are interacting correctly and ~.66 when no particles are interacting
          //
          printf( ", absmin = %lf, absavg = %lf", absmin, absavg);
          if (absmin < 0.4) printf ("\nThe minimum distance is below 0.4 meaning that some particle is not interacting");
          if (absavg < 0.8) printf ("\nThe average distance is below 0.8 meaning that most particles are not interacting");
      }
      printf("\n");     
        
      //  
      // Printing summary data
      //  
      if( fsum)
        fprintf(fsum,"%d %d %g\n",n,n_proc,simulation_time);
    }
  
    //
    //  release resources
    //
    if ( fsum )
        fclose( fsum );
    if( fsave )
        fclose( fsave );
    
    MPI_Finalize( );
    
    return 0;
}
