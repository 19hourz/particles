#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <vector>
#include "omp.h"
#include "common.h"

using std::vector;

#define _cutoff 0.01
#define _density 0.0005

double binSize,gridSize;
int binNum;
vector<particle_t>* data;

void buildBins(int n,particle_t* particles)
{
	gridSize = sqrt(n*_density);
	binSize = 3*_cutoff;  //Should not have the constant 3... But if not, the simulation would fail...Strange
	
	binNum = int(gridSize / binSize)+1; // Should be around sqrt(N)
	printf("Grid Size: %.4lf\n",gridSize);
	printf("Number of Bins: %d*%d\n",binNum,binNum);
	printf("Bin Size: %.4lf\n",binSize);
	// Increase\Decrease binNum to be something like 2^k?
	data = new vector<particle_t>[binNum*binNum];
	for(int i=0;i<n;i++)
	{
		int x = int(particles[i].x / binSize);
		int y = int(particles[i].y / binSize);
		data[x*binNum+y].push_back(particles[i]);
	}
}

//
//  benchmarking program
//
int main( int argc, char **argv )
{    
    int navg,nabsavg=0;
    double davg,dmin, absmin=1.0, absavg=0.0;

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
    
    FILE *fsave = savename ? fopen( savename, "w" ) : NULL;
    FILE *fsum = sumname ? fopen ( sumname, "a" ) : NULL;

    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );
    set_size( n );
    init_particles( n, particles );
    
    buildBins(n,particles);
    //
    //  simulate a number of time steps
    //
    //omp_set_num_threads(32);
    //printf("Threads: %d\n",omp_get_num_threads());

	int numthreads;
    double simulation_time = read_timer( );
	
    for( int step = 0; step < NSTEPS; step++ )
    {
    	//if (step%10==0)printf("%d ",step);
		navg = 0;
        davg = 0.0;
		dmin = 1.0;
        //
        //  compute forces
        //
        
        /*for( int i = 0; i < n; i++ )
        {
            particles[i].ax = particles[i].ay = 0;
            for (int j = 0; j < n; j++ )
				apply_force( particles[i], particles[j],&dmin,&davg,&navg);
        }*/
        for(int i=0;i<binNum;i++)
        	for(int j=0;j<binNum;j++)
        	{
        		vector<particle_t>&vec = data[i*binNum+j];
        		for(int k=0;k<vec.size();k++)
        			vec[k].ax=vec[k].ay=0;
				for(int dx=-1;dx<=1;dx++)
    				for(int dy=-1;dy<=1;dy++)
        				if (i+dx>=0 && i+dx<binNum && j+dy>=0 && j+dy<binNum)
        				{
        					vector<particle_t>&vec2 = data[(i+dx)*binNum+j+dy];
        		//for(int t=0;t<binNum*binNum;t++){
        		//			vector<particle_t>&vec2 = data[t];
        					for(int k=0;k<vec.size();k++)
        						for(int l=0;l<vec2.size();l++)
	        						apply_force( vec[k], vec2[l],&dmin,&davg,&navg);
        				}
        	}
 
        //
        //  move particles
        //
        /*for( int i = 0; i < n; i++ ) 
            move( particles[i] );		*/
		
		for(int i=0;i<binNum;i++)
        	for(int j=0;j<binNum;j++)
        	{
        		vector<particle_t>&vec = data[i*binNum+j];
        		int tail=vec.size(),k=0;
        		for(;k<tail;)
        		{
					move( vec[k] );
					int x = int(vec[k].x / binSize);
					int y = int(vec[k].y / binSize);
					if (x==i && y==j)
						k++;
					else
					{
						data[x*binNum+y].push_back(vec[k]);
						vec[k] = vec[--tail];
					}
        		}
        		vec.resize(k);
        	}
        /*int count=0;
		for(int i=0;i<binNum;i++)
        	for(int j=0;j<binNum;j++)
        	{
        		vector<particle_t>&vec = data[i*binNum+j];
        		count+=vec.size();
        		for(int k=0;k<vec.size();k++)
        		{
					int x = int(vec[k].x / binSize);
					int y = int(vec[k].y / binSize);
					if (!(x==i && y==j))
						printf("Error!\n");
        		}
        	}
    	assert(count==n);*/
        if( find_option( argc, argv, "-no" ) == -1 )
        {
          //
          // Computing statistical data
          //
          	if (navg) {
	        	absavg +=  davg/navg;
	        	nabsavg++;
          	}
          	if (dmin < absmin) absmin = dmin;
		
          //
          //  save if necessary
          //
          	if( fsave && (step%SAVEFREQ) == 0 )
				save( fsave, n, particles );
        }
    }
    simulation_time = read_timer( ) - simulation_time;
    
    
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
        fprintf(fsum,"%d %g\n",n,simulation_time);
 
    //
    // Clearing space
    //
    if( fsum )
        fclose( fsum );    
    free( particles );
    //delete(data);
    if( fsave )
        fclose( fsave );
    
    return 0;
}
