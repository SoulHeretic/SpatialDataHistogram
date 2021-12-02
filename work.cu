#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <stdint.h>
#include "timing.cuh"


#define BOX_SIZE	23000 /* size of the data box on one dimension            */
#define BUCKET_TYPE unsigned long long // data type of a bucket

/* descriptors for single atom in the tree */
typedef struct atomdesc {
	double x_pos;
	double y_pos;
	double z_pos;
} atom;

typedef struct hist_entry{
	unsigned long long d_cnt;   /* need a long long type as the count might be huge */
} bucket;


bucket * histogram;		/* list of all buckets in the histogram   */
bucket * d_histogram;	
bucket * d_to_h_histogram;	
long long	n_atoms;	/* total number of data points            */
int num_buckets;		/* total number of buckets in the histogram */
double   PDH_res;		/* value of w                             */
atom * atom_list;		/* list of all data points                */
atom * d_atom_list;

int blockSize;

/* These are for an old way of tracking time */
struct timezone Idunno;	
struct timeval startTime, endTime;


/* 
	distance of two points in the atom_list 
*/
double p2p_distance(int ind1, int ind2) {
	
	double x1 = atom_list[ind1].x_pos;
	double x2 = atom_list[ind2].x_pos;
	double y1 = atom_list[ind1].y_pos;
	double y2 = atom_list[ind2].y_pos;
	double z1 = atom_list[ind1].z_pos;
	double z2 = atom_list[ind2].z_pos;
		
	return sqrt((x1 - x2)*(x1-x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
}

/* 
	brute-force SDH solution in a single CPU thread 
*/
int PDH_baseline() {
	int i, j, h_pos;
	double dist;
	
	for(i = 0; i < n_atoms; i++) {
		for(j = i+1; j < n_atoms; j++) {
			dist = p2p_distance(i,j);
			h_pos = (int) (dist / PDH_res);
			histogram[h_pos].d_cnt++;
		} 
	}
	return 0;
}
//Kernel will calculate distance between atoms
__device__ double d_p2p_distance(atom &L, atom &R) {
	
	double x1 = L.x_pos;
	double x2 = R.x_pos;
	double y1 = L.y_pos;
	double y2 = R.y_pos;
	double z1 = L.z_pos;
	double z2 = R.z_pos;
		
	return sqrt((x1 - x2) * (x1-x2) + (y1 - y2) * (y1 - y2) + (z1 - z2)  *(z1 - z2));
}

extern __shared__ atom R[];
__global__ void kernel(long long n_atoms, bucket *d_histogram, double PDH_res, atom *d_atom_list){
	int h_pos;
	double dist;
	int B = blockDim.x;
	int t = threadIdx.x;
	int b = blockIdx.x;
    long long tid = blockDim.x * blockIdx.x + threadIdx.x;
	atom L;
    L = d_atom_list[tid];

	bucket* SHMout = (bucket *)&R[B];

    int num_buckets = (int)(BOX_SIZE * 1.732 / PDH_res) + 1;

	int HalfB = (int)(B/2);	
    if(B % 2 == 0 && t >= HalfB){
		HalfB = HalfB - 1;
	}

	for(int i = t; i <  num_buckets; i = i + B){
		SHMout[i].d_cnt = 0;
	}
	__syncthreads();	

	for(int i = b + 1 ; i < gridDim.x ; i++) {
		int blockTravel = i * B;
		R[t] = d_atom_list[t + blockTravel];

		if(blockTravel < n_atoms){
			for(int j = 0; j < B; j++) {
				if(j + blockTravel < n_atoms){
					dist = d_p2p_distance(L, R[j]);
					h_pos = (int) (dist / PDH_res);
					atomicAdd(&(SHMout[h_pos].d_cnt), 1);	 				
				}
			}
		}
		__syncthreads();
	}

	R[t] = L;

	if(tid < n_atoms){
		for(int j = 0; j < HalfB; j++){
			int index = (t + j + 1) % B;
			if(index + ( B * b ) < n_atoms){
				dist = d_p2p_distance(L, R[index]);
				h_pos = (int) (dist / PDH_res);
				atomicAdd(&(SHMout[h_pos].d_cnt), 1);
			}
		}
	}
	__syncthreads();
	for(int i = t; i <  num_buckets; i = i + B){
		atomicAdd(&(d_histogram[i].d_cnt), (SHMout[i].d_cnt));
	}
}



/* 
	set a checkpoint and show the (natural) running time in seconds 
*/
double report_running_time() {
	long sec_diff, usec_diff;
	gettimeofday(&endTime, &Idunno);
	sec_diff = endTime.tv_sec - startTime.tv_sec;
	usec_diff= endTime.tv_usec-startTime.tv_usec;
	if(usec_diff < 0) {
		sec_diff --;
		usec_diff += 1000000;
	}
	printf("Running time for CPU version: %ld.%06ld\n", sec_diff, usec_diff);
	return (double)(sec_diff*1.0 + usec_diff/1000000.0);
}


void output_histogram(){
	int i; 
	long long total_cnt = 0;
	for(i=0; i< num_buckets; i++) {
		if(i%5 == 0) /* we print 5 buckets in a row */
			printf("\n%02d: ", i);
		printf("%15lld ", histogram[i].d_cnt);
		total_cnt += histogram[i].d_cnt;
	  	/* we also want to make sure the total distance count is correct */
		if(i == num_buckets - 1)	
			printf("\n T:%lld \n", total_cnt);
		else printf("| ");
	}
}

void output_histogram2(){
	int i; 
	long long total_cnt = 0;

	for(i=0; i< num_buckets; i++) {
		if(i%5 == 0) /* we print 5 buckets in a row */
			printf("\n%02d: ", i);
		printf("%15lld ", d_to_h_histogram[i].d_cnt);
		total_cnt += d_to_h_histogram[i].d_cnt;
	  	/* we also want to make sure the total distance count is correct */
		if(i == num_buckets - 1)	
			printf("\n T:%lld \n", total_cnt);
		else printf("| ");
	}
}

/* histogram differences */
void comparison_histogram(){
	int i; 
	long long total_cnt = 0;
	for(i=0; i< num_buckets; i++) {
		if(i%5 == 0) /* we print 5 buckets in a row */
			printf("\n%02d: ", i);
		printf("%15lld ", histogram[i].d_cnt-d_to_h_histogram[i].d_cnt);
		total_cnt += histogram[i].d_cnt-d_to_h_histogram[i].d_cnt;
	  	/* we also want to make sure the total distance count is correct */
		if(i == num_buckets - 1)	
			printf("\n T:%lld \n", total_cnt);
		else printf("| ");
	}
}


int main(int argc, char **argv)
{
	int i;

	n_atoms = atoi(argv[1]);
	PDH_res	 = (double)atof(argv[2]);
	blockSize = atoi(argv[3]);

	num_buckets = (int)(BOX_SIZE * 1.732 / PDH_res) + 1;

	histogram = (bucket *)malloc(sizeof(bucket)*num_buckets);
	atom_list = (atom *)malloc(sizeof(atom)*n_atoms);

	
	srand(1);
	/* generate data following a uniform distribution */
	for(i = 0;  i < n_atoms; i++) {
		atom_list[i].x_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		atom_list[i].y_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		atom_list[i].z_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
	}

	/* start counting time */
	gettimeofday(&startTime, &Idunno);

	/* call CPU single thread version to compute the histogram */
	PDH_baseline();

	/* check the total running time */ 
	report_running_time();

	/* print out the histogram */
	printf("\nCPU results: \n");
	output_histogram();

	/* ---------------------------- GPU version ---------------------------- */

	d_histogram = (bucket *)malloc(sizeof(bucket)*num_buckets);
	d_to_h_histogram = (bucket *)malloc(sizeof(bucket)*num_buckets);
	d_atom_list = (atom *)malloc(sizeof(atom)*n_atoms);

	cudaMalloc((void**)&d_atom_list, sizeof(atom) * n_atoms);
	cudaMemcpy(d_atom_list, atom_list, sizeof(atom) * n_atoms, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&d_histogram, sizeof(bucket) * num_buckets);
	//cudaMemcpy(d_histogram, d_to_h_histogram,sizeof(bucket) * num_buckets, cudaMemcpyHostToDevice);
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	int no_of_blocks = ceil(n_atoms/(double)blockSize);
	
	kernel<<< no_of_blocks , blockSize, sizeof(bucket) * num_buckets + sizeof(atom) * blockSize>>>(n_atoms, d_histogram, PDH_res, d_atom_list);
	cudaDeviceSynchronize();

	cudaEventRecord(stop, 0); 
	cudaEventSynchronize(stop);
	float elapsedTime; 
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf( "Time to generate: %0.5f ms\n", elapsedTime );
	cudaEventDestroy(start); 
	cudaEventDestroy(stop); 
	
	cudaMemcpy(d_to_h_histogram, d_histogram, sizeof(bucket) * num_buckets, cudaMemcpyDeviceToHost);	
	/* print out the histogram */
	printf("\nGPU results: \n");
	output_histogram2();

	/* print out the difference between cpu and gpu results */
	printf("\nDIFFERENCES: \n");
	comparison_histogram();
	
	return 0;
}
