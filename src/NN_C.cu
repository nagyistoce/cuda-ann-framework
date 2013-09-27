
#include "NN_struct.h"


int main(void) {

	int devID = 0;
	cudaDeviceProp devProp;
	CUDA_CHECK_RETURN(cudaGetDevice(&devID));
	CUDA_CHECK_RETURN(cudaGetDeviceProperties(&devProp,devID));
	fprintf(stdout,"Using GPU %s \n\t Maximum %i Threads per Blocks clocked at %f GHz\n\t Compute capability %i,%i Detected\n",devProp.name,devProp.maxThreadsPerBlock,(((float)devProp.clockRate)/1048576.0f),devProp.major,devProp.minor);
	readIn();
	CUDA_CHECK_RETURN(cudaDeviceReset());
	fprintf(stdout, "Ending program !");
	return 0;
}

