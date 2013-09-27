#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <cuda_runtime.h>

#define MAXLAYER 	50
#define MAXLINEBUFF 2500

#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

typedef struct{
	void *st_ptr;
	size_t p;
	int cols;
	int rows;
}W_Grid;

typedef struct{
	void *st_ptr;
	int nbr;
}I_Grid;

typedef struct{
	void *loc;
	size_t p;
	cudaError_t err;
}memStr;

typedef struct{
	int nbrlayer;
	I_Grid **dInVect;
	I_Grid **hInVect;
	int nbrInputL1;
	W_Grid **dWVect;
	W_Grid **hWVect;
	int *NeurPerLayer;
	void *dout;
	void *hout;
	int *computeProp;

}mPtrVect;


/***************************/
/*		DEV LIB			   */
/***************************/
__device__ void *sigmoidVect(void *iwMult, int nbr);
__device__ void *iwMul(void *weightGrid, void *inputVector, void *targetGrid, int col,int nbr);
__device__ void correctW(void *iwMult,void* wGrid,void *iVect,void *acc,int col,int nbr,float *err);
__global__ void calcLayerOut(void *wGrid, void *iVect, void *targetGrid, int col, int nbr);
__global__ void backPropError(void *wGrid, void *iVect, void *outVect,void *acc, int col, int nbr, float *err);

/***************************/
/*		CMD LIB			   */
/***************************/
int cmdMemAlloc(mPtrVect *ptr, int neurPerLayer[MAXLAYER] );
int cmdMemFree(mPtrVect *master);
void usage();

/***************************/
/*		INPUT LIB		   */
/***************************/
void lineExt(float *ret);
void readIn();

/***************************/
/*		OPS LIB			   */
/***************************/
void *writeInput2Device(void *dmemPtr, void *hmemPtr);
void *writeWeight2Device(W_Grid *DMemPtr, W_Grid *grid);
void createWeightGrid(W_Grid *grid,W_Grid **wgridout);
void *createInputGrid(I_Grid *grid, I_Grid **ptrset);
void *writeInput2Host(I_Grid *inPtr, I_Grid *hmemPtr);
void setLayerWeight(mPtrVect *master, int layer, float *data);
void optimizeBTRatio(mPtrVect *master);
void randomInitW(mPtrVect *master);
void randomInitI(mPtrVect *master);
void computeNet(mPtrVect *master);
void setInput(mPtrVect *master,float* inptr);
void verifyInput(mPtrVect *master);
void verifyWeight(int layer, mPtrVect *master);
