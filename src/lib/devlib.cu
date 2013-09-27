#include "../NN_struct.h"

//the accumulator should be the size of sizeof(float)*col
__global__ void backPropW(void *wGrid, void *iVect, void *outVect,void *acc, int col, int nbr, float *err){
	iwMul(wGrid, iVect,outVect,col,nbr);
	__syncthreads();
	correctW(outVect,wGrid,iVect,acc,col,nbr,err);

}
// correct the weight vector
__device__ void correctW(void *iwMult,void* wGrid,void *iVect,void *acc,int col,int nbr,float *err){
	int index = index = blockIdx.x * blockDim.x + threadIdx.x;
	if(index < nbr){
		for(int e =0;e<col;e++){
			((float *)acc)[e]=((float *)iwMult)[index]*(1-((float *)iwMult)[index])*((float *)iVect)[e];
			((float *)wGrid)[e+col*index]+=((float *)acc)[e]*err[e];
			__syncthreads();
		}
	}
}


//at this point the outvect should have been overwritten by iwMult and passed as an arg
//the error vector should be produced by the correctW function call
__device__ void correctE(void *iwMult,void *iVect ,void *errVect,int col, int nbr ){
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if(index < nbr){
		((float *)iVect)[index] = 0.0f;
		for(int e =0;e<col;e++){
			((float *)iVect)[index] += ((float *)errVect)[index + e*col];
		}
	}
}

/*********************************************************************************************************************
 * 									calcLayer
 * 	Function: This function will create the output response from a layer and return the pointer
 * 	Note :  	1 - for each neuron the function to be calculated equate to f (multiplication of matrix + w0)
 * 				2 - the input vector will always be unidimentional  of size N and the weight vector always a MxN matrix
 * 					where N is the number of input in the input vector and M the amount of Neurons
 * 				3 - The resultant will always be a matrix of 1xM
**********************************************************************************************************************/

__global__ void calcLayerOut(void *wGrid, void *iVect, void *outVect, int col, int nbr){

	iwMul(wGrid, iVect,outVect,col,nbr);
	//ensure computation done before starting the sig calculation
	__syncthreads();
	sigmoidVect(outVect,nbr);
}


/*********************************************************************************************************************
*									sigmoidVect
*	Function : Use the expf() call to calculate the sigmoid result of each element in the iw vector
*	Note:		1 - ---- WARNING ------ Overwrite the original vector
*				2 - Should be called with at least as many threads than elements in the vector. [Surprised ?]
*				3 - The vector is located inside a memStr structure
**********************************************************************************************************************/
__device__ void *sigmoidVect(void *iwMult,int nbr){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if(index < nbr){
		((float *)iwMult)[index]=1/(1+expf(-((float *)iwMult)[index]));
		__syncthreads();
	}
	return(iwMult);
}
/*********************************************************************************************************************
* 									iwMul
* 	Function : This will produce the input/weight product and put the result in the target memStr structure
*
**********************************************************************************************************************/
__device__ void *iwMul(void *weightGrid, void *inputVector, void *targetGrid,int col,int nbr){
	//Generate the output vector to pass to sigmoidVect()
	//use x threads id correspond to the grid index
	//so for x=1 all y will accumulate in tgrid[1] etc ...
	float acc = 0;
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if( index < nbr){
#pragma unroll
		for(int e = 0; e<col;acc += ((float *)inputVector)[e]*((float *)weightGrid)[e+col*index],e++);
		//Don't need to sync here since every threads access only one element
		((float *)targetGrid)[index] = acc;
		__syncthreads();
	}
	return(targetGrid);
}
