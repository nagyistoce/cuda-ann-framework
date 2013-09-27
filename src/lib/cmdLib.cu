#include "../NN_struct.h"

/****************************************************************************************************
 * 										CMDMemAlloc
 * Description: This function instantiate a Neural network memory Grid from specified parameter
 *
 * Note : The maximum number of layer is MAXLAYER, most neural network only use 3 or 4 ...
 *
 *****************************************************************************************************/
int cmdMemAlloc(mPtrVect *ptr, int neurPerLayer[MAXLAYER] ) {

	int ret = 0;

	ptr->dInVect = (I_Grid **)malloc(sizeof(I_Grid)*ptr->nbrlayer);
		if(ptr->dInVect == NULL)
				ret = 1;
	ptr->dWVect	= (W_Grid **)malloc(sizeof(W_Grid)*ptr->nbrlayer);
		if(ptr->dWVect == NULL)
				ret = 2;
	ptr->hInVect = (I_Grid **)malloc(sizeof(I_Grid)*ptr->nbrlayer);
		if(ptr->hInVect == NULL)
				ret = 3;
	ptr->hWVect  = (W_Grid **)malloc(sizeof(W_Grid)*ptr->nbrlayer);
		if(ptr->hWVect == NULL)
				ret = 4;
	ptr->NeurPerLayer = (int *)malloc(sizeof(int)*ptr->nbrlayer);
		if(ptr->NeurPerLayer == NULL)
				ret = 5;

	for(int i = 0; i<ptr->nbrlayer;i++){
		fprintf(stdout,"layer %i\n \t with %i Neurons\n",i,neurPerLayer[i]);
		fflush(stdout);

		ptr->NeurPerLayer[i] = neurPerLayer[i];

		//create weight and input grid for each layer for device (dW, dI) and host (hW, hI)

		if(i==0){
			//The struct as per say (couple pointer and int )
			CUDA_CHECK_RETURN(cudaHostAlloc((void **) &(ptr->hWVect[i]),sizeof(W_Grid),0));
			CUDA_CHECK_RETURN(cudaHostAlloc((void **) &(ptr->hInVect[i]),sizeof(I_Grid),0));

			//The layer grid pointed by st_ptr
			CUDA_CHECK_RETURN(cudaHostAlloc((void **) &(ptr->hWVect[i]->st_ptr), sizeof(float)*ptr->nbrInputL1*ptr->NeurPerLayer[i],0));
			CUDA_CHECK_RETURN(cudaHostAlloc((void **) &(ptr->hInVect[i]->st_ptr),sizeof(float)*ptr->nbrInputL1,0));

			//While we iterating might as well
			ptr->hWVect[i]->cols = ptr->nbrInputL1;
			ptr->hWVect[i]->rows = ptr->NeurPerLayer[i];
			ptr->hInVect[i]->nbr = ptr->nbrInputL1;

			//The structure for the device
			CUDA_CHECK_RETURN(cudaHostAlloc((void **) &(ptr->dWVect[i]),sizeof(W_Grid),0));
			CUDA_CHECK_RETURN(cudaHostAlloc((void **) &(ptr->dInVect[i]),sizeof(I_Grid),0));

			//The layer Grid pointed by st_ptr

			CUDA_CHECK_RETURN(cudaMalloc((void **) &((ptr->dWVect[i])->st_ptr),ptr->nbrInputL1*sizeof(float)*ptr->NeurPerLayer[i]));
			fprintf(stdout,"WGrid %i size is %i x %i\n",i,ptr->nbrInputL1,ptr->NeurPerLayer[i]);
			fflush(stdout);
			CUDA_CHECK_RETURN(cudaMalloc((void **) &(ptr->dInVect[i]->st_ptr),sizeof(float)*ptr->nbrInputL1));
			fprintf(stdout,"IGrid %i size is %i\n",i,ptr->nbrInputL1);
			fflush(stdout);
		}
		else{
			CUDA_CHECK_RETURN(cudaHostAlloc((void **) &(ptr->hWVect[i]),sizeof(W_Grid),0));
			CUDA_CHECK_RETURN(cudaHostAlloc((void **) &(ptr->hInVect[i]),sizeof(I_Grid),0));

			CUDA_CHECK_RETURN(cudaHostAlloc((void **) &(ptr->hWVect[i]->st_ptr),sizeof(float)*ptr->NeurPerLayer[i-1]*ptr->NeurPerLayer[i],0));
			CUDA_CHECK_RETURN(cudaHostAlloc((void **) &(ptr->hInVect[i]->st_ptr),sizeof(float)*ptr->NeurPerLayer[i-1],0));

			ptr->hWVect[i]->cols = ptr->NeurPerLayer[i-1];
			ptr->hWVect[i]->rows = ptr->NeurPerLayer[i];
			ptr->hInVect[i]->nbr = ptr->NeurPerLayer[i-1];

			CUDA_CHECK_RETURN(cudaHostAlloc((void **) &(ptr->dWVect[i]),sizeof(W_Grid),0));
			CUDA_CHECK_RETURN(cudaHostAlloc((void **) &(ptr->dInVect[i]),sizeof(I_Grid),0));

			//CUDA_CHECK_RETURN(cudaMallocPitch((void **) &(ptr->dWVect[i]->st_ptr),&(ptr->dWVect[i]->p),ptr->NeurPerLayer[i-1]*sizeof(float),neurPerLayer[i]));
			CUDA_CHECK_RETURN(cudaMalloc((void **) &((ptr->dWVect[i])->st_ptr),ptr->NeurPerLayer[i-1]*sizeof(float)*ptr->NeurPerLayer[i]));
			fprintf(stdout,"WGrid %i size is %i x %i\n",i,ptr->NeurPerLayer[i-1],ptr->NeurPerLayer[i]);
			fflush(stdout);
			CUDA_CHECK_RETURN(cudaMalloc((void **) &(ptr->dInVect[i]->st_ptr),sizeof(float)*ptr->NeurPerLayer[i-1]));
			fprintf(stdout,"IGrid %i size is %i\n",i,ptr->NeurPerLayer[i-1]);
			fflush(stdout);
		}
	}

	//Create the output grid
	CUDA_CHECK_RETURN(cudaHostAlloc((void **) &(ptr->hout),sizeof(I_Grid),0));
	CUDA_CHECK_RETURN(cudaHostAlloc((void **) &(((I_Grid *)(ptr->hout))->st_ptr),sizeof(float)*ptr->NeurPerLayer[ptr->nbrlayer - 1],0));
	CUDA_CHECK_RETURN(cudaHostAlloc((void **) &(ptr->dout),sizeof(I_Grid),0));
	CUDA_CHECK_RETURN(cudaMalloc((void **) &(((I_Grid *)(ptr->dout))->st_ptr),sizeof(float)*ptr->NeurPerLayer[ptr->nbrlayer - 1]));
	ptr->computeProp = NULL;
	fprintf(stdout,"\nOutput Grid has %i entries\n\n",ptr->NeurPerLayer[ptr->nbrlayer - 1]);
	ret != 0?fprintf(stderr,"\t FAILED TO INSTANTIATE THE NEURAL MEMORY GRID CMD 1 : %i return value\n",ret):fprintf(stdout,"\t CMD 1 SUCEEDED : NEURAL MEMORY GRID INSTANTIATED \n");
	return(ret);
}

int cmdMemFree(mPtrVect *master){
	if(master != NULL){
		fprintf(stdout,"\n\t\tFreeing memory ...\n");
		for(int i =0;i<master->nbrlayer;i++){
			cudaFree(((W_Grid *)master->dWVect[i])->st_ptr);
			cudaFree(master->dWVect[i]);
			cudaFree(((I_Grid *)master->dInVect[i])->st_ptr);
			cudaFree(master->dInVect[i]);
			cudaFreeHost(((I_Grid *)(master->hInVect[i]))->st_ptr);
			cudaFreeHost(master->hInVect[i]);
			cudaFreeHost(master->hWVect[i]);
		}
		cudaFree(((I_Grid *)master->dout)->st_ptr);
		cudaFree(master->dout);
		cudaFreeHost(((I_Grid *)master->hout)->st_ptr);
		cudaFreeHost(master->hout);
		free(master->dInVect);
		free(master->dWVect);
		free(master->hInVect);
		free(master ->hWVect);
		free(master->computeProp);
		free(master);
		master = NULL;
		fprintf(stdout,"\n\t\tNeural Grid deallocated \n");
		return(0);
	}
	else
		fprintf(stdout,"\n\t\tMaster Memory Grid not allocated -> Skipping memory free\n");
		return(1);
}

void usage(){

	fprintf(stdout,"\n\t\t--------------- USAGE for CNCF ---------------\n");
	fprintf(stdout,"\t\t----- The cuda neural computing framework ----\n");
	fprintf(stdout,"\t\t----------------------------------------------\n");

	fprintf(stdout,"\n------------------------------------------------------------------------------\n");
	fprintf(stdout,"CMD 0x01\n");
	fprintf(stdout,"\t Initialise net : 0x01;NL;nbrInputL1;[Neurons L1;L2;L3;...;L#];EOI\n\n");
	fprintf(stdout,"\t\tWhere -> NL is the number of layers\n");
	fprintf(stdout,"\t\tWhere nbrInputL1 is the number of inputs of the neural grid \n");
	fprintf(stdout,"\t\tWhere -> L# is the number of neurons on the layer #\n");
	fprintf(stdout,"\n------------------------------------------------------------------------------\n");
	fprintf(stdout,"CMD 0x02\n");
	fprintf(stdout,"\tSet Weights: 0x02;L;w(L,1);w(L,2);[...];w(L,n) \n\n");
	fprintf(stdout,"\t\tWhere -> n is the number of neurons on the layer L\n");
	fprintf(stdout,"\n------------------------------------------------------------------------------\n");
	fprintf(stdout,"CMD 0x03\n");
	fprintf(stdout,"\tSet Input: 0x03;Input1;Input2;[...];Input[nbrInput];EOI\n\n");
	fprintf(stdout,"\n------------------------------------------------------------------------------\n");
	fprintf(stdout,"CMD 0x04\n");
	fprintf(stdout,"\tThis will forward propagate the values of the inputs through the network\n");
	fprintf(stdout,"\tNN compute: 0x04;EOI\n\n");
	fprintf(stdout,"\n------------------------------------------------------------------------------\n");
	fprintf(stdout,"CMD 0x05\n");
	fprintf(stdout,"\n------------------------------------------------------------------------------\n");
	fprintf(stdout,"CMD 0x06\n");
	fprintf(stdout,"\tThis will free the allocated memory for the NN\n");
	fprintf(stdout,"\tNN memFree: 0x06;EOI\n\n");
	fprintf(stdout,"\n------------------------------------------------------------------------------\n");
	fprintf(stdout,"CMD 0x07\n");
	fprintf(stdout,"\tThis will show *THIS usage\n");
	fprintf(stdout,"\tCNCF usage: 0x07;EOI\n\n");
	fprintf(stdout,"\n------------------------------------------------------------------------------\n");
	fprintf(stdout,"CMD 0x08\n");
	fprintf(stdout,"\tThis will terminate the NN properly\n");
	fprintf(stdout,"\tNN TERM: 0x08;EOI\n\n");
	fprintf(stdout,"\n------------------------------------------------------------------------------\n");
	fprintf(stdout,"CMD 0x09\n");
	fprintf(stdout,"\tThis will modify the thread/block ratio for cuda kernel call IOT optimize it\n");
	fprintf(stdout,"\n\t\tWARNING : The network must be initialized\n");
	fprintf(stdout,"\tNN OP_BT: 0x09;EOI\n\n");
	fprintf(stdout,"\n------------------------------------------------------------------------------\n");
	fprintf(stdout,"CMD 0x0A\n");
	fprintf(stdout,"\tThis will output values of the layers\n");
	fprintf(stdout,"\tNN VERIF_W: 0x0A;EOI\n\n");
	fprintf(stdout,"\n------------------------------------------------------------------------------\n");

}

