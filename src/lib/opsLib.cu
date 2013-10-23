#include "../NN_struct.h"

/*****************************************************************************************************************************
 * 									CreateNeuronGrid
 * 	function : This function will create a MEMORY space on the device with the data supplied by the W_Grid structure
 * 	that has been given to her
 * 	Note : 1 - The pinned memory here is of very small size and should not degrade performance since it could be called once
 *  	   	the creation of neural network. For large grid of mixed neural net I do not recommend using this function another
 *  		implementation might be required or programmer can free memory storing is pointer elsewhere.
 *  	   2 - Should be considered to free the memory of the memStr after the function returned and proper use has been made
 *  	   	of its content
 *  	   3 - WARNING ---- The programmer should check for error in the memPtr before using the memPtr -----
 *  	   4 - THIS STRUCTURE REPRESENT A LAYER OF WEIGHT IN THE NN
 *****************************************************************************************************************************/
void createWeightGrid(W_Grid *grid,W_Grid **wgridout){
	//allocate space for the struct
	cudaHostAlloc(&wgridout[0], sizeof(grid),0);
	//allocate memory on the device
	cudaMallocPitch(&(wgridout[1]->st_ptr), &(wgridout[1]->p),grid->cols* sizeof(float),grid->rows );
	//fill the memory location with the data of the struct that has been passed to the function
	cudaMemcpy2D(wgridout[1]->st_ptr,wgridout[1]->p,grid->st_ptr,grid->p,grid->cols* sizeof(float),grid->rows,cudaMemcpyHostToDevice);
}
/*****************************************************************************************************************************
 * 									WriteWeight2Device
 * 	Function: This function will write a weight vector on the device
 * 	Note : 1 - The programmer should be aware of the size of the NN to do so (Bah no joking ...)
 * 		   2 - The programmer should check for error in the memPtr struct before assuming ... ( No joking again )
 *****************************************************************************************************************************/
void *writeWeight2Device(W_Grid *DMemPtr, W_Grid *grid){
	CUDA_CHECK_RETURN(cudaMemcpy2D(DMemPtr->st_ptr,DMemPtr->p,grid->st_ptr,grid->p,grid->cols* sizeof(float),grid->rows,cudaMemcpyHostToDevice));
		//return the pointer of the memory location
		return((void *)DMemPtr);
}
/*****************************************************************************************************************************
 * 									CreateINPUTGrid
 * 	function : This function will create a MEMORY space on the device with the data supplied by the W_Grid structure
 * 	that has been given to her
 * 	Note : 1 - The pinned memory here is of very small size and should not degrade performance since it could be called once
 *  	   	the creation of neural network. For large grid of mixed neural net I do not recommend using this function another
 *  		implementation might be required or programmer can free memory storing is pointer elsewhere.
 *  	   2 - Should be considered to free the memory of the memStr after the function returned and proper use has been made
 *  	   	of its content
 *  	   3 - WARNING ---- The programmer should check for error in the memPtr before using the memPtr -----
 *  	   4 - THIS STRUCTURE REPRESENT THE INPUT VECTOR IN THE NN
 *****************************************************************************************************************************/
void *createInputGrid(I_Grid *grid, I_Grid **ptrset){
	I_Grid *hmemPtr, *dmemPtr;
	//allocate space for the struct on the host
	//memory location is pinned since there will be transfert all the time with that location
	cudaHostAlloc(&hmemPtr, sizeof(I_Grid),0);
	cudaHostAlloc(&(hmemPtr->st_ptr),sizeof(float)*grid->nbr,0);
	//copy the struct
	cudaMemcpy(hmemPtr, grid,sizeof(hmemPtr), cudaMemcpyHostToHost);
	//copy the data pointed by the struct
	cudaMemcpy(hmemPtr->st_ptr,grid->st_ptr,sizeof(float)*grid->nbr,cudaMemcpyHostToHost);
	//allocate memory on the device for the struct
	CUDA_CHECK_RETURN(cudaMalloc(&dmemPtr,sizeof(dmemPtr)));
	//allocate the memory on the device for the data pointed by the struct
	CUDA_CHECK_RETURN(cudaMalloc(&(dmemPtr->st_ptr),sizeof(float)*hmemPtr->nbr));
	//fill the device memory location with the struct
	CUDA_CHECK_RETURN(cudaMemcpy(dmemPtr,hmemPtr,sizeof(hmemPtr),cudaMemcpyHostToDevice));
	//fill the memory with the data itself
	CUDA_CHECK_RETURN(cudaMemcpy(dmemPtr->st_ptr, hmemPtr->st_ptr, sizeof(float)*hmemPtr->nbr,cudaMemcpyHostToDevice));
	ptrset[0] = hmemPtr;
	ptrset[1] = dmemPtr;
	return((void *)ptrset);
}


void *createOutput(int nbrOut, I_Grid **ptrSet){
	I_Grid *doutVect, *houtVect;
	//allocate space for the struct on the host
	//memory location is pinned since there will be transfert all the time with that location
	cudaHostAlloc(&houtVect,sizeof(I_Grid),0);
	cudaHostAlloc(&(houtVect->st_ptr),sizeof(float)*nbrOut,0);
	houtVect->nbr = nbrOut;
	//allocate memory on the device for the struct
	CUDA_CHECK_RETURN(cudaMalloc(&doutVect,sizeof(I_Grid)));
	//copy the struct data to device
	CUDA_CHECK_RETURN(cudaMemcpy(doutVect,houtVect,sizeof(I_Grid),cudaMemcpyHostToDevice));
	//allocate the memory on the device for the data pointed by the struct
	CUDA_CHECK_RETURN(cudaMalloc(&(doutVect->st_ptr),sizeof(float)*nbrOut));
	//fill the device memory location with the struct

	ptrSet[0]=houtVect;
	ptrSet[1]=doutVect;
	return((void *)ptrSet);
}

void *writeInput2Host(I_Grid *inPtr, I_Grid *hmemPtr){
	//copy the data pointed by the struct
	CUDA_CHECK_RETURN(cudaMemcpy(hmemPtr->st_ptr,inPtr->st_ptr,sizeof(float)*inPtr->nbr,cudaMemcpyHostToHost));
	return(hmemPtr);
}
/*********************************************************************************************************************
 * 									WriteInput2Device
 *	Function: This function will write an INPUT vector on the device
 * 	Note : 1 - The programmer should be aware of the size of the NN to do so (Bah no joking ...)
 * 		   2 - The programmer should check for error in the memPtr struct before assuming ... ( No joking again )
 **********************************************************************************************************************/

void *writeInput2Device(void *dmemPtr, void *hmemPtr){
	CUDA_CHECK_RETURN(cudaMemcpy(((I_Grid *)dmemPtr)->st_ptr, ((I_Grid *)hmemPtr)->st_ptr, sizeof(float)*((I_Grid *)hmemPtr)->nbr,cudaMemcpyHostToDevice));
		//return the pointer of the memory location
	return(dmemPtr);
}






/**********************************************************************************************************************************
 * 												ComputeNet
 *
 * Description : Compute the neural net by launching a kernel
 *
 **********************************************************************************************************************************/

void computeNet(mPtrVect *master){
	cudaEvent_t start, stop;
	float ms;
	//float *test;
	dim3 numB,numT;
	CUDA_CHECK_RETURN(cudaEventCreate(&start));
	CUDA_CHECK_RETURN(cudaEventCreate(&stop));
	//Forward propagate input in the net
	for(int i =0; i<master->nbrlayer;i++){
		//unless its the last output link the input layer with the previous output
		fprintf(stdout,"Propagating layer %i\n",i);
		if(i<( master->nbrlayer -1 )){
			CUDA_CHECK_RETURN(cudaEventRecord(start,NULL));

			if(master->computeProp != NULL){
				numB.x = master->computeProp[i];
				numT.x =(int)ceil((float)master->NeurPerLayer[i]/master->computeProp[i]);
			}
			else{
				numB.x = 1;
				numT.x = master->NeurPerLayer[i];
			}
			calcLayerOut<<<numB,numT>>>(master->dWVect[i]->st_ptr,master->dInVect[i]->st_ptr,master->dInVect[i+1]->st_ptr,master->hWVect[i]->cols,master->NeurPerLayer[i]);
			CUDA_CHECK_RETURN(cudaDeviceSynchronize());
			CUDA_CHECK_RETURN(cudaEventRecord(stop,NULL));
			CUDA_CHECK_RETURN(cudaEventSynchronize(stop));
			CUDA_CHECK_RETURN(cudaEventElapsedTime(&ms, start, stop));
			fprintf(stdout,"Layer %i computed in %f ��s using %i Blocks\n",i,ms*1000,numB.x);
		}
		else{

			if(master->computeProp != NULL){
				numB.x = master->computeProp[i];
				numT.x =(int)ceil((float)master->NeurPerLayer[i]/master->computeProp[i]);
			}
			else{
				numB.x = 1;
				numT.x = master->NeurPerLayer[i];
			}
			CUDA_CHECK_RETURN(cudaEventRecord(start,NULL));

			//-----------------------------------------------------------------------//
			//****************WARNING HOST TESTING CODE *****************************//
			//----------------Should stay commented unless checking timings----------//
			//----------------NOT COMPLETED WILL NOT COMPUTE PROPERLY ---------------//
//
//			CUDA_CHECK_RETURN(cudaMemcpy(master->hWVect[i]->st_ptr,master->dWVect[i]->st_ptr,sizeof(float)*master->NeurPerLayer[i]*master->hWVect[i]->cols, cudaMemcpyDeviceToHost));
//			CUDA_CHECK_RETURN(cudaMemcpy(master->hInVect[i]->st_ptr,master->dInVect[i]->st_ptr,sizeof(float)*master->hWVect[i]->cols, cudaMemcpyDeviceToHost));
//
//			test = (float *)malloc(sizeof(float)*master->NeurPerLayer[i]);
//
//			for(int k =0;k<master->NeurPerLayer[k];k++){
//				for(int j;j<master->hWVect[i]->cols;j++){
//					test[k]+= ((float *)master->dWVect[i]->st_ptr)[j+i*master->NeurPerLayer[k]]
//				}
//			}
//
//			free(test);
			//---------------------------------------------------------------------//
			calcLayerOut<<<numB,numT>>>(master->dWVect[i]->st_ptr,master->dInVect[i]->st_ptr,((I_Grid *)(master->dout))->st_ptr,master->hWVect[i]->cols,master->NeurPerLayer[i]);
			CUDA_CHECK_RETURN(cudaDeviceSynchronize());
			CUDA_CHECK_RETURN(cudaEventRecord(stop,NULL));
			CUDA_CHECK_RETURN(cudaEventSynchronize(stop));
			CUDA_CHECK_RETURN(cudaEventElapsedTime(&ms, start, stop));
			fprintf(stdout,"OUT layer computed in %f ��s\n",ms*1000);
		}
	}
	CUDA_CHECK_RETURN(cudaMemcpy(((I_Grid *)(master->hout))->st_ptr, ((I_Grid *)(master->dout))->st_ptr, sizeof(float) * master->NeurPerLayer[master->nbrlayer -1], cudaMemcpyDeviceToHost));
}

void setInput(mPtrVect *master,float* inptr){
	CUDA_CHECK_RETURN(cudaMemcpy(master->hInVect[0]->st_ptr,inptr,sizeof(float)*master->nbrInputL1,cudaMemcpyHostToHost));
	CUDA_CHECK_RETURN(cudaMemcpy(master->dInVect[0]->st_ptr,master->hInVect[0]->st_ptr,sizeof(float)*master->nbrInputL1,cudaMemcpyHostToDevice));
	//-------------------------------------//
	//Debug loop
	//for(int i =0;i<master->nbrInputL1;fprintf(stdout,"hI = %f",((float *)(master->hInVect[0]->st_ptr))[i]),i++);
	//------------------------------------//
	fprintf(stdout,"Input vector set on device; ready to compute\n");
}

void verifyInput(mPtrVect *master){
	float boite[master->nbrInputL1];
	cudaMemcpy(boite,master->dInVect[0]->st_ptr,sizeof(float)*master->nbrInputL1,cudaMemcpyDeviceToHost);
	for(int i =0; i<master->nbrInputL1;fprintf(stdout,"\tChecking entry %i -> %f \n",i,boite[i]),i++);
}

void setLayerWeight(mPtrVect *master, int layer, float *data){

	CUDA_CHECK_RETURN(cudaMemcpy(master->hWVect[layer]->st_ptr,data,sizeof(float)*master->hWVect[layer]->cols*master->hWVect[layer]->rows,cudaMemcpyHostToHost));
	CUDA_CHECK_RETURN(cudaMemcpy(master->dWVect[layer]->st_ptr,master->hWVect[layer]->st_ptr,sizeof(float)*master->hWVect[layer]->cols*master->hWVect[layer]->rows,cudaMemcpyHostToDevice));
	//---------------------------------------------//
	//Debug loop
	//for(int i = 0;i<master->hWVect[layer]->cols*master->hWVect[layer]->rows;fprintf(stdout,"w=%f",((float *)(master->hWVect[layer]->st_ptr))[i]),i++);
	//---------------------------------------------//
	fprintf(stdout,"\tWeight set on layer %i\n",layer);
}

void verifyWeight(int layer, mPtrVect *master){
	float boite[master->NeurPerLayer[layer]*master->hWVect[layer]->cols*master->hWVect[layer]->rows];
	cudaMemcpy(boite,master->dWVect[layer]->st_ptr,master->hWVect[layer]->cols*sizeof(float)*master->hWVect[layer]->rows,cudaMemcpyDeviceToHost);
	for(int i = 0; i<master->hWVect[layer]->cols*master->hWVect[layer]->rows;fprintf(stdout,"\tChecking entry %i -> %f\n",i,boite[i]),i++);
}

void optimizeBTRatio(mPtrVect *master){
	//Launch multiple computation to test response time depending on Block/threads ratio
	//This is done per layer since layer computation launch different kernels with different ratios
	cudaEvent_t starttest, stoptest;
	float ms,acc;
	dim3 numB(1,1,1);
	dim3 numT(1,1,1);
	CUDA_CHECK_RETURN(cudaEventCreate(&starttest));
	CUDA_CHECK_RETURN(cudaEventCreate(&stoptest));
	int devID = 0;
	cudaDeviceProp devProp;
	//Getting device properties
	CUDA_CHECK_RETURN(cudaGetDevice(&devID));
	CUDA_CHECK_RETURN(cudaGetDeviceProperties(&devProp,devID));

	master->computeProp = (int *)calloc(master->nbrlayer,sizeof(int));
	(devProp.major < 2)?devID = 16:devID = 32;
	for(int i = 0; i< master->nbrlayer;i++){
		//testing power of 2 number of block starting at 1,2,4,8,16,32* (stop at 16 if fermi or under compute capability)
		if(master->NeurPerLayer[i] >= 32){
			//by default use one block
			master->computeProp[i]=1;
			for(int j=1;j<=devID;j++){
				//repeat 100 times to ensure average results
				numB.x=j;
				numT.x=(int)ceil((float)master->NeurPerLayer[i]/j);
				for(int e=0;e<100;e++){
					if(i< (master->nbrlayer -1)){
						CUDA_CHECK_RETURN(cudaEventRecord(starttest,NULL));
						//the ceiling function ensure enough threads are started to cover the range
						// the float cast are only to ensure correct approx of the ceil() func
						calcLayerOut<<<numB,numT>>>(master->dWVect[i]->st_ptr,master->dInVect[i]->st_ptr,master->dInVect[i+1]->st_ptr,master->hWVect[i]->cols,master->NeurPerLayer[i]);
						CUDA_CHECK_RETURN(cudaDeviceSynchronize());
						CUDA_CHECK_RETURN(cudaEventRecord(stoptest,NULL));
						CUDA_CHECK_RETURN(cudaEventSynchronize(stoptest));
						CUDA_CHECK_RETURN(cudaEventElapsedTime(&ms, starttest, stoptest));
						j==1?acc=ms:(acc<ms?master->computeProp[i]=j:ms=0);
						//fprintf(stdout,"testing %i block on layer %i\n",j,i);
					}
					else{
						CUDA_CHECK_RETURN(cudaEventRecord(starttest,NULL));
						calcLayerOut<<<numB,numT>>>(master->dWVect[i]->st_ptr,master->dInVect[i]->st_ptr,((I_Grid *)(master->dout))->st_ptr,master->hWVect[i]->cols,master->NeurPerLayer[i]);
						CUDA_CHECK_RETURN(cudaDeviceSynchronize());
						CUDA_CHECK_RETURN(cudaEventRecord(stoptest,NULL));
						CUDA_CHECK_RETURN(cudaEventSynchronize(stoptest));
						CUDA_CHECK_RETURN(cudaEventElapsedTime(&ms, starttest, stoptest));
						j==1?acc=ms:(acc<ms?master->computeProp[i]=j:ms=0);
					}
				}
			}
			fprintf(stdout,"The optimum configuration for layer %i is using %i block and computed in %f\n",i,master->computeProp[i],acc);
		}
		else{
			//if there is not enough neurons just use one block ----
			master->computeProp[i]=1;
			fprintf(stdout,"Not enough neurons on layer %i using default config to 1\n",i);
		}
	}

}

void randomInitW(mPtrVect *master){
	float *data;
	for(int i = 0; i<master->nbrlayer;i++){
		data = (float *)malloc(sizeof(float)*master->hWVect[i]->cols*master->hWVect[i]->rows);
		for(int j= 0; j<master->hWVect[i]->cols;j++){
				for(int e = 0 ;e<master->hWVect[i]->rows;data[j+e*master->hWVect[i]->rows]= (float)(rand()%10000)/10000,e++);
		}
		CUDA_CHECK_RETURN(cudaMemcpy(master->hWVect[i]->st_ptr,data,sizeof(float)*master->hWVect[i]->cols*master->hWVect[i]->rows,cudaMemcpyHostToHost));
		CUDA_CHECK_RETURN(cudaMemcpy(master->dWVect[i]->st_ptr,master->hWVect[i]->st_ptr,sizeof(float)*master->hWVect[i]->cols*master->hWVect[i]->rows,cudaMemcpyHostToDevice));
		free(data);
	}

}

void randomInitI(mPtrVect *master){
	float *data=(float *)malloc(sizeof(float)*master->nbrInputL1);
	for(int j=0; j<master->hInVect[0]->nbr;j++){
			data[j] = (float)(rand()%10000)/10000;
	}
	CUDA_CHECK_RETURN(cudaMemcpy(master->hInVect[0]->st_ptr,data,sizeof(float)*master->nbrInputL1,cudaMemcpyHostToHost));
	CUDA_CHECK_RETURN(cudaMemcpy(master->dInVect[0]->st_ptr,master->hInVect[0]->st_ptr,sizeof(float)*master->nbrInputL1,cudaMemcpyHostToDevice));
	free(data);
}

void backProp(mPtrVect *master, float *err){

	float *buff;
	dim3 numB(1,1,1);
	dim3 numT(1,1,1);

	if(master->computeProp != NULL){
		numB.x = master->computeProp[master->nbrlayer-1];
		numT.x =(int)ceil((float)master->NeurPerLayer[master->nbrlayer-1]/master->computeProp[master->nbrlayer-1]);
	}
	else{
		numB.x = 1;
		numT.x = master->NeurPerLayer[master->nbrlayer-1];
	}
	//Starting kernel correct weight for each neurons on last layer
	CUDA_CHECK_RETURN(cudaMalloc(&buff,sizeof(float)*master->hWVect[master->nbrlayer-1]->cols));
	backPropW<<<numB,numT>>>(master->dWVect[master->nbrlayer-1]->st_ptr,master->dInVect[master->nbrlayer-1]->st_ptr,((I_Grid *)(master->dout))->st_ptr,buff, master->hWVect[master->nbrlayer-1]->cols,master->NeurPerLayer[master->nbrlayer-1],err);
	CUDA_CHECK_RETURN(cudaFree(buff));
	//Computing error vect for next layer


}

