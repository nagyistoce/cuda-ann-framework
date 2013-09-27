#include "../NN_struct.h"

/*****************************************************************************************************
 * 									Line Extractor
 *	DESCRIPTION : This function will hold on stdin until a complete CMD SENTENCE is formed
 *	NOTE : The return string should be freed after used !
 *
 ****************************************************************************************************/
void lineExt(float *ret){

	char **boite;
	size_t len = MAXLINEBUFF -1;
	char *token;

	boite = (char **)calloc(1,sizeof(char **));
	fprintf(stdout,"NN Waiting for input ...\n");
	getline(boite,&len,stdin);
	token = strtok(*boite,"; \n");
	if(token != NULL){
		//The first argument of the line should always be the CMD
		sscanf(token,"%f",&ret[0]);
		for(int i =0;i < (MAXLINEBUFF -1);i++){
			token = strtok(NULL,"; \n");
			if(token == NULL ){
					//fprintf(stdout,"\tINPUT didn't reach : EOI waiting for more inputs\n");
					//boite = (char **)calloc(1,sizeof(char **));
					getline(boite,&len,stdin);
					token = strtok(*boite,"; \n");
					if(token != NULL){
						if(strcmp(token,"EOI")==0){
							ret[i+1] = (int)NULL;
							fprintf(stdout,"\tCMD %i received and EOI reached, processing ...\n\n", (int)ret[0]);
							fflush(stdout);
							break;
						}
						else{
							sscanf(token,"%f",&ret[i+1]);
						}
					}
			}
			else{
				if(strcmp(token,"EOI")==0){
					ret[i+1] = (int)NULL;
					fprintf(stdout,"\tCMD %i received and EOI reached, processing ...\n", (int)ret[0]);
					fflush(stdout);
					break;
				}
				else{
					sscanf(token,"%f",&ret[i+1]);
				}
			}
		}
	}
	else{
		//should not be possible
		fprintf(stdout,"NULL TOKEN");
		fflush(stdout);
		ret[0]=(int)NULL;
	}
	free(boite);
}

void readIn(){
	mPtrVect *master=NULL;
	int layerBuff[MAXLAYER];
	float *lineConv = (float *)calloc(MAXLINEBUFF, sizeof(float));
	float *lineBuff = (float *)calloc(MAXLINEBUFF, sizeof(float));
	cudaEvent_t start, stop;
	float ms;
	CUDA_CHECK_RETURN(cudaEventCreate(&start));
	CUDA_CHECK_RETURN(cudaEventCreate(&stop));

	while(true){
		lineExt(lineConv);
		switch((int)lineConv[0]){
			case 0x01:
				//case init
				//instanciation string structure, 0x01;Number of layer;nbrInputL1;[Neurons L1;L2;L3;...;L Nbr of Layer];EOI

				//Extracted line verification
				if(lineConv[1] == (int)NULL | lineConv[2] == (int)NULL){
					fprintf(stdout," WRONG CMD FORMAT :\n \t Usage: \n \t\t 0x01;Number of layer;nbrInputL1;[Neurons L1;L2;L3;...;L Nbr of Layer];EOI\n");
					break;
				}
				else{
					for(int i=3; i < MAXLAYER + 3 ;i++){
						if(lineConv[i]!= (int)NULL){
							layerBuff[i-3] = (int)lineConv[i];
						}
						else{
							if(i-3 == lineConv[1]){
								master = (mPtrVect *)malloc(sizeof(mPtrVect));
								master->nbrlayer = (int)lineConv[1];
								master->nbrInputL1 = (int)lineConv[2];
								if(cmdMemAlloc(master,layerBuff)!=0){
									fprintf(stdout, "FAILED to initialize the network try again later ...\n");
									free(master);
									master=NULL;
								}

								break;
							}
							else{
								fprintf(stdout," WRONG CMD FORMAT :\n \t Usage: \n \t\t 0x01;Number of layer;nbrInputL1;[Neurons L1;L2;L3;...;L Nbr of Layer];EOI (2)\n");
								break;
							}
						}
					}
				}

				break;
			case 0x02:
				if(master != NULL){
					if((uint)lineConv[(uint)(master->hWVect[(uint)lineConv[1]]->cols*master->hWVect[(uint)lineConv[1]]->rows)+2]==(int)NULL){

						for(uint i=2;i < (uint)(master->hWVect[(uint)lineConv[1]]->cols*master->hWVect[(uint)lineConv[1]]->rows)+2;lineBuff[i-2]=lineConv[i],fprintf(stdout,"\tW%i,%i = %f\n",(uint)floor((i-2)/(uint)(master->hWVect[(uint)lineConv[1]]->cols)),(i-2)%(uint)(master->hWVect[(uint)lineConv[1]]->cols),lineBuff[i-2]),i++);
						setLayerWeight(master,(uint)lineConv[1],lineBuff);
						//verifyWeight((int)lineConv[1],master);
						fprintf(stdout,"CMD 2 SUCCEEDED : WEIGHT LAYER DATA ENTERED\n");

					}
					else
						fprintf(stdout,"CMD 2 FAILED : NULL not found -> WRONG NUMBER OF INPUTS ENTERED ON CMDLINE\n");
				}
				else
					fprintf(stdout,"CMD 2 FAILED : Master Grid not initialized\n");
				break;
			case 0x03:
				//Extract the subchain
				if(master != NULL){
					if((uint)lineConv[master->nbrInputL1+1]== (uint)NULL){
						for(int i=1;i < master->nbrInputL1+1;lineBuff[i-1]=lineConv[i],fprintf(stdout,"\n\tINPUT %i = %f\n",i,lineConv[i]),i++);
						setInput(master,lineBuff);
						//verifyInput(master);
						fprintf(stdout,"CMD 3 SUCCEEDED : INPUT LAYER DATA ENTERED\n");
					}
					else
						fprintf(stdout,"CMD 3 FAILED : NULL not found -> WRONG NUMBER OF INPUTS ENTERED ON CMDLINE\n");
				}
				else
					fprintf(stdout,"CMD 3 FAILED : Master Grid not initialized\n");
					break;
			case 0x04:
				//case NN compute
				if(master != NULL){
					CUDA_CHECK_RETURN(cudaEventRecord(start,NULL));
					computeNet(master);
					CUDA_CHECK_RETURN(cudaEventRecord(stop,NULL));
					CUDA_CHECK_RETURN(cudaEventSynchronize(stop));
					CUDA_CHECK_RETURN(cudaEventElapsedTime(&ms, start, stop));
					for(int i =0;i<master->NeurPerLayer[master->nbrlayer -1];fprintf(stdout,"\n\t Out nbr %i = %f \n",i+1,((float *)((I_Grid *)(master->hout))->st_ptr)[i]),i++);
					fprintf(stdout,"\nNetwork computed in %f μs\n\n",ms*1000);
					fprintf(stdout,"\t\tCMD4 SUCCEEDED : NEURAL GRID COMPUTED\n");
				}
				else
					fprintf(stderr,"\tCMD4 FAILED : NEURAL GRID NOT INITIALIZED ! \n");
				break;
			case 0x05:
				//case NN train NOT YET TO BE IMPLEMENTED
				break;
			case 0x06:
				//terminate exec, deallocate mem.
				cmdMemFree(master);
				break;
			case 0x07:
				//usage
				usage();
				break;
			case 0x08:
				cmdMemFree(master);
				fprintf(stdout,"TERMINATING, Have a nice day ;-)");
				exit(0);
				break;
			case 0x09:
				if(master != NULL){
					optimizeBTRatio(master);
				}
				break;
			case 0x0A:
				if(master != NULL){
					fprintf(stdout,"\nVerifying input layer ...\n\n ");
					verifyInput(master);
					for(int i =0;i<master->nbrlayer;fprintf(stdout,"\nVerifying weight layer %i... \n\n",i),verifyWeight(i,master),i++);
				}
				break;
			default:
					fprintf(stdout,"\n\tDefault CMD 0x00 or UNK CMD\n");
				break;
		}
	}
}

