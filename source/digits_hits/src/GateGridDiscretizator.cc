
/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

/*!
  \class  GridDiscretizator
  
  This module allows to simulate the readout of strip and pixelated detectors
  
  Last modification (Adaptation to GND): June 2023 by Mohamed-Jordan Soumano mjsoumano@yahoo.com
*/

#include "GateGridDiscretizator.hh"
#include "GateGridDiscretizatorMessenger.hh"
#include "GateDigi.hh"

#include "GateDigitizerMgr.hh"

#include "G4SystemOfUnits.hh"
#include "G4EventManager.hh"
#include "G4Event.hh"
#include "G4SDManager.hh"
#include "G4DigiManager.hh"
#include "G4ios.hh"
#include "G4UnitsTable.hh"
#include "GateVolumeID.hh"
#include "GateConfiguration.h"
#include "GateObjectStore.hh"
#include "GateConstants.hh"
#include "GateTools.hh"

GateGridDiscretizator::GateGridDiscretizator(GateSinglesDigitizer *digitizer, G4String name)
  :GateVDigitizerModule(name,"digitizerMgr/"+digitizer->GetSD()->GetName()+"/SinglesDigitizer/"+digitizer->m_digitizerName+"/"+name,digitizer,digitizer->GetSD()),
   m_GateGridDiscretizator(0),
   m_outputDigi(0),
   m_OutputDigiCollection(0),
   m_digitizer(digitizer)
 {
	G4String colName = digitizer->GetOutputName() ;
	collectionName.push_back(colName);
	m_Messenger = new GateGridDiscretizatorMessenger(this);

	m_name=m_digitizer->GetSD()->GetName();
    /*numberStripsX= 1;
    numberStripsY= 1;
    numberStripsZ= 1;

    numberReadOutBlocksX= 1;
    numberReadOutBlocksY= 1;
    numberReadOutBlocksZ= 1;

    stripOffsetX=0;
    stripOffsetY=0;
    stripOffsetZ=0;

    stripWidthX=0;
    stripWidthY=0;
    stripWidthZ=0;

    deadSpX=0;
    deadSpY=0;
    deadSpZ=0;*/
 }


GateGridDiscretizator::~GateGridDiscretizator()
{
  delete m_Messenger;

}

void GateGridDiscretizator::SetGridPoints3D( int indexX, int indexY,int indexZ, G4ThreeVector& pos ){
     double posX;
     double posY;
     double posZ;
     //If there are no strips in a direction (an zero offset, deadspace), index should be 0. and stripWidth whole detector so position in te middle

     if(indexX>=0  && indexY>=0  &&  indexZ>=0  ){
         posX=indexX*pitchX-(volSize.getX()-stripWidthX-2*stripOffsetX)/2;
         posY=indexY*pitchY-(volSize.getY()-stripWidthY-2*stripOffsetY)/2;
         posZ=indexZ*pitchZ-(volSize.getZ()-stripWidthZ-2*stripOffsetZ)/2;


         pos.setX(posX);
         pos.setY(posY);
         pos.setZ(posZ);
     }
     else{
         GateError("GateGridDiscretizator" << GetObjectName() << "There is a problem with the index. Please ensure the index corresponds to the correct volume.");
     }


 }

void GateGridDiscretizator::Digitize()
{
	//G4cout<< "Discretization = "<< m_GateGridDiscretizator <<G4endl;



	G4String digitizerName = m_digitizer->m_digitizerName;
	G4String outputCollName = m_digitizer-> GetOutputName();

	m_OutputDigiCollection = new GateDigiCollection(GetName(),outputCollName); // to create the Digi Collection

	G4DigiManager* DigiMan = G4DigiManager::GetDMpointer();




	GateDigiCollection* IDC = 0;
	IDC = (GateDigiCollection*) (DigiMan->GetDigiCollection(m_DCID));

	GateDigi* inputDigi;

	std::vector< GateDigi* >* OutputDigiCollectionVector = m_OutputDigiCollection->GetVector ();
	std::vector<GateDigi*>::iterator iter;


	index_X_list.clear();
	index_Y_list.clear();
	index_Z_list.clear();
	blockIndex.erase(blockIndex.begin(), blockIndex.end());


  if (IDC)
     {
	  G4int n_digi = IDC->entries();

	  //loop over input digits
	  for (G4int i=0;i<n_digi;i++)
	  {
		  inputDigi=(*IDC)[i];

		  //const GateDigiCollection* InputDigiCollectionVector = 0;

		  int current_indexX=INVALID_INDEX;
		  int current_indexY=INVALID_INDEX;
		  int current_indexZ=INVALID_INDEX;
		  int NumberPB_X;
		  int NumberPB_Y;
		  int NumberPB_Z;

		  std::tuple<int, int, int> blockTempIndex;

		  m_outputDigi = new GateDigi(*inputDigi);


		if (((m_outputDigi->GetVolumeID()).GetBottomCreator())){ /// ????

			// I can not access to Solid volume nor in GateGridDiscretizator::ChooseVolume neither in the constructor. So I check here if the size of the volume has loaded for the
			//considered volume and if not I enter the values
			if(volSize.getX()==0){
				//Fill volumeSize
				m_outputDigi->GetVolumeID().GetBottomCreator()->GetLogicalVolume()->GetSolid()->CalculateExtent(kXAxis, limits, at, min, max);
				volSize.setX(max-min);
				m_outputDigi->GetVolumeID().GetBottomCreator()->GetLogicalVolume()->GetSolid()->CalculateExtent(kYAxis, limits, at, min, max);
				volSize.setY(max-min);
				m_outputDigi->GetVolumeID().GetBottomCreator()->GetLogicalVolume()->GetSolid()->CalculateExtent(kZAxis, limits, at, min, max);
				volSize.setZ(max-min);

				if( volSize.getX()<(numberStripsX*stripWidthX+2*stripOffsetX)){
					GateError("The volume defined by number of strips, width and offset is larger that the SD size in X-axis direction ");
				}
				else if ( volSize.getY()<(numberStripsY*stripWidthY+2*stripOffsetY)) {
					GateError("The volume defined by number of strips, width and offset is larger that the SD size in Y-axis direction ");

				}
				else if ( volSize.getZ()<(numberStripsZ*stripWidthZ+2*stripOffsetZ)) {
					GateError("The volume defined by number of strips, width and offset is larger that the SD size in Z-axis direction ");

				}
				else{
					//Fill deadSpace and pitch
					//deadspace is the inactive space between strips or pixels. If there is only one strip dead space is zero. The inactive material in the border is considere offset (stripOffset parameter)
					 if (numberStripsX==1){
						 deadSpX=0.0;
						 if (stripOffsetX==0){
							 //avoid precision errors
							 if (abs(stripWidthX-volSize.getX())>10*EPSILON){
								 GateError("Error when setting strip width, offset or number  in X direction ");
							 }

							 stripWidthX=volSize.getX();
						 }
					 }
					else{
						deadSpX=(volSize.getX()-2*stripOffsetX-stripWidthX)/(numberStripsX-1)-stripWidthX;
						if( deadSpX<EPSILON) deadSpX=0.0;
					}
					 pitchX=stripWidthX+deadSpX;
					//


					 if (numberStripsY==1){
						 deadSpY=0.0;
						 if (stripOffsetY==0){
							 //avoid precision errors
							 if (abs(stripWidthY-volSize.getY())>10*EPSILON){
								 GateError("Error when setting strip width, offset or number in Y direction  ");
							 }
							 stripWidthY=volSize.getY();
						 }
					 }
					 else{
						 deadSpY=(volSize.getY()-2*stripOffsetY-stripWidthY)/(numberStripsY-1)-stripWidthY;
						 if( deadSpY<EPSILON) deadSpY=0;
					 }
					 pitchY=stripWidthY+deadSpY;
					 //

					 if (numberStripsZ==1){
						 deadSpZ=0.0;
						 if (stripOffsetZ==0){
							 //avoid precision errors
							 if (abs(stripWidthZ-volSize.getZ())>10*EPSILON){
								 GateError("Error when setting strip width, offset or number in Z direction  ");
							 }
							 stripWidthZ=volSize.getZ();
						 }


					 }
					else{

						 deadSpZ=(volSize.getZ()-2*stripOffsetZ-stripWidthZ)/(numberStripsZ-1)-stripWidthZ;
						 if( deadSpZ<EPSILON) deadSpZ=0.0;
					}
					 pitchZ=stripWidthZ+deadSpZ;

				}
			}

			//This info makes sense only for idealAdder
			m_outputDigi->SetEnergyIniTrack(-1);
			m_outputDigi->SetEnergyFin(-1);


	        NumberPB_Y=numberStripsY;
	        NumberPB_X=numberStripsX;
	        NumberPB_Z=numberStripsZ;

	        if(numberReadOutBlocksX>0  ){
	            NumberPB_X=int(numberStripsX/numberReadOutBlocksX);
	        }
	        if(numberReadOutBlocksY>0){
	            NumberPB_Y=int(numberStripsY/numberReadOutBlocksY);

	        }
	        if(numberReadOutBlocksZ>0){
	            NumberPB_Z=int(numberStripsZ/numberReadOutBlocksZ);
	        }

			G4ThreeVector PosLocal = m_outputDigi->GetLocalPos();

			current_indexX=GetXIndex(PosLocal.getX());
			if(current_indexX!=INVALID_INDEX){
				current_indexY=GetYIndex(PosLocal.getY());
				current_indexZ=GetZIndex(PosLocal.getZ());
			}


		   //G4cout<<"  Berfor analysing: indexX="<<current_indexX<<"   indexY="<<current_indexY<<G4endl;

			if(current_indexX!=INVALID_INDEX &&current_indexY!=INVALID_INDEX && current_indexZ!=INVALID_INDEX ){

				SetGridPoints3D(current_indexX, current_indexY,current_indexZ, PosLocal);

				m_outputDigi->SetLocalPos(PosLocal);
				m_outputDigi->SetGlobalPos(m_outputDigi->GetVolumeID().MoveToAncestorVolumeFrame(m_outputDigi->GetLocalPos()));
				//Check output pulses from the end if there is a pulse with the same indexX an Y are summed in energy otherwise  a new input
				bool flagPulseIsAdded=false;
				if(OutputDigiCollectionVector->empty()){
					//index_X_list and outputlist same size
					m_OutputDigiCollection->insert(m_outputDigi);
					index_X_list.push_back(current_indexX);
					index_Y_list.push_back(current_indexY);
					index_Z_list.push_back(current_indexZ);

					//Changed for 3d
					std::get<0>(blockTempIndex)= int (current_indexX/NumberPB_X);
					std::get<1>(blockTempIndex)= int (current_indexY/NumberPB_Y);
					std::get<2>(blockTempIndex)= int (current_indexZ/NumberPB_Z);
					//blockTempIndex.first=int (current_indexX/NumberPB_X);
					//blockTempIndex.second=int (current_indexY/NumberPB_Y);

					//If block readout is applied
					if(NumberPB_X!=numberStripsX ||NumberPB_Y!=numberStripsY || NumberPB_Z!=numberStripsZ ){
					   //the strucutre has been changed
						blockIndex.insert(std::pair<std::tuple<int,int,int>, std::vector<int>>(blockTempIndex,{int(OutputDigiCollectionVector->size()-1)}));
					}

				}
				else{
					////////////////////////////Test to deal with multiple ocurrances of indexX/////////////
					std::vector<int>::iterator it_indexX = index_X_list.begin();
					//check the structure
					int indexXCounter=std::count(index_X_list.begin(), index_X_list.end(), current_indexX);
					int indexYCounter=std::count(index_Y_list.begin(), index_Y_list.end(), current_indexY);
					int indexZCounter=std::count(index_Z_list.begin(), index_Z_list.end(), current_indexZ);

					if (indexXCounter==0 ||indexYCounter==0 || indexZCounter==0 ){

						m_OutputDigiCollection->insert(m_outputDigi);
						index_X_list.push_back(current_indexX);
						index_Y_list.push_back(current_indexY);
						index_Z_list.push_back(current_indexZ);
					}
					else{
						//If there is a pulse in the same volumeID with the same indexes, they are merged
						while ((it_indexX = std::find(it_indexX, index_X_list.end(), current_indexX)) != index_X_list.end()) {

							//For that event there is already a pulse created with the same  X-index
							int posListX=std::distance(index_X_list.begin(),it_indexX);
							//See if I have also coincidence in Y indexes (in the same position)

							if(index_Y_list.at(posListX)==current_indexY && index_Z_list.at(posListX)==current_indexZ){
								//Check volumeID not to mix indexes beteween differnet volumes created by the repeater
								if(OutputDigiCollectionVector->at(posListX)->GetVolumeID()==m_outputDigi->GetVolumeID()){
									CentroidMerge(inputDigi,m_outputDigi);
									delete m_outputDigi;
									flagPulseIsAdded=true;
									break;
								}

							} 
							it_indexX++;
						}

						//If there is no pulse stored in the smae volume with the same indexes create one
						if (flagPulseIsAdded==false){
							m_OutputDigiCollection->insert(m_outputDigi);
							index_X_list.push_back(current_indexX);
							index_Y_list.push_back(current_indexY);
							index_Z_list.push_back(current_indexZ);
						}

					}
					//This is for the blocks
					if(flagPulseIsAdded==false){
						if(NumberPB_X!=numberStripsX ||NumberPB_Y!=numberStripsY || NumberPB_Z!=numberStripsZ){

							std::get<0>(blockTempIndex)= int (current_indexX/NumberPB_X);
							std::get<1>(blockTempIndex)= int (current_indexY/NumberPB_Y);
							std::get<2>(blockTempIndex)= int (current_indexZ/NumberPB_Z);
							//because if the key-value exists
								 if(blockIndex.find(blockTempIndex)!=blockIndex.end()){
									blockIndex[blockTempIndex].push_back(int(OutputDigiCollectionVector->size()-1));
								 }
								 else{
									  blockIndex.insert(std::pair<std::tuple<int,int,int>, std::vector<int>>(blockTempIndex,{int(OutputDigiCollectionVector->size()-1)}));
								 }


							}
							 
					}







				}


			}
			else{
				delete m_outputDigi;
			}

		}
		else{

			//If the pulse is not in the selected volume we do not process it, but we save it in the list
			// To be able to associate the position in the outputpulse list wit the position in the index I need both of the same size
			if (nVerboseLevel==1){
				G4cout<<"pulse in"<<((inputDigi->GetVolumeID()).GetBottomCreator())->GetObjectName()<< "is not processed by the grid digitizer "<<G4endl;
			}
			m_OutputDigiCollection->insert(m_outputDigi);
			index_X_list.push_back(current_indexX);
			index_Y_list.push_back(current_indexY);
			index_Z_list.push_back(current_indexZ);
		}

		if (nVerboseLevel==1) {
			G4cout << "[GateGridDiscretizator::Digitize]: returning output pulse-list with " << OutputDigiCollectionVector->size() << " entries\n";
			for (iter=OutputDigiCollectionVector->begin(); iter!= OutputDigiCollectionVector->end() ; ++iter)
				G4cout << **iter << Gateendl;
			G4cout << Gateendl;
		}
	  } //loop  over input digits
	  ApplyBlockReadOut();

    }
  else
    {
  	  if (nVerboseLevel>1)
  	  	G4cout << "[GateGridDiscretizator::Digitize]: input digi collection is null -> nothing to do\n\n";
  	    return;
    }
  StoreDigiCollection(m_OutputDigiCollection);

}





void GateGridDiscretizator::DescribeMyself(size_t indent )
{
	G4cout << GateTools::Indent(indent) << "  and number of strips in X"
	           << numberStripsX<<  Gateendl;
}

int GateGridDiscretizator::GetXIndex(G4double posX){
	int index_i;

	    //position in the new reference sys where the first strip active area starts
	     double pos_SRN=posX-stripOffsetX+ volSize.getX()/2;
	    int indexT=(int)(pos_SRN/pitchX);
	    if(pos_SRN<0){
	         //Invalid index to those that interact in the left offset
	        index_i=INVALID_INDEX;
	    }
	    else{
	        if(pos_SRN>(pitchX*indexT+stripWidthX)){
	             //Invalid index to those that interact in dead space
	             index_i=INVALID_INDEX;
	        }
	        else{
	            if(indexT>=numberStripsX){
	                double l_sp=pos_SRN-numberStripsX*pitchX;
	                if(l_sp<EPSILON){
	                    //hit in the limit of the last strip. Hit assigned to the last strip
	                    index_i=numberStripsX-1;
	                }
	                else{
	                     index_i=INVALID_INDEX;
	                    //hit in the right offset of the detector
	                    if(l_sp>stripOffsetX){
	                        G4cout<<"[GateGridDiscretization::GetXIndex]: Check grid discretization parameters. A hit is registerd outside the defined grid "<<G4endl;
	                    }
	                }

	            }
	            else {
	                index_i=indexT;
	            }
	        }

	    }


	    return index_i;

}

int GateGridDiscretizator::GetYIndex(G4double posY){

	 int index_j;

	      double pos_SRN=posY-stripOffsetY+volSize.getY()/2;
	      int indexT=(int)(pos_SRN/pitchY);
	      if(pos_SRN<0){
	           //Invalid index to those that interact in the left offset
	          index_j=INVALID_INDEX;
	      }
	      else{
	          if(pos_SRN>(pitchY*indexT+stripWidthY)){
	              //Invalid index to those that interact in dead space
	               index_j=INVALID_INDEX;
	          }
	          else{
	              if(indexT>=numberStripsY){
	                  double l_sp=pos_SRN-numberStripsY*pitchY;
	                  if(l_sp<EPSILON){
	                     //hit in the last strip
	                     index_j=numberStripsY-1;
	                  }
	                  else{
	                       index_j=INVALID_INDEX;
	                      //hit in the top offset of the detector
	                      if(l_sp>stripOffsetY){
	                          G4cout<<"[GateGridDiscretization::GetYIndex]:Check grid discretization parameters. A hit is registerd outside the defined grid "<<G4endl;
	                      }
	                  }

	              }
	              else {
	                index_j=indexT;
	             }
	          }
	      }
	      return index_j;
}

int GateGridDiscretizator::GetZIndex(G4double posZ){
    int index_j;

     double pos_SRN=posZ-stripOffsetZ+volSize.getZ()/2;
     int indexT=(int)(pos_SRN/pitchZ);
     if(pos_SRN<0){
          //Invalid index to those that interact in the left offset
         index_j=INVALID_INDEX;
     }
     else{
         if(pos_SRN>(pitchZ*indexT+stripWidthZ)){
             //Invalid index to those that interact in dead space
              index_j=INVALID_INDEX;
         }
         else{
             if(indexT>=numberStripsZ){
                 double l_sp=pos_SRN-numberStripsZ*pitchZ;
                 if(l_sp<EPSILON){
                    //hit in the last strip
                    index_j=numberStripsZ-1;
                 }
                 else{
                      index_j=INVALID_INDEX;
                     //hit in the top offset of the detector
                     if(l_sp>deadSpZ){
                         G4cout<<"[GateGridDiscretization::GetZIndex]:Check grid discretization parameters. A hit is registerd outside the defined grid "<<G4endl;
                     }
                 }

             }
             else {
               index_j=indexT;
            }
         }
     }
     return index_j;
}


void GateGridDiscretizator::ApplyBlockReadOut(){
  if(!m_OutputDigiCollection->entries()){
	  std::vector< GateDigi* >* OutputDigiCollectionVector = m_OutputDigiCollection->GetVector ();
      //It can be applied to different volumesID even if they share the same volume name if they are generated by the repeater
          std::vector<unsigned int> posErase;

          for (auto const& x :  blockIndex){
                     std::map<G4String,std::vector<int>> posToMerge;
                     if(x.second.size()>1){
                         //We have more then one pulse in the same block pair (x,y) (if repeaters were used they can correspond to diferent volumeID)
                         //Important to  check if they are in same volumeID before merging (they can share volume name and not volume ID)
                         //Middle sep to create posToMerge, map where the name of the current volume taking into account the copy number is used as keyvalue to analyse the coincidence for blockIndexPair key
                         for(unsigned int i1=0; i1<x.second.size(); i1++){
                             G4String currentVolumeName=(OutputDigiCollectionVector->at(x.second[i1])->GetVolumeID().GetBottomCreator())->GetObjectName();
                             int currentNumber=OutputDigiCollectionVector->at(x.second[i1])->GetVolumeID().GetBottomVolume()->GetCopyNo();
                             currentVolumeName=currentVolumeName+std::to_string(currentNumber);
                             if(posToMerge.find(currentVolumeName)!=posToMerge.end()){
                                 posToMerge[currentVolumeName].push_back(x.second[i1]);

                             }
                             else{
                                 posToMerge.insert(std::pair<G4String,std::vector<int>> (currentVolumeName,{x.second[i1]}));
                             }

                         }
                     }
                    //If  the size of any of the vector associated to the same volume name  is bigger than 1 then yes merge
                     for (auto const& ps :  posToMerge){
                        if(ps.second.size()>1){
                              //Do operations in the first index output and erase the others
                            for(unsigned int i=1; i<ps.second.size(); i++){

                                // ????? OutputDigiCollectionVector->at(ps.second[0])->CentroidMerge(OutputDigiCollectionVector->at(ps.second[i]));
                                posErase.push_back(ps.second[i]);
                            }

                         }
                     }



           }
          if(posErase.size()>0){

              std::sort (posErase.begin(), posErase.end());
              //delete the merged pulses if any

              for (unsigned int k=0; k<posErase.size(); k++)
              {
                  //G4cout<<"pos to erase"<<  posErase.at(k)<<G4endl,
                  delete OutputDigiCollectionVector->at(posErase.at(k)-k);
                  OutputDigiCollectionVector->erase(OutputDigiCollectionVector->begin()+posErase.at(k)-k); /// ???
              }

          }

  }
}



/*G4int GateGridDiscretizator::ChooseVolume(G4String val)
{
    GateObjectStore* m_store = GateObjectStore::GetInstance();


    if (m_store->FindCreator(val)!=0) {
    
        numberStripsX= 1;
        numberStripsY= 1;
        numberStripsZ= 1;
        
        numberReadOutBlocksX= 1;
        numberReadOutBlocksY= 1;
        numberReadOutBlocksZ= 1;
        
        stripOffsetX=0;
        stripOffsetY=0;
        stripOffsetZ=0;
        
        stripWidthX=0;
        stripWidthY=0;
        stripWidthZ=0;
        
        deadSpX=0;
        deadSpY=0;
        deadSpZ=0;



        return 1;
    }
    else {
        G4cout << "Wrong Volume Name\n";

        return 0;
    }


}
*/
