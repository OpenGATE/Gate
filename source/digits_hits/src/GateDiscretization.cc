
/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

/*!
  \class  GateDiscretization
  
  This module allows to simulate the readout of strip and pixelated detectors
  
  Last modification (Adaptation to GND): June 2023 by Mohamed-Jordan Soumano mjsoumano@yahoo.com
*/

#include "GateDiscretization.hh"
#include "GateDiscretizationMessenger.hh"
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

GateDiscretization::GateDiscretization(GateSinglesDigitizer *digitizer, G4String name)
  :GateVDigitizerModule(name,"digitizerMgr/"+digitizer->GetSD()->GetName()+"/SinglesDigitizer/"+digitizer->m_digitizerName+"/"+name,digitizer,digitizer->GetSD()),
   m_GateDiscretization(0),
   m_outputDigi(0),
   m_OutputDigiCollection(0),
   m_digitizer(digitizer)
 {
	G4String colName = digitizer->GetOutputName() ;
	collectionName.push_back(colName);
	m_Messenger = new GateDiscretizationMessenger(this);
}


GateDiscretization::~GateDiscretization()
{
  delete m_Messenger;

}

void GateDiscretization::SetGridPoints3D( int indexX, int indexY,int indexZ, G4ThreeVector& pos ){
     double posX;
     double posY;
     double posZ;
     //If there are no strips in a direction (an zero offset, deadspace), index should be 0. and stripWidth whole detector so position in te middle

     if(indexX>=0  && indexY>=0  &&  indexZ>=0  ){
         posX=indexX*m_param.pitchX-(m_param.volSize.getX()-m_param.stripWidthX-2*m_param.stripOffsetX)/2;
         posY=indexY*m_param.pitchY-(m_param.volSize.getY()-m_param.stripWidthY-2*m_param.stripOffsetY)/2;
         posZ=indexZ*m_param.pitchZ-(m_param.volSize.getZ()-m_param.stripWidthZ-2*m_param.stripOffsetZ)/2;


         pos.setX(posX);
         pos.setY(posY);
         pos.setZ(posZ);
     }
     else{
         G4cout<<"problems with index (is  the right volume?)"<<G4endl;
     }


 }

void GateDiscretization::Digitize()
{
	//G4cout<< "Discretization = "<< m_GateDiscretization <<G4endl;


	G4String digitizerName = m_digitizer->m_digitizerName;
	G4String outputCollName = m_digitizer-> GetOutputName();

	m_OutputDigiCollection = new GateDigiCollection(GetName(),outputCollName); // to create the Digi Collection

	G4DigiManager* DigiMan = G4DigiManager::GetDMpointer();




	GateDigiCollection* IDC = 0;
	IDC = (GateDigiCollection*) (DigiMan->GetDigiCollection(m_DCID));

	GateDigi* inputDigi;

	std::vector< GateDigi* >* OutputDigiCollectionVector = m_OutputDigiCollection->GetVector ();
	std::vector<GateDigi*>::iterator iter;

  if (IDC)
     {
	  G4int n_digi = IDC->entries();

	  //loop over input digits
	  for (G4int i=0;i<n_digi;i++)
	  {
		  inputDigi=(*IDC)[i];

		  const GateDigiCollection* InputDigiCollectionVector = 0;

		   index_X_list.clear();
		   index_Y_list.clear();
		   index_Z_list.clear();
		 blockIndex.erase(blockIndex.begin(), blockIndex.end());


		 if (nVerboseLevel==1) {
			 G4cout << "[" << GetObjectName() << "::ProcessPulseList]: returning output pulse-list with " << OutputDigiCollectionVector->size() << " entries\n";
			 for (iter = OutputDigiCollectionVector->begin() ; iter != OutputDigiCollectionVector->end() ; ++iter)
			   G4cout << **iter << Gateendl;
			 G4cout << Gateendl;
		}


		int current_indexX=INVALID_INDEX;
		int current_indexY=INVALID_INDEX;
		int current_indexZ=INVALID_INDEX;
		int NumberPB_X;
		int NumberPB_Y;
		int NumberPB_Z;

		std::tuple<int, int, int> blockTempIndex;

		GateDigi* m_outputDigi = new GateDigi(*inputDigi);


		if (((m_outputDigi->GetVolumeID()).GetBottomCreator())){

			// I can not access to Solid volume nor in GateDiscretization::ChooseVolume neither in the constructor. So I check here if the size of the volume has loaded for the
			//considered volume and if not I enter the values
			if(m_param.volSize.getX()==0){
				//Fill volumeSize
				m_outputDigi->GetVolumeID().GetBottomCreator()->GetLogicalVolume()->GetSolid()->CalculateExtent(kXAxis, limits, at, min, max);
				m_param.volSize.setX(max-min);
				m_outputDigi->GetVolumeID().GetBottomCreator()->GetLogicalVolume()->GetSolid()->CalculateExtent(kYAxis, limits, at, min, max);
				m_param.volSize.setY(max-min);
				m_outputDigi->GetVolumeID().GetBottomCreator()->GetLogicalVolume()->GetSolid()->CalculateExtent(kZAxis, limits, at, min, max);
				m_param.volSize.setZ(max-min);
				G4cout<<"vol "<<m_param.volSize.getX()/cm<<" "<<m_param.volSize.getY()/cm<<"  "<<m_param.volSize.getZ()/cm<<G4endl; //

				if( m_param.volSize.getX()<(m_param.numberStripsX*m_param.stripWidthX+2*m_param.stripOffsetX)){
					 GateError("The volume defined by number of strips, width and offset is larger that the SD size in X-axis direction ");
				}
				else if ( m_param.volSize.getY()<(m_param.numberStripsY*m_param.stripWidthY+2*m_param.stripOffsetY)) {
					GateError("The volume defined by number of strips, width and offset is larger that the SD size in Y-axis direction ");

				}
				else if ( m_param.volSize.getZ()<(m_param.numberStripsZ*m_param.stripWidthZ+2*m_param.stripOffsetZ)) {
					GateError("The volume defined by number of strips, width and offset is larger that the SD size in Z-axis direction ");

				}
				else{
					//Fill deadSpace and pitch
					//deadspace is the inactive space between strips or pixels. If there is only one strip dead space is zero. The inactive material in the border is considere offset (stripOffset parameter)
					if (m_param.numberStripsX==1){
						m_param.deadSpX=0.0;
						if (m_param.stripOffsetX==0){
							//avoid precision errors
							if (abs(m_param.stripWidthX-m_param.volSize.getX())>10*EPSILON){
								GateError("Error when setting strip width, offset or number  in X direction ");
							}

							m_param.stripWidthX=m_param.volSize.getX();
						}
					}
					else{
						m_param.deadSpX=(m_param.volSize.getX()-2*m_param.stripOffsetX-m_param.stripWidthX)/(m_param.numberStripsX-1)-m_param.stripWidthX;
						if( m_param.deadSpX<EPSILON) m_param.deadSpX=0.0;
					}
					m_param.pitchX=m_param.stripWidthX+m_param.deadSpX;
					
					if (m_param.numberStripsY==1){
						m_param.deadSpY=0.0;
						if (m_param.stripOffsetY==0){
							//avoid precision errors
							if (abs(m_param.stripWidthY-m_param.volSize.getY())>10*EPSILON){
								GateError("Error when setting strip width, offset or number in Y direction  ");
							}
							m_param.stripWidthY=m_param.volSize.getY();
						}
					}
					else{
						m_param.deadSpY=(m_param.volSize.getY()-2*m_param.stripOffsetY-m_param.stripWidthY)/(m_param.numberStripsY-1)-m_param.stripWidthY;
						if( m_param.deadSpY<EPSILON) m_param.deadSpY=0;
					}
					m_param.pitchY=m_param.stripWidthY+m_param.deadSpY;
					
					if (m_param.numberStripsZ==1){
						m_param.deadSpZ=0.0;
						if (m_param.stripOffsetZ==0){
							//avoid precision errors
							if (abs(m_param.stripWidthZ-m_param.volSize.getZ())>10*EPSILON){
								GateError("Error when setting strip width, offset or number in Z direction  ");
							}
							m_param.stripWidthZ=m_param.volSize.getZ();
						}


					}
					else{
						m_param.deadSpZ=(m_param.volSize.getZ()-2*m_param.stripOffsetZ-m_param.stripWidthZ)/(m_param.numberStripsZ-1)-m_param.stripWidthZ;
						if( m_param.deadSpZ<EPSILON) m_param.deadSpZ=0.0;
					}
					m_param.pitchZ=m_param.stripWidthZ+m_param.deadSpZ;

				}
			}

			//This info makes sense only for idealAdder
			m_outputDigi->SetEnergyIniTrack(-1);
			m_outputDigi->SetEnergyFin(-1);

			NumberPB_Y=m_param.numberStripsY;
			NumberPB_X=m_param.numberStripsX;
			NumberPB_Z=m_param.numberStripsZ;

			if(m_param.numberReadOutBlocksX>0  ){
				NumberPB_X=int(m_param.numberStripsX/m_param.numberReadOutBlocksX);
			}
			if(m_param.numberReadOutBlocksY>0){
				NumberPB_Y=int(m_param.numberStripsY/m_param.numberReadOutBlocksY);

			}
			if(m_param.numberReadOutBlocksZ>0){
				NumberPB_Z=int(m_param.numberStripsZ/m_param.numberReadOutBlocksZ);
			}

			G4ThreeVector PosLocal = m_outputDigi->GetLocalPos();

			current_indexX=GetXIndex(PosLocal.getX());
			if(current_indexX!=INVALID_INDEX){
				current_indexY=GetYIndex(PosLocal.getY());
				current_indexZ=GetZIndex(PosLocal.getZ());
			}


		   //G4cout<<"  Berfor analysing: indexX="<<current_indexX<<"   indexY="<<current_indexY<<G4endl;

			if(current_indexX!=INVALID_INDEX &&current_indexY!=INVALID_INDEX && current_indexZ!=INVALID_INDEX ){

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
					if(NumberPB_X!=m_param.numberStripsX ||NumberPB_Y!=m_param.numberStripsY || NumberPB_Z!=m_param.numberStripsZ ){
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
						if(NumberPB_X!=m_param.numberStripsX ||NumberPB_Y!=m_param.numberStripsY || NumberPB_Z!=m_param.numberStripsZ){

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
			G4cout << "[GateDiscretization::Digitize]: returning output pulse-list with " << OutputDigiCollectionVector->size() << " entries\n";
			for (iter=OutputDigiCollectionVector->begin(); iter!= OutputDigiCollectionVector->end() ; ++iter)
				G4cout << **iter << Gateendl;
			G4cout << Gateendl;
		}
	  } //loop  over input digits
    }
  else
    {
  	  if (nVerboseLevel>1)
  	  	G4cout << "[GateDiscretization::Digitize]: input digi collection is null -> nothing to do\n\n";
  	    return;
    }
  StoreDigiCollection(m_OutputDigiCollection);

}





void GateDiscretization::DescribeMyself(size_t indent )
{
  ;
}

int GateDiscretization::GetXIndex(G4double posX){

    int index_i;

    //position in the new reference sys where the first strip active area starts
     double pos_SRN=posX-m_param.stripOffsetX+ m_param.volSize.getX()/2;
    int indexT=(int)(pos_SRN/m_param.pitchX);
    if(pos_SRN<0){
         //Invalid index to those that interact in the left offset
        index_i=INVALID_INDEX;
    }
    else{
        if(pos_SRN>(m_param.pitchX*indexT+m_param.stripWidthX)){
             //Invalid index to those that interact in dead space
             index_i=INVALID_INDEX;
        }
        else{
            if(indexT>=m_param.numberStripsX){
                double l_sp=pos_SRN-m_param.numberStripsX*m_param.pitchX;
                if(l_sp<EPSILON){
                    //hit in the limit of the last strip. Hit assigned to the last strip
                    index_i=m_param.numberStripsX-1;
                }
                else{
                     index_i=INVALID_INDEX;
                    //hit in the right offset of the detector
                    if(l_sp>m_param.stripOffsetX){
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

int GateDiscretization::GetYIndex(G4double posY){

      int index_j;

      double pos_SRN=posY-m_param.stripOffsetY+m_param.volSize.getY()/2;
      int indexT=(int)(pos_SRN/m_param.pitchY);
      if(pos_SRN<0){
           //Invalid index to those that interact in the left offset
          index_j=INVALID_INDEX;
      }
      else{
          if(pos_SRN>(m_param.pitchY*indexT+m_param.stripWidthY)){
              //Invalid index to those that interact in dead space
               index_j=INVALID_INDEX;
          }
          else{
              if(indexT>=m_param.numberStripsY){
                  double l_sp=pos_SRN-m_param.numberStripsY*m_param.pitchY;
                  if(l_sp<EPSILON){
                     //hit in the last strip
                     index_j=m_param.numberStripsY-1;
                  }
                  else{
                       index_j=INVALID_INDEX;
                      //hit in the top offset of the detector
                      if(l_sp>m_param.stripOffsetY){
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

int GateDiscretization::GetZIndex(G4double posZ){

      int index_j;

      double pos_SRN=posZ-m_param.stripOffsetZ+m_param.volSize.getZ()/2;
      int indexT=(int)(pos_SRN/m_param.pitchZ);
      if(pos_SRN<0){
           //Invalid index to those that interact in the left offset
          index_j=INVALID_INDEX;
      }
      else{
          if(pos_SRN>(m_param.pitchZ*indexT+m_param.stripWidthZ)){
              //Invalid index to those that interact in dead space
               index_j=INVALID_INDEX;
          }
          else{
              if(indexT>=m_param.numberStripsZ){
                  double l_sp=pos_SRN-m_param.numberStripsZ*m_param.pitchZ;
                  if(l_sp<EPSILON){
                     //hit in the last strip
                     index_j=m_param.numberStripsZ-1;
                  }
                  else{
                       index_j=INVALID_INDEX;
                      //hit in the top offset of the detector
                      if(l_sp>m_param.deadSpZ){
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

G4int GateDiscretization::ChooseVolume(G4String val)
{
    G4cout<<" GateDiscretizATION::ChooseVolume Begin"<<G4endl;
    GateObjectStore* m_store = GateObjectStore::GetInstance();


    if (m_store->FindCreator(val)!=0) {
    
        m_param.numberStripsX= 1;
        m_param.numberStripsY= 1;
        m_param.numberStripsZ= 1;
        
        m_param.numberReadOutBlocksX= 1;
        m_param.numberReadOutBlocksY= 1;
        m_param.numberReadOutBlocksZ= 1;
        
        m_param.stripOffsetX=0;
        m_param.stripOffsetY=0;
        m_param.stripOffsetZ=0;
        
        m_param.stripWidthX=0;
        m_param.stripWidthY=0;
        m_param.stripWidthZ=0;
        
        m_param.deadSpX=0;
        m_param.deadSpY=0;
        m_param.deadSpZ=0;



        return 1;
    }
    else {
        G4cout << "Wrong Volume Name\n";

        return 0;
    }


}

