

#include "GateGridDiscretization.hh"

#include "G4UnitsTable.hh"
#include "GateVolumeID.hh"
#include "GateGridDiscretizationMessenger.hh"
#include "G4Electron.hh"
#include "GateConfiguration.h"
#include "GateObjectStore.hh"
#include "GateConstants.hh"
#include "GateTools.hh"


GateGridDiscretization::GateGridDiscretization(GatePulseProcessorChain* itsChain,
                                               const G4String& itsName)
    : GateVPulseProcessor(itsChain,itsName)
{

    m_messenger = new GateGridDiscretizationMessenger(this);

    DescribeMyself(1);
    //GateMap<G4String,param> ::iterator im;
    im=m_table.begin();
}

GateGridDiscretization::~GateGridDiscretization()
{
	delete m_messenger;
}


G4int GateGridDiscretization::ChooseVolume(G4String val)
{
    G4cout<<" GateGridDiscretizATION::ChooseVolume Begin"<<G4endl;
    GateObjectStore* m_store = GateObjectStore::GetInstance();


    if (m_store->FindCreator(val)!=0) {
        //  m_param.threshold = -1;
        m_param.numberStripsX= 1;
        m_param.numberStripsY= 1;
        m_param.numberStripsZ= 1;
        //
        m_param.numberReadOutBlocksX= 1;
        m_param.numberReadOutBlocksY= 1;
        m_param.numberReadOutBlocksZ= 1;
        //
        m_param.stripOffsetX=0;
        m_param.stripOffsetY=0;
        m_param.stripOffsetZ=0;
        //
        m_param.stripWidthX=0;
        m_param.stripWidthY=0;
        m_param.stripWidthZ=0;
        //
        m_param.deadSpX=0;
        m_param.deadSpY=0;
        m_param.deadSpZ=0;


        m_table[val] = m_param;

        return 1;
    }
    else {
        G4cout << "Wrong Volume Name\n";

        return 0;
    }


}


void GateGridDiscretization::ProcessOnePulse(const GatePulse* inputPulse,GatePulseList& outputPulseList)
{

    int current_indexX=INVALID_INDEX;
    int current_indexY=INVALID_INDEX;
    int current_indexZ=INVALID_INDEX;
    int NumberPB_X;
    int NumberPB_Y;
    int NumberPB_Z;

    std::tuple<int, int, int> blockTempIndex;

    im=m_table.find(((inputPulse->GetVolumeID()).GetBottomCreator())->GetObjectName());
    GatePulse* outputPulse = new GatePulse(*inputPulse);
    //G4cout<<"pulse Hit: vol="<<((outputPulse->GetVolumeID()).GetBottomCreator())->GetObjectName()<<"  pos: X="<<outputPulse->GetGlobalPos().getX()<<"  Y="<<outputPulse->GetGlobalPos().getY()<<"  Z="<<outputPulse->GetGlobalPos().getZ()<<G4endl;


    if(im != m_table.end()){

        // I can not access to Solid volume nor in GateGridDiscretization::ChooseVolume neither in the constructor. So I check here if the size of the volume has loaded for the
        //considered volume and if not I enter the values
        if((*im).second.volSize.getX()==0){
            //Fill volumeSize
            outputPulse->GetVolumeID().GetBottomCreator()->GetLogicalVolume()->GetSolid()->CalculateExtent(kXAxis, limits, at, min, max);
            (*im).second.volSize.setX(max-min);
            outputPulse->GetVolumeID().GetBottomCreator()->GetLogicalVolume()->GetSolid()->CalculateExtent(kYAxis, limits, at, min, max);
            (*im).second.volSize.setY(max-min);
            outputPulse->GetVolumeID().GetBottomCreator()->GetLogicalVolume()->GetSolid()->CalculateExtent(kZAxis, limits, at, min, max);
            (*im).second.volSize.setZ(max-min);
            //G4cout<<"vol "<<(*im).second.volSize.getX()/cm<<"  "<<(*im).second.volSize.getY()/cm<<"  "<<(*im).second.volSize.getZ()/cm<<G4endl;

            if( (*im).second.volSize.getX()<((*im).second.numberStripsX*(*im).second.stripWidthX+2*(*im).second.stripOffsetX)){
                 GateError("The volume defined by number of strips, width and offset is larger that the SD size in X-axis direction ");
            }
            else if ( (*im).second.volSize.getY()<((*im).second.numberStripsY*(*im).second.stripWidthY+2*(*im).second.stripOffsetY)) {
                GateError("The volume defined by number of strips, width and offset is larger that the SD size in Y-axis direction ");

            }
            else if ( (*im).second.volSize.getZ()<((*im).second.numberStripsZ*(*im).second.stripWidthZ+2*(*im).second.stripOffsetZ)) {
                GateError("The volume defined by number of strips, width and offset is larger that the SD size in Z-axis direction ");

            }
            else{
                //Fill deadSpace and pitch
                //deadspace is the inactive space between strips or pixels. If there is only one strip dead space is zero. The inactive material in the border is considere offset (stripOffset parameter)
                if ((*im).second.numberStripsX==1){
                    (*im).second.deadSpX=0.0;
                    if ((*im).second.stripOffsetX==0){
                        //avoid precision errors
                        if (abs((*im).second.stripWidthX-(*im).second.volSize.getX())>10*EPSILON){
                            GateError("Error when setting strip width, offset or number  in X direction ");
                        }

                        (*im).second.stripWidthX=(*im).second.volSize.getX();
                    }
                }
                else{
                    (*im).second.deadSpX=((*im).second.volSize.getX()-2*(*im).second.stripOffsetX-(*im).second.stripWidthX)/((*im).second.numberStripsX-1)-(*im).second.stripWidthX;
                    if( (*im).second.deadSpX<EPSILON) (*im).second.deadSpX=0.0;
                }
                (*im).second.pitchX=(*im).second.stripWidthX+(*im).second.deadSpX;
                //
                if ((*im).second.numberStripsY==1){
                    (*im).second.deadSpY=0.0;
                    if ((*im).second.stripOffsetY==0){
                        //avoid precision errors
                        if (abs((*im).second.stripWidthY-(*im).second.volSize.getY())>10*EPSILON){
                            GateError("Error when setting strip width, offset or number in Y direction  ");
                        }
                        (*im).second.stripWidthY=(*im).second.volSize.getY();
                    }
                }
                else{
                    (*im).second.deadSpY=((*im).second.volSize.getY()-2*(*im).second.stripOffsetY-(*im).second.stripWidthY)/((*im).second.numberStripsY-1)-(*im).second.stripWidthY;
                    if( (*im).second.deadSpY<EPSILON) (*im).second.deadSpY=0;
                }
                (*im).second.pitchY=(*im).second.stripWidthY+(*im).second.deadSpY;
                //
                if ((*im).second.numberStripsZ==1){
                    (*im).second.deadSpZ=0.0;
                    if ((*im).second.stripOffsetZ==0){
                        //avoid precision errors
                        if (abs((*im).second.stripWidthZ-(*im).second.volSize.getZ())>10*EPSILON){
                            GateError("Error when setting strip width, offset or number in Z direction  ");
                        }
                        (*im).second.stripWidthZ=(*im).second.volSize.getZ();
                    }


                }
                else{
                    (*im).second.deadSpZ=((*im).second.volSize.getZ()-2*(*im).second.stripOffsetZ-(*im).second.stripWidthZ)/((*im).second.numberStripsZ-1)-(*im).second.stripWidthZ;
                    if( (*im).second.deadSpZ<EPSILON) (*im).second.deadSpZ=0.0;
                }
                (*im).second.pitchZ=(*im).second.stripWidthZ+(*im).second.deadSpZ;

            }
        }

        //This info makes sense only for idealAdder
        outputPulse->SetEnergyIniTrack(-1);
        outputPulse->SetEnergyFin(-1);

        NumberPB_Y=(*im).second.numberStripsY;
        NumberPB_X=(*im).second.numberStripsX;
        NumberPB_Z=(*im).second.numberStripsZ;

        if((*im).second.numberReadOutBlocksX>0  ){
            NumberPB_X=int((*im).second.numberStripsX/(*im).second.numberReadOutBlocksX);
        }
        if((*im).second.numberReadOutBlocksY>0){
            NumberPB_Y=int((*im).second.numberStripsY/(*im).second.numberReadOutBlocksY);

        }
        if((*im).second.numberReadOutBlocksZ>0){
            NumberPB_Z=int((*im).second.numberStripsZ/(*im).second.numberReadOutBlocksZ);
        }

        //G4ThreeVector PosLocal = outputPulse->GetVolumeID().MoveToBottomVolumeFrame(outputPulse->GetGlobalPos()); //local (deberia ser igual que la local guardada)
        G4ThreeVector PosLocal = outputPulse->GetLocalPos();

        current_indexX=GetXIndex(PosLocal.getX());
        if(current_indexX!=INVALID_INDEX){
            current_indexY=GetYIndex(PosLocal.getY());
            current_indexZ=GetZIndex(PosLocal.getZ());
        }


       //G4cout<<"  Berfor analysing: indexX="<<current_indexX<<"   indexY="<<current_indexY<<G4endl;

        if(current_indexX!=INVALID_INDEX &&current_indexY!=INVALID_INDEX && current_indexZ!=INVALID_INDEX ){

            SetGridPoints3D(current_indexX, current_indexY,current_indexZ, PosLocal);
            outputPulse->SetLocalPos(PosLocal);
            outputPulse->SetGlobalPos(outputPulse->GetVolumeID().MoveToAncestorVolumeFrame(outputPulse->GetLocalPos()));
            //Check output pulses from the end if there is a pulse with the same indexX an Y are summed in energy otherwise  a new input
            bool flagPulseIsAdded=false;
            if(outputPulseList.empty()){
                //index_X_list and outputlist same size
                outputPulseList.push_back(outputPulse);
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
                if(NumberPB_X!=(*im).second.numberStripsX ||NumberPB_Y!=(*im).second.numberStripsY || NumberPB_Z!=(*im).second.numberStripsZ ){
                   //the strucutre has been changed
                    blockIndex.insert(std::pair<std::tuple<int,int,int>, std::vector<int>>(blockTempIndex,{int(outputPulseList.size()-1)}));
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
                    //if( std::find (index_X_list.begin(), index_X_list.end(), current_indexX)==index_X_list.end()){

                    outputPulseList.push_back(outputPulse);
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
                            if(outputPulseList.at(posListX)->GetVolumeID()==outputPulse->GetVolumeID()){
                                outputPulseList.at(posListX)->CentroidMerge(outputPulse);
                                delete outputPulse;
                                flagPulseIsAdded=true;
                                break;
                            }

                        }
                        it_indexX++;
                    }

                    //If there is no pulse stored in the smae volume with the same indexes create one
                    if (flagPulseIsAdded==false){
                        outputPulseList.push_back(outputPulse);
                        index_X_list.push_back(current_indexX);
                        index_Y_list.push_back(current_indexY);
                        index_Z_list.push_back(current_indexZ);
                    }

                }
                //This is for the blocks
                if(flagPulseIsAdded==false){
                    if(NumberPB_X!=(*im).second.numberStripsX ||NumberPB_Y!=(*im).second.numberStripsY || NumberPB_Z!=(*im).second.numberStripsZ){

                        std::get<0>(blockTempIndex)= int (current_indexX/NumberPB_X);
                        std::get<1>(blockTempIndex)= int (current_indexY/NumberPB_Y);
                        std::get<2>(blockTempIndex)= int (current_indexZ/NumberPB_Z);
                        //buscar si el keyvalue existe
                             if(blockIndex.find(blockTempIndex)!=blockIndex.end()){
                               // G4cout<<"size before "<<  blockIndex[blockTempIndex].size()<<G4endl;
                                //if(blockIndex[blockTempIndex].size()>0) G4cout<<"element="<< blockIndex[blockTempIndex].back()<<G4endl;
                                blockIndex[blockTempIndex].push_back(int(outputPulseList.size()-1));
                               // G4cout<<"inserted elem"<< blockIndex[blockTempIndex].back()<<G4endl;
                             }
                             else{
                                  blockIndex.insert(std::pair<std::tuple<int,int,int>, std::vector<int>>(blockTempIndex,{int(outputPulseList.size()-1)}));
                             }


                        }
                         //G4cout<<"pulse "<<"indexX="<<current_indexX<<"   indexY="<<current_indexY<<" blockIndexX="<<blockTempIndex.first<<" blockIndexY="<<blockTempIndex.second<<G4endl;
                }







            }


        }
        else{
            delete outputPulse;
           // G4cout<<"pulse deleted: invalid index. Interaction in the dead space between strips or pixels "<<G4endl;

        }

    }
    else{

        //If the pulse is not in the selected volume we do not process it, but we save it in the list
        // To be able to associate the position in the outputpulse list wit the position in the index I need both of the same size
        if (nVerboseLevel==1){
            G4cout<<"pulse in"<<((inputPulse->GetVolumeID()).GetBottomCreator())->GetObjectName()<< "is not processed by the grid digitizer "<<G4endl;
        }
        outputPulseList.push_back(outputPulse);
        index_X_list.push_back(current_indexX);
        index_Y_list.push_back(current_indexY);
        index_Z_list.push_back(current_indexZ);
       // blockIndex_Y_list.push_back(INVALID_INDEX);
       // blockIndex_Y_list.push_back(INVALID_INDEX);
    }



}


GatePulseList* GateGridDiscretization::ProcessPulseList(const GatePulseList* inputPulseList)
{
  if (!inputPulseList)
    return 0;

  size_t n_pulses = inputPulseList->size();
  if (nVerboseLevel==1)
        G4cout << "[" << GetObjectName() << "::ProcessPulseList]: processing input list with " << n_pulses << " entries\n";
  if (!n_pulses)
    return 0;

  GatePulseList* outputPulseList = new GatePulseList(GetObjectName());

    index_X_list.clear();
    index_Y_list.clear();
    index_Z_list.clear();
   blockIndex.erase(blockIndex.begin(), blockIndex.end());
 // for(imBlock=blockIndex.begin(); imBlock!=blockIndex.end(); ++imBlock )blockIndex.erase()
 //G4cout<<"####################################   NEW EVENT   #######################################"<<G4endl;
  GatePulseConstIterator iter;
  for (iter = inputPulseList->begin() ; iter != inputPulseList->end() ; ++iter)
      ProcessOnePulse( *iter, *outputPulseList);
  ApplyBlockReadOut(*outputPulseList);

  //ApplyEnergyThreshold(*outputPulseList);



  if (nVerboseLevel==1) {
      G4cout << "[" << GetObjectName() << "::ProcessPulseList]: returning output pulse-list with " << outputPulseList->size() << " entries\n";
      for (iter = outputPulseList->begin() ; iter != outputPulseList->end() ; ++iter)
        G4cout << **iter << Gateendl;
      G4cout << Gateendl;
  }

  return outputPulseList;
}


void GateGridDiscretization::ApplyBlockReadOut( GatePulseList& outputPulseList){
  if(!outputPulseList.empty()){

      //It can be applied to different volumesID even if they share the same volume name if they are generated by the repeater
          std::vector<unsigned int> posErase;

          for (auto const& x :  blockIndex){
                     std::map<G4String,std::vector<int>> posToMerge;
                     if(x.second.size()>1){
                         //We have more then one pulse in the same block pair (x,y) (if repeaters were used they can correspond to diferent volumeID)
                         //Important to  check if they are in same volumeID before merging (they can share volume name and not volume ID)
                         //Middle sep to create posToMerge, map where the name of the current volume taking into account the copy number is used as keyvalue to analyse the coincidence for blockIndexPair key
                         for(unsigned int i1=0; i1<x.second.size(); i1++){
                             G4String currentVolumeName=(outputPulseList.at(x.second[i1])->GetVolumeID().GetBottomCreator())->GetObjectName();
                             int currentNumber=outputPulseList.at(x.second[i1])->GetVolumeID().GetBottomVolume()->GetCopyNo();
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

                                outputPulseList.at(ps.second[0])->CentroidMerge(outputPulseList.at(ps.second[i]));
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
                  delete outputPulseList.at(posErase.at(k)-k);
                  outputPulseList.erase(outputPulseList.begin()+posErase.at(k)-k);
              }

          }

  }
}

/*void GateGridDiscretization::ApplyEnergyThreshold( GatePulseList& outputPulseList){
    // Checkear la energia de los pulsos con  indices validos o en el volumen escogido.
    //Borrar los que no cumplan con la condicion

    if(!outputPulseList.empty()){
        std::vector<unsigned int> posErase;

        GatePulseList::iterator iterIntern;
        for (iterIntern = outputPulseList.begin() ; iterIntern != outputPulseList.end() ; ++iterIntern ){
            im=m_table.find((((*iterIntern)->GetVolumeID()).GetBottomCreator())->GetObjectName());
            if(im != m_table.end()){
                //El pulso esta en el volumen selecionadao
                if((*iterIntern)->GetEnergy()<(*im).second.threshold){
                    posErase.push_back(std::distance(outputPulseList.begin(),iterIntern));
                    //Reject the pulse (if i detele it here I will have problems
                    //outputPulseList.erase(iterIntern);


                }



            }
            else{
                //Pulse is not in the volume where the digitizer is applied
            }
        }
        //G4cout<<"antes de  borrrar"<<G4endl;
        //Aqui voy a borrar

        for(unsigned int i=0; i<posErase.size(); i++){
            //G4cout<<"pos en lista"<<posErase.at(i)<<G4endl;
            //G4cout<<"size ants"<<index_X_list.size()<<"  "<<outputPulseList.size()<<G4endl;
            index_X_list.erase(index_X_list.begin()+posErase.at(i)-i);
            index_Y_list.erase(index_Y_list.begin()+posErase.at(i)-i);
            outputPulseList.erase(outputPulseList.begin()+posErase.at(i)-i);
            //G4cout<<"size despies"<<index_X_list.size()<<"  "<<outputPulseList.size()<<G4endl;
        }

    }
}*/





void GateGridDiscretization::DescribeMyself(size_t  indent)
{
    for (im=m_table.begin(); im!=m_table.end(); im++)
        G4cout << GateTools::Indent(indent) << "  and number of strips in X"
           << (*im).second.numberStripsX<<  Gateendl;
}

int GateGridDiscretization::GetXIndex(G4double posX){

    int index_i;

    //position in the new reference sys where the first strip active area starts
     double pos_SRN=posX-(*im).second.stripOffsetX+ (*im).second.volSize.getX()/2;
    int indexT=(int)(pos_SRN/(*im).second.pitchX);
    if(pos_SRN<0){
         //Invalid index to those that interact in the left offset
        index_i=INVALID_INDEX;
    }
    else{
        if(pos_SRN>((*im).second.pitchX*indexT+(*im).second.stripWidthX)){
             //Invalid index to those that interact in dead space
             index_i=INVALID_INDEX;
        }
        else{
            if(indexT>=(*im).second.numberStripsX){
                double l_sp=pos_SRN-(*im).second.numberStripsX*(*im).second.pitchX;
                if(l_sp<EPSILON){
                    //hit in the limit of the last strip. Hit assigned to the last strip
                    index_i=(*im).second.numberStripsX-1;
                }
                else{
                     index_i=INVALID_INDEX;
                    //hit in the right offset of the detector
                    if(l_sp>(*im).second.stripOffsetX){
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

int GateGridDiscretization::GetYIndex(G4double posY){

      int index_j;

      double pos_SRN=posY-(*im).second.stripOffsetY+(*im).second.volSize.getY()/2;
      int indexT=(int)(pos_SRN/(*im).second.pitchY);
      if(pos_SRN<0){
           //Invalid index to those that interact in the left offset
          index_j=INVALID_INDEX;
      }
      else{
          if(pos_SRN>((*im).second.pitchY*indexT+(*im).second.stripWidthY)){
              //Invalid index to those that interact in dead space
               index_j=INVALID_INDEX;
          }
          else{
              if(indexT>=(*im).second.numberStripsY){
                  double l_sp=pos_SRN-(*im).second.numberStripsY*(*im).second.pitchY;
                  if(l_sp<EPSILON){
                     //hit in the last strip
                     index_j=(*im).second.numberStripsY-1;
                  }
                  else{
                       index_j=INVALID_INDEX;
                      //hit in the top offset of the detector
                      if(l_sp>(*im).second.stripOffsetY){
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

int GateGridDiscretization::GetZIndex(G4double posZ){

      int index_j;

      double pos_SRN=posZ-(*im).second.stripOffsetZ+(*im).second.volSize.getZ()/2;
      int indexT=(int)(pos_SRN/(*im).second.pitchZ);
      if(pos_SRN<0){
           //Invalid index to those that interact in the left offset
          index_j=INVALID_INDEX;
      }
      else{
          if(pos_SRN>((*im).second.pitchZ*indexT+(*im).second.stripWidthZ)){
              //Invalid index to those that interact in dead space
               index_j=INVALID_INDEX;
          }
          else{
              if(indexT>=(*im).second.numberStripsZ){
                  double l_sp=pos_SRN-(*im).second.numberStripsZ*(*im).second.pitchZ;
                  if(l_sp<EPSILON){
                     //hit in the last strip
                     index_j=(*im).second.numberStripsZ-1;
                  }
                  else{
                       index_j=INVALID_INDEX;
                      //hit in the top offset of the detector
                      if(l_sp>(*im).second.deadSpZ){
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

 void GateGridDiscretization::SetGridPoints3D( int indexX, int indexY,int indexZ, G4ThreeVector& pos ){
     double posX;
     double posY;
     double posZ;
     //If there are no strips in a direction (an zero offset, deadspace), index should be 0. and stripWidth whole detector so position in te middle

     if(indexX>=0  && indexY>=0  &&  indexZ>=0  ){
         //posX=indexX*pitchX-(sizeVol[0]-(*im).second.stripWidthX-2*(*im).second.stripOffsetX)/2;
         //posY=indexY*pitchY-(sizeVol[1]-(*im).second.stripWidthY-2*(*im).second.stripOffsetY)/2;
         posX=indexX*(*im).second.pitchX-((*im).second.volSize.getX()-(*im).second.stripWidthX-2*(*im).second.stripOffsetX)/2;
         posY=indexY*(*im).second.pitchY-((*im).second.volSize.getY()-(*im).second.stripWidthY-2*(*im).second.stripOffsetY)/2;
         posZ=indexZ*(*im).second.pitchZ-((*im).second.volSize.getZ()-(*im).second.stripWidthZ-2*(*im).second.stripOffsetZ)/2;


         pos.setX(posX);
         pos.setY(posY);
         pos.setZ(posZ);
     }
     else{
         G4cout<<"problems with index (is  the right volume?)"<<G4endl;
     }


 }


//this is standalone only because it repeats twice in processOnePulse()
inline void GateGridDiscretization::PulsePushBack(const GatePulse* inputPulse, GatePulseList& outputPulseList)
{
	GatePulse* outputPulse = new GatePulse(*inputPulse);
    if (nVerboseLevel>1)
		G4cout << "Created new pulse for volume " << inputPulse->GetVolumeID() << ".\n"
		<< "Resulting pulse is: \n"
		<< *outputPulse << Gateendl << Gateendl ;
	outputPulseList.push_back(outputPulse);
}


