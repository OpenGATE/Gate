

#include "GateGridDiscretization.hh"

#include "G4UnitsTable.hh"
#include "GateVolumeID.hh"
#include "GateGridDiscretizationMessenger.hh"
#include "G4Electron.hh"
#include "GateConfiguration.h"
#include "GateObjectStore.hh"
#include "GateConstants.hh"
#include "GateTools.hh"

#include "GateVVolume.hh"
#include "G4VoxelLimits.hh"




GateGridDiscretization::GateGridDiscretization(GatePulseProcessorChain* itsChain,
                                                               const G4String& itsName)
    : GateVPulseProcessor(itsChain,itsName)
{


 m_messenger = new GateGridDiscretizationMessenger(this);

    DescribeMyself(1);
    //GateMap<G4String,param> ::iterator im;





    sizeVol=new double [2];

  // G4cout<<"access to m_name in constructor"<< m_name<<G4endl;

//Es como que lo guarda mas tarde
   im=m_table.begin();
   pitchX=0;
   pitchY=0;
   VolCN="";

  // G4cout<<(*im).second.numberStripsX<<G4endl;

   // G4cout<<(G4String)(*im).first<<G4endl;
    //No tengo ahi acceso a los parameteros
   //G4cout<<"access to m_param number of strips in constructor"<< m_param.numberStripsX<<G4endl;
   // G4cout<<"access to m_param number of strips in constructor"<<   m_table[m_name].numberStripsX<<G4endl;
    //No recupero parametros introducidos
       //pitchX=(sizeVol[0]-2*)
}

GateGridDiscretization::~GateGridDiscretization()
{
	delete m_messenger;
}


G4int GateGridDiscretization::ChooseVolume(G4String val)
{

  GateObjectStore* m_store = GateObjectStore::GetInstance();


  if (m_store->FindCreator(val)!=0) {
      m_param.threshold = -1;
      m_param.numberStripsX= -1;
      m_param.numberStripsY= -1;
      m_param.stripOffsetX=-1;
      m_param.stripOffsetY=-1;
      m_param.stripWidthX=-1;
      m_param.stripWidthY=-1;
      m_table[val] = m_param;


      G4VoxelLimits limits;
      G4double min, max;
      G4AffineTransform at;


      //Saving the dimensions of the volumeSince local module I will not have problems ?
      m_store->FindCreator(val)->GetLogicalVolume()->GetSolid()->CalculateExtent(kXAxis, limits, at, min, max);
      //G4cout<< min<<G4endl;
      sizeVol[0] = max-min;
      m_store->FindCreator(val)->GetLogicalVolume()->GetSolid()->CalculateExtent(kYAxis, limits, at, min, max);
      sizeVol[1] = max-min;
      //    G4cout<<"size in x direction"<< s izeVol[0]<<"size in y direction"<< sizeVol[1]<<G4endl;


      return 1;
  }
  else {
    G4cout << "Wrong Volume Name\n";
    return 0;
  }

}


void GateGridDiscretization::ProcessOnePulse(const GatePulse* inputPulse,GatePulseList& outputPulseList)
{

    int current_indexX=-1;
    int current_indexY=-1;
    im=m_table.find(((inputPulse->GetVolumeID()).GetBottomCreator())->GetObjectName());
    GatePulse* outputPulse = new GatePulse(*inputPulse);

    if(im != m_table.end()){
        //Check for diff volID with same name (repeaters)
        //value here is loaded Not in the constructor
        //G4cout<<"Numb stripsX"<<(*im).second.numberStripsX<<G4endl;

        //G4ThreeVector PosLocal = outputPulse->GetVolumeID().MoveToBottomVolumeFrame(outputPulse->GetGlobalPos()); //local (deberia ser igual que la local guardada)
        G4ThreeVector PosLocal = outputPulse->GetLocalPos();

        current_indexX=GetXIndex(PosLocal.getX());
        if(current_indexX!=INVALID_INDEX){
            current_indexY=GetYIndex(PosLocal.getY());
        }


        if(current_indexX>=0 &&current_indexY>=0){
            SetGridPoints2D(current_indexX, current_indexY, PosLocal);
            //GetGridPoints2D(PosLocal);
            outputPulse->SetLocalPos(PosLocal);
            outputPulse->SetGlobalPos(outputPulse->GetVolumeID().MoveToAncestorVolumeFrame(outputPulse->GetLocalPos()));
            //Check output pulses from the end if there is a pulse with the same indexX an Y are summed in energy otherwise  a new input
            if(outputPulseList.empty()){
                outputPulseList.push_back(outputPulse);
                index_X_list.push_back(current_indexX);
                index_Y_list.push_back(current_indexY);

            }
            else{
                ////////////////////////////Test to deal with multiple ocurrances of indexX/////////////
                std::vector<int>::iterator it_indexX = index_X_list.begin();
                //check the structure
                if( std::find (index_X_list.begin(), index_X_list.end(), current_indexX)==index_X_list.end()){

                    outputPulseList.push_back(outputPulse);
                    index_X_list.push_back(current_indexX);
                    index_Y_list.push_back(current_indexY);
                }
                while ((it_indexX = std::find(it_indexX, index_X_list.end(), current_indexX)) != index_X_list.end()) {

                    //For that event there is already a pulse created with the same  X-index
                    int posListX=std::distance(index_X_list.begin(),it_indexX);
                    //See if I have also coincidence in Y indexes (in the same position)

                    if(index_Y_list.at(posListX)==current_indexY){
                        //This does not change the posiiton
                        // (*iter)->CentroidMergeComptPhotIdeal(inputPulse);
                        //This sum the energy and cmpute the centroid of positio but being the same position the resutls should be the same
                        if(outputPulseList.at(posListX)->GetVolumeID()==outputPulse->GetVolumeID()){
                            // Checking that I am summing in the same volumeID with repeaters otherwise it  can be a problem
                            outputPulseList.at(posListX)->CentroidMerge(outputPulse);

                        }
                        else{
                            outputPulseList.push_back(outputPulse);
                            index_X_list.push_back(current_indexX);
                            index_Y_list.push_back(current_indexY);
                        }


                    }
                    else{
                        outputPulseList.push_back(outputPulse);
                        index_X_list.push_back(current_indexX);
                        index_Y_list.push_back(current_indexY);
                    }

                    it_indexX++;
                }




            }


        }
        else{
            //This pulses must be rejected
            delete outputPulse;

        }

    }
    else{

        //If the pulse is not in the selected volume we do not process it, but we save it in the list
        // To be able to associate the position in the outputpulse list wit the position in the index I need both of the same size
        outputPulseList.push_back(outputPulse);
        index_X_list.push_back(current_indexX);
        index_Y_list.push_back(current_indexY);
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


  GatePulseConstIterator iter;
  for (iter = inputPulseList->begin() ; iter != inputPulseList->end() ; ++iter)
      ProcessOnePulse( *iter, *outputPulseList);
  ApplyEnergyThreshold(*outputPulseList);



  if (nVerboseLevel==1) {
      G4cout << "[" << GetObjectName() << "::ProcessPulseList]: returning output pulse-list with " << outputPulseList->size() << " entries\n";
      for (iter = outputPulseList->begin() ; iter != outputPulseList->end() ; ++iter)
        G4cout << **iter << Gateendl;
      G4cout << Gateendl;
  }

  return outputPulseList;
}


void GateGridDiscretization::ApplyEnergyThreshold( GatePulseList& outputPulseList){
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
}





void GateGridDiscretization::DescribeMyself(size_t  indent)
{
    for (im=m_table.begin(); im!=m_table.end(); im++)
        G4cout << GateTools::Indent(indent) << "Threshold of " << (*im).first << ":\n"
           << GateTools::Indent(indent+1) << G4BestUnit( (*im).second.threshold,"Energy")<< "  and number of strips in X"
           << (*im).second.numberStripsX<<  Gateendl;
}

int GateGridDiscretization::GetXIndex(G4double posX){

    int index_i;
    if((*im).first!=VolCN){
        G4cout<<"VolC="<<VolCN<<"  imVolName="<<(*im).first<<G4endl;
        //I should calculate this only once but I am not able of obtaining the quantities in the constructor
        // In choose volume? What will happen if I apply the module to two volumes?
        dspX=(sizeVol[0]-2*(*im).second.stripOffsetX-(*im).second.stripWidthX)/((*im).second.numberStripsX-1)-(*im).second.stripWidthX;
        pitchX=(*im).second.stripWidthX+dspX;
        VolCN=(*im).first;
    }
    double pos_SRN=posX-(*im).second.stripOffsetX+sizeVol[0]/2;
    int indexT=(int)(pos_SRN/pitchX);
    if(pos_SRN<0){
         //Invalid index to those that interact in the left offset
        index_i=INVALID_INDEX;
    }
    else{
        if(pos_SRN>(pitchX*indexT+(*im).second.stripWidthX)){
             //Invalid index to those that interact in dead space
             index_i=INVALID_INDEX;
        }
        else{
            index_i=indexT;
        }

    }

    return index_i;
}

int GateGridDiscretization::GetYIndex(G4double posY){

      int index_j;
     //I should calculate this only once (dspY, pticY,..) but I am not able of obtaining the quantities in the constructor
      if((*im).first!=VolCN){
      //If the volumes is the same as previous one do not calculate again quantities
          G4cout<<"VolC="<<VolCN<<"  imVolName="<<(*im).first<<G4endl;
          dspY=(sizeVol[1]-2*(*im).second.stripOffsetY-(*im).second.stripWidthY)/((*im).second.numberStripsY-1)-(*im).second.stripWidthY;
          pitchY=(*im).second.stripWidthY+dspY;
          VolCN=(*im).first;
      }
      double pos_SRN=posY-(*im).second.stripOffsetY+sizeVol[1]/2;
      int indexT=(int)(pos_SRN/pitchY);
      if(pos_SRN<0){
           //Invalid index to those that interact in the left offset
          index_j=INVALID_INDEX;
      }
      else{
          if(pos_SRN>(pitchY*indexT+(*im).second.stripWidthY)){
              //Invalid index to those that interact in dead space
               index_j=INVALID_INDEX;
          }
          else{
              index_j=indexT;
          }

      }

      return index_j;
}


 void GateGridDiscretization::SetGridPoints2D( int indexX, int indexY, G4ThreeVector& pos ){
     double posX;
     double posY;
     if(indexX>=0  && indexY>=0 ){
         posX=indexX*pitchX-(sizeVol[0]-(*im).second.stripWidthX-2*(*im).second.stripOffsetX)/2;
         posY=indexY*pitchY-(sizeVol[1]-(*im).second.stripWidthY-2*(*im).second.stripOffsetY)/2;


         pos.setX(posX);
          pos.setY(posY);
     }
     else{
         G4cout<<"problems with index (is  the right volume?)"<<G4endl;
     }


 }

//void GateGridDiscretization::GetGridPoints2D(G4ThreeVector & pos ){
//    double posX;
//    double posY;
//    unsigned int index_i;
//    unsigned int index_j;

//    //I should calculate this only once but I am not able of obtaining the quantities in the constructor
//   double pitchX=(sizeVol[0]-2*m_table[m_name].stripOffsetX)/(m_table[m_name].numberStripsX-1);
//   double pitchY=(sizeVol[1]-2*m_table[m_name].stripOffsetY)/(m_table[m_name].numberStripsY-1);
//   //##########  Change of reference system   #########
//   posX=pos.getX()+sizeVol[0]/2 -m_table[m_name].stripOffsetX+pitchX/2;
//   posY=pos.getY()+sizeVol[1]/2 -m_table[m_name].stripOffsetY+pitchY/2;
//   //##########  obtain index  #########
//  index_i=(int) posX/pitchX;
//  index_j=(int) posY/pitchY;
//    //##########  obtain position in the grid  #########
//  posX=-sizeVol[0]/2 +m_table[m_name].stripOffsetX+pitchX*index_i;
//  posY=-sizeVol[1]/2 +m_table[m_name].stripOffsetY+pitchY*index_j;

//  pos.setX(posX);
//   pos.setY(posY);


//}

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


