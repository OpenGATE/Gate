

#include "GateStripSpatialDiscretization.hh"

#include "G4UnitsTable.hh"
#include "GateVolumeID.hh"
#include "GateStripSpatialDiscretizationMessenger.hh"
#include "G4Electron.hh"
#include "GateConfiguration.h"
#include "GateObjectStore.hh"
#include "GateConstants.hh"
#include "GateTools.hh"

#include "GateVVolume.hh"
#include "G4VoxelLimits.hh"




GateStripSpatialDiscretization::GateStripSpatialDiscretization(GatePulseProcessorChain* itsChain,
                                                               const G4String& itsName)
    : GateVPulseProcessor(itsChain,itsName)
{


 m_messenger = new GateStripSpatialDiscretizationMessenger(this);

    DescribeMyself(1);
    //GateMap<G4String,param> ::iterator im;





    sizeVol=new double [2];

   G4cout<<"access to m_name in constructor"<< m_name<<G4endl;

//Es como que lo guarda mas tarde
   im=m_table.begin();

   G4cout<<(*im).second.numberStripsX<<G4endl;

   // G4cout<<(G4String)(*im).first<<G4endl;
    //No tengo ahi acceso a los parameteros
   //G4cout<<"access to m_param number of strips in constructor"<< m_param.numberStripsX<<G4endl;
   // G4cout<<"access to m_param number of strips in constructor"<<   m_table[m_name].numberStripsX<<G4endl;
    //No recupero parametros introducidos
       //pitchX=(sizeVol[0]-2*)
}

GateStripSpatialDiscretization::~GateStripSpatialDiscretization()
{
	delete m_messenger;
}


G4int GateStripSpatialDiscretization::ChooseVolume(G4String val)
{

  GateObjectStore* m_store = GateObjectStore::GetInstance();


  if (m_store->FindCreator(val)!=0) {
      m_param.threshold = -1;
      m_param.numberStripsX= -1;
      m_param.numberStripsY= -1;
      m_param.stripOffsetX=-1;
      m_param.stripOffsetY=-1;
      m_table[val] = m_param;


      G4VoxelLimits limits;
      G4double min, max;
      G4AffineTransform at;

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


void GateStripSpatialDiscretization::ProcessOnePulse(const GatePulse* inputPulse,GatePulseList& outputPulseList)
{

     int current_indexX=-1;
     int current_indexY=-1;
    im=m_table.find(((inputPulse->GetVolumeID()).GetBottomCreator())->GetObjectName());
    GatePulse* outputPulse = new GatePulse(*inputPulse);

    if(im != m_table.end()){//Be careful repeated volumes have the same name
        //Meter checks sobre los parametrps
       // G4cout<<"Name finded"<<((inputPulse->GetVolumeID()).GetBottomCreator())->GetObjectName()<<G4endl;
         //AQUI SI SE CARGA EL VALOR
        //G4cout<<"Numb stripsX"<<(*im).second.numberStripsX<<G4endl;

              //I have change the pulse position for the corresponding position in the grid
              G4ThreeVector PosLocal = outputPulse->GetVolumeID().MoveToBottomVolumeFrame(outputPulse->GetGlobalPos()); //local (deberia ser igual que la local guardada)
             //G4cout<<" X posiiton antes"<<PosLocal.getX()<<G4endl;
             current_indexX=GetXIndex(PosLocal.getX());
             current_indexY=GetYIndex(PosLocal.getY());
             SetGridPoints2D(current_indexX, current_indexY, PosLocal);
             //GetGridPoints2D(PosLocal);
             outputPulse->SetLocalPos(PosLocal);
             outputPulse->SetGlobalPos(outputPulse->GetVolumeID().MoveToAncestorVolumeFrame(outputPulse->GetLocalPos()));

             if(current_indexX>=0 &&current_indexY>=0){
                 //Check output pulses from the end if there is a pulse with the same indexX an Y are summed in energy otherwise  a new input
                 //time of the pulse maybe also should be changed (time, energy,volume, position ane eventID are the same)
                 if(outputPulseList.empty()){
                     outputPulseList.push_back(outputPulse);
                     index_X_list.push_back(current_indexX);
                     index_Y_list.push_back(current_indexY);

                 }
                 else{

                     std::vector<G4int>::iterator it_indexX = std::find (index_X_list.begin(), index_X_list.end(), current_indexX);
                     if (it_indexX != index_X_list.end()){
                         //Coincidnecia en indices en X
                         int posListX=std::distance(index_X_list.begin(),it_indexX);
                         //G4cout<<"pos="<<posListX<<"number of pulses "<<outputPulseList.size()<<G4endl;
                         //See if I have also coincidence in Y indexes (in the same position)
                        if(index_Y_list.at(posListX)==current_indexY){
                             //Tengo que sumar la energia y toma min de los tiempos
                             //This does not change the posiiton
                             // (*iter)->CentroidMergeComptPhotIdeal(inputPulse);
                               //This sum the energy and cmpute the centroid of positio but being the same position the resutls should be the same
                            if(outputPulseList.at(posListX)->GetVolumeID()==outputPulse->GetVolumeID()){// tengo que checkear que estemos en el mismo volumen y no en una copia
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

                     }
                     else{

                         outputPulseList.push_back(outputPulse);
                         index_X_list.push_back(current_indexX);
                         index_Y_list.push_back(current_indexY);
                     }



                 }


             }
             else{
                 //Problems identifying the indexes
             }

    }
    else{

        //If the pulse is not in the selected volume we do not process it
        outputPulseList.push_back(outputPulse);
        index_X_list.push_back(current_indexX);
        index_Y_list.push_back(current_indexY);
    }





}


GatePulseList* GateStripSpatialDiscretization::ProcessPulseList(const GatePulseList* inputPulseList)
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
  //Apply energythreshold4Activation
  //No me deja llamar a la funcion ahi dnetro ???????
  ApplyEnergyThreshold(*outputPulseList);
  ApplyNeighbouringConditions(*outputPulseList);


  if (nVerboseLevel==1) {
      G4cout << "[" << GetObjectName() << "::ProcessPulseList]: returning output pulse-list with " << outputPulseList->size() << " entries\n";
      for (iter = outputPulseList->begin() ; iter != outputPulseList->end() ; ++iter)
        G4cout << **iter << Gateendl;
      G4cout << Gateendl;
  }

  return outputPulseList;
}


void GateStripSpatialDiscretization::ApplyEnergyThreshold( GatePulseList& outputPulseList){
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


 void GateStripSpatialDiscretization::ApplyNeighbouringConditions(GatePulseList& outputPulseList){


     std::vector<int> copyNumberLocalV;
     if(outputPulseList.size()>1){

        bool flagDeleteoutList=false;
         GatePulseList::iterator iterIntern;
         for (iterIntern = outputPulseList.begin() ; iterIntern != outputPulseList.end() ; ++iterIntern ){
             copyNumberLocalV.push_back(-1);
             im=m_table.find((((*iterIntern)->GetVolumeID()).GetBottomCreator())->GetObjectName());
            //G4cout<<"volume copy number"<< ((*iterIntern)->GetVolumeID()).GetBottomVolume()->GetCopyNo()<<G4endl;
             if(im != m_table.end()){
                 copyNumberLocalV.at(std::distance(outputPulseList.begin(),iterIntern))=((*iterIntern)->GetVolumeID()).GetBottomVolume()->GetCopyNo();

             }
         }
         int NumCopies=*std::max_element( copyNumberLocalV.begin(),  copyNumberLocalV.end())+1;
         std::multiset<int> mymultiset (copyNumberLocalV.begin(),copyNumberLocalV.end());
         for(int i=0; i<NumCopies; i++){

             if( mymultiset.count(i)>=2){
                 //two pulses in the same strip volume we throw the event
                 //Alguna condicion para utilizar senales en strips continuos en vez de tirar todo
                 //        //Quiza lo logico permitir 4 puntos alrededor de strips (differencia  indices en X e Y  (en mabos) menor o igual que 1)
                 flagDeleteoutList=true;
                 break;
             }


         }
         if(flagDeleteoutList==true){
             while (outputPulseList.size()) {
                 delete outputPulseList.back();
                 outputPulseList.erase(outputPulseList.end()-1);
             }
         }
     }






}


void GateStripSpatialDiscretization::DescribeMyself(size_t  indent)
{
    for (im=m_table.begin(); im!=m_table.end(); im++)
        G4cout << GateTools::Indent(indent) << "Threshold of " << (*im).first << ":\n"
           << GateTools::Indent(indent+1) << G4BestUnit( (*im).second.threshold,"Energy")<< "  and number of strips in X"
           << (*im).second.numberStripsX<<  Gateendl;
}

double GateStripSpatialDiscretization::GetXIndex(G4double posX){

     unsigned int index_i;
     //I should calculate this only once but I am not able of obtaining the quantities in the constructor
      pitchX=(sizeVol[0]-2*m_table[m_name].stripOffsetX)/(m_table[m_name].numberStripsX-1);
      double pos=posX+sizeVol[0]/2 -m_table[m_name].stripOffsetX+pitchX/2;
      index_i=(int) pos/pitchX;
      return index_i;
}

double GateStripSpatialDiscretization::GetYIndex(G4double posY){

     unsigned int index_j;
     //I should calculate this only once but I am not able of obtaining the quantities in the constructor
     pitchY=(sizeVol[1]-2*m_table[m_name].stripOffsetY)/(m_table[m_name].numberStripsY-1);
      double pos=posY+sizeVol[1]/2 -m_table[m_name].stripOffsetY+pitchY/2;
      index_j=(int) pos/pitchY;
      return index_j;
}


 void GateStripSpatialDiscretization::SetGridPoints2D( int indexX, int indexY, G4ThreeVector& pos ){
     double posX;
     double posY;
     if(indexX>=0  && indexY>=0 ){
        posX=-sizeVol[0]/2 +m_table[m_name].stripOffsetX+pitchX*indexX;
         posY=-sizeVol[1]/2 +m_table[m_name].stripOffsetY+pitchY*indexY;

         pos.setX(posX);
          pos.setY(posY);
     }
     else{
         G4cout<<"problems with index (is  the right volume?)"<<G4endl;
     }


 }

//void GateStripSpatialDiscretization::GetGridPoints2D(G4ThreeVector & pos ){
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
inline void GateStripSpatialDiscretization::PulsePushBack(const GatePulse* inputPulse, GatePulseList& outputPulseList)
{
	GatePulse* outputPulse = new GatePulse(*inputPulse);
    if (nVerboseLevel>1)
		G4cout << "Created new pulse for volume " << inputPulse->GetVolumeID() << ".\n"
		<< "Resulting pulse is: \n"
		<< *outputPulse << Gateendl << Gateendl ;
	outputPulseList.push_back(outputPulse);
}


