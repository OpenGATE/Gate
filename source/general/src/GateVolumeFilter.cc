/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#include "GateVolumeFilter.hh"

#include "GateObjectStore.hh"
#include "GateObjectChildList.hh"

#include "G4TouchableHistory.hh"

//---------------------------------------------------------------------------
GateVolumeFilter::GateVolumeFilter(G4String name)
  :GateVFilter(name)
{
  IsInitialized=false;
  pVolumeFilterMessenger = new GateVolumeFilterMessenger(this);
}
//---------------------------------------------------------------------------


//---------------------------------------------------------------------------
G4bool GateVolumeFilter::Accept(const G4Step* aStep) 
{
   if(!IsInitialized) Initialize();

   G4TouchableHistory* theTouchable = (G4TouchableHistory*)(aStep->GetPreStepPoint()->GetTouchable());
   G4LogicalVolume * currentVol = theTouchable->GetVolume(0)->GetLogicalVolume();

   for(unsigned int k =0 ; k<theListOfLogicalVolume.size();k++)
            if(theListOfLogicalVolume[k]== currentVol ) _FILTER_RETURN_WITH_INVERSION true;
   
   _FILTER_RETURN_WITH_INVERSION false;
}
//---------------------------------------------------------------------------


//---------------------------------------------------------------------------
G4bool GateVolumeFilter::Accept(const G4Track* t)
{
   if(!IsInitialized) Initialize();

   G4TouchableHistory* theTouchable = (G4TouchableHistory*)(t->GetTouchable());
   G4LogicalVolume * currentVol = theTouchable->GetVolume(0)->GetLogicalVolume();

   for(unsigned int k =0 ; k<theListOfLogicalVolume.size();k++)
            if(theListOfLogicalVolume[k]== currentVol ) _FILTER_RETURN_WITH_INVERSION true;
   
   _FILTER_RETURN_WITH_INVERSION false;
}

//---------------------------------------------------------------------------
void GateVolumeFilter::addVolume(G4String volName)
{
  theTempoListOfVolumeName.push_back(volName);
}
//---------------------------------------------------------------------------


//---------------------------------------------------------------------------
void GateVolumeFilter::Initialize()
{
  IsInitialized=true;
  
  for(unsigned int k =0 ; k<theTempoListOfVolumeName.size();k++)
  {
    GateVVolume * mGateVolume = GateObjectStore::GetInstance()->FindVolumeCreator(theTempoListOfVolumeName[k]);
    std::vector<GateVVolume *> theListOfTempoGateVVolume;
    if(mGateVolume)
    {
      theListOfVolume.push_back(mGateVolume);
      GateObjectChildList * child = mGateVolume->GetTheChildList();
      for(unsigned int i =0;i<child->size();i++)
      {
         theListOfTempoGateVVolume.push_back(child->GetVolume(i));
      }
      if(theListOfTempoGateVVolume.size()>0)
      {
         mGateVolume = theListOfTempoGateVVolume[0];
         int it=0;
         while(mGateVolume)
         {
           theListOfVolume.push_back(mGateVolume);
           child = mGateVolume->GetTheChildList();
           for(unsigned int i =0;i<child->size();i++)
           {
              theListOfTempoGateVVolume.push_back(child->GetVolume(i));
           }   
	   it++;
	   if(it>=(int)theListOfTempoGateVVolume.size()) mGateVolume=0;
	   else mGateVolume = theListOfTempoGateVVolume[it];  
         }
       } 
    }
    else GateError("In GateVolumeFilter: "<<GetObjectName()<<" -> Volume "<<theTempoListOfVolumeName[k]<<" does not exist!");
  }
    
  for(unsigned int k =0 ; k<theListOfVolume.size();k++)
  {    
    theListOfLogicalVolume.push_back(theListOfVolume[k]->GetLogicalVolume());   
  }
}
//---------------------------------------------------------------------------


//---------------------------------------------------------------------------
void GateVolumeFilter::show(){
  G4cout << "------Filter: "<<GetObjectName()<<" ------\n";


  G4cout << "-------------------------------------------\n";

}
//---------------------------------------------------------------------------
