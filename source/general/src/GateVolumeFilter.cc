/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
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

   for(std::vector<G4LogicalVolume*>::iterator k =theListOfLogicalVolume.begin(); k!=theListOfLogicalVolume.end();k++)
            if(*k == currentVol ) return true;
   
   return false;
}
//---------------------------------------------------------------------------


//---------------------------------------------------------------------------
G4bool GateVolumeFilter::Accept(const G4Track* t)
{
   if(!IsInitialized) Initialize();

   G4TouchableHistory* theTouchable = (G4TouchableHistory*)(t->GetTouchable());
   G4LogicalVolume * currentVol = theTouchable->GetVolume(0)->GetLogicalVolume();

   for(std::vector<G4LogicalVolume*>::iterator k =theListOfLogicalVolume.begin(); k!=theListOfLogicalVolume.end();k++)
            if(*k == currentVol ) return true;
   
   return false;
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
  
  for(std::vector<G4String>::iterator k =theTempoListOfVolumeName.begin(); k!=theTempoListOfVolumeName.end();k++)
  {
    GateVVolume * mGateVolume = GateObjectStore::GetInstance()->FindVolumeCreator(*k);
    std::vector<GateVVolume *> theListOfTempoGateVVolume;
    if(mGateVolume)
    {
      theListOfVolume.push_back(mGateVolume);
      GateObjectChildList * child = mGateVolume->GetTheChildList();
      for(GateObjectChildList::iterator i=child->begin();i!=child->end();i++)
      {
         theListOfTempoGateVVolume.push_back((GateVVolume*)(*i));
      }
      if(theListOfTempoGateVVolume.size()>0)
      {
         mGateVolume = theListOfTempoGateVVolume[0];
         int it=0;
         while(mGateVolume)
         {
           theListOfVolume.push_back(mGateVolume);
           child = mGateVolume->GetTheChildList();
           for(GateObjectChildList::iterator i=child->begin();i!=child->end();i++)
           {
              theListOfTempoGateVVolume.push_back((GateVVolume*)(*i));
           }   
	   it++;
	   if(it>=(int)theListOfTempoGateVVolume.size()) mGateVolume=0;
	   else mGateVolume = theListOfTempoGateVVolume[it];  
         }
       } 
    }
    else GateError("In GateVolumeFilter: "<<GetObjectName()<<" -> Volume "<<*k<<" does not exist!");
  }
    
  for(std::vector<GateVVolume *>::iterator k = theListOfVolume.begin();
		  k!=theListOfVolume.end(); k++)
  {    
    theListOfLogicalVolume.push_back((*k)->GetLogicalVolume());
  }
}
//---------------------------------------------------------------------------


//---------------------------------------------------------------------------
void GateVolumeFilter::show(){
  G4cout << "------Filter: "<<GetObjectName()<<" ------\n";


  G4cout << "-------------------------------------------\n";

}
//---------------------------------------------------------------------------
