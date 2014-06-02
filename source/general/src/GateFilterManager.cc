/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateFilterManager.hh"



//---------------------------------------------------------------------------
GateFilterManager::GateFilterManager(G4String name)
  : G4VSDFilter(name)
{
  theFilters.clear();
  mFilterName = name;
}
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
GateFilterManager::~GateFilterManager()
{
  GateVFilter* filter;
  while(!theFilters.empty())
  {
    // get first 'element'
    filter = theFilters.front();
        
    // remove it from the list
    theFilters.erase(theFilters.begin());

    // delete the pointer
    delete filter;
  }
}
//---------------------------------------------------------------------------


//---------------------------------------------------------------------------
G4bool GateFilterManager::Accept(const G4Step* aStep) const
{
//  std::vector<GateVFilter*>::iterator sit;
 // for(sit= theFilters.begin(); sit!=theFilters.end(); ++sit)
      //if(!(*sit)->Accept(aStep)) return false;

  for(unsigned int i = 0;i<theFilters.size();i++)
     if(!theFilters[i]->Accept(aStep)) return false;

  return true;
}
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
G4bool GateFilterManager::Accept(const G4Track* aTrack) const
{
//  std::vector<GateVFilter*>::iterator sit;
 // for(sit= theFilters.begin(); sit!=theFilters.end(); ++sit)
      //if(!(*sit)->Accept(aStep)) return false;

    for(unsigned int i = 0;i<theFilters.size();i++)
      if(!theFilters[i]->Accept(aTrack)) return false;

  return true;
}
//---------------------------------------------------------------------------



//---------------------------------------------------------------------------
void GateFilterManager::show(){
  G4cout << "------Filter Manager: "<<mFilterName<<" ------"<<G4endl;

  std::vector<GateVFilter*>::iterator sit;
  for(sit= theFilters.begin(); sit!=theFilters.end(); ++sit)
     (*sit)->show();
  
  G4cout << "-------------------------------------------"<<G4endl;
}
//---------------------------------------------------------------------------


