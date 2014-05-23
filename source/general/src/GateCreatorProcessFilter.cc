/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#include "GateCreatorProcessFilter.hh"
#include "G4VProcess.hh"


//---------------------------------------------------------------------------
GateCreatorProcessFilter::GateCreatorProcessFilter(G4String name) : GateVFilter(name)
{
  creatorProcesses.clear();
  pMessenger = new GateCreatorProcessFilterMessenger(this);
}
//---------------------------------------------------------------------------


//---------------------------------------------------------------------------
GateCreatorProcessFilter::~GateCreatorProcessFilter()
{
  delete pMessenger;
}
//---------------------------------------------------------------------------


//---------------------------------------------------------------------------
G4bool GateCreatorProcessFilter::Accept(const G4Track* aTrack) 
{
  const G4VProcess *creatorProcess = aTrack->GetCreatorProcess();
  if (!creatorProcess) return false;

  G4String creatorProcessName = creatorProcess->GetProcessName();
  for (CreatorProcesses::const_iterator iter=creatorProcesses.begin(); iter!=creatorProcesses.end(); iter++)
    if (*iter==creatorProcessName)
      return true;

  return false;
}

//---------------------------------------------------------------------------
void GateCreatorProcessFilter::AddCreatorProcess(const G4String& processName)
{
  creatorProcesses.push_back(processName);
}
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
void GateCreatorProcessFilter::show(){
  GateVFilter::show();

  G4cout << "creatorProcesses=" << G4endl;
  for (CreatorProcesses::const_iterator iter=creatorProcesses.begin(); iter!=creatorProcesses.end(); iter++)
    G4cout << *iter << G4endl;
}
//---------------------------------------------------------------------------
