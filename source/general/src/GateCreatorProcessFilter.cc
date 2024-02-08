/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
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
  if (!creatorProcess) _FILTER_RETURN_WITH_INVERSION false;

  G4String creatorProcessName = creatorProcess->GetProcessName();
  for (CreatorProcesses::const_iterator iter=creatorProcesses.begin(); iter!=creatorProcesses.end(); iter++)
    if (*iter==creatorProcessName)
      _FILTER_RETURN_WITH_INVERSION true;

  _FILTER_RETURN_WITH_INVERSION false;
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

  G4cout << "creatorProcesses=\n";
  for (CreatorProcesses::const_iterator iter=creatorProcesses.begin(); iter!=creatorProcesses.end(); iter++)
    G4cout << *iter << Gateendl;
}
//---------------------------------------------------------------------------
