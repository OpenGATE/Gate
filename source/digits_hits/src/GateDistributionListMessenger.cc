/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/


#include "GateDistributionListManager.hh"
#include "GateDistributionListMessenger.hh"
#include "GateDistributionGauss.hh"
#include "GateDistributionExponential.hh"
#include "GateDistributionFlat.hh"
#include "GateDistributionFile.hh"
#include "GateDistributionManual.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3Vector.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithoutParameter.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithADouble.hh"

#include "G4UImanager.hh"

#include "GateListManager.hh"
#include "GateMessageManager.hh"

std::map<G4String,GateDistributionListMessenger::distType_t> GateDistributionListMessenger::fgkTypes;
// constructor
GateDistributionListMessenger::GateDistributionListMessenger(GateDistributionListManager* itsListManager)
  : GateListMessenger(itsListManager)
  ,m_distribVector()
{
  G4String guidance;

  guidance = G4String("Control the GATE Distributions" );
  SetDirectoryGuidance(guidance);

  G4String cmdName;

  if (fgkTypes.empty()){
    fgkTypes["File"]         =kFile;
    fgkTypes["Manual"]       =kManual;
    fgkTypes["Gaussian"]     =kGaussian;
    fgkTypes["Exponential"]  =kExponential;
    fgkTypes["Flat"]         =kFlat;
  }
}

// destructor
GateDistributionListMessenger::~GateDistributionListMessenger()
{
}

// Lists all the Distribution-names into a string
const G4String& GateDistributionListMessenger::DumpMap()
{
  static G4String ans="";
  if (ans.empty())
    for (std::map<G4String,distType_t>::const_iterator it=fgkTypes.begin()
           ; it != fgkTypes.end()
	   ;++it)
      ans += (*it).first+' ';
  return ans;
}


// Pure virtual method: create and insert a new attachment
void GateDistributionListMessenger::DoInsertion(const G4String& typeName)
{
  // G4cout << " GateDistributionListMessenger::DoInsertion \n";

  if (fgkTypes.find(typeName) == fgkTypes.end()) return;
  if (GetNewInsertionBaseName().empty())
    SetNewInsertionBaseName(typeName);

  AvoidNameConflicts();
  G4String name = GetListManager()->MakeElementName(GetNewInsertionBaseName());

  // G4cout<<"Creating element "<<name<< Gateendl;
  switch (fgkTypes[typeName]){
  case kFile         : m_distribVector.push_back(new GateDistributionFile(name)) ; break;
  case kManual       : m_distribVector.push_back(new GateDistributionManual(name)) ; break;
  case kGaussian     : m_distribVector.push_back(new GateDistributionGauss(name)) ; break;
  case kExponential  : m_distribVector.push_back(new GateDistributionExponential(name)) ; break;
  case kFlat         : m_distribVector.push_back(new GateDistributionFlat(name)) ; break;
  }
}
