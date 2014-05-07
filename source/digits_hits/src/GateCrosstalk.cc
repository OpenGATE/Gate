/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateCrosstalk.hh"

#include "GateCrosstalkMessenger.hh"
#include "GateTools.hh"
#include <vector>
#include "G4ThreeVector.hh"

#include "GateVolumeID.hh"
#include "GateOutputVolumeID.hh"
#include "GateDetectorConstruction.hh"
#include "GateCrystalSD.hh"
#include "GateVSystem.hh"
#include "GateArrayParamsFinder.hh"

#include "G4UnitsTable.hh"

#include "GateObjectStore.hh"

// Static pointer to the GateCrosstalk singleton
GateCrosstalk* GateCrosstalk::theGateCrosstalk=0;

/*    	This function allows to retrieve the current instance of the GateCrosstalk singleton
      	If the GateCrosstalk already exists, GetInstance only returns a pointer to this singleton.
	If this singleton does not exist yet, GetInstance creates it by calling the private
	GateCrosstalk constructor
*/
GateCrosstalk* GateCrosstalk::GetInstance(GatePulseProcessorChain* itsChain,
					  const G4String& itsName, G4double itsEdgesFraction,
					  G4double itsCornersFraction)
{
  if (!theGateCrosstalk)
    if (itsChain)
      theGateCrosstalk = new GateCrosstalk(itsChain, itsName, itsEdgesFraction, itsCornersFraction);
  return theGateCrosstalk;
}


// Private constructor
GateCrosstalk::GateCrosstalk(GatePulseProcessorChain* itsChain,
			     const G4String& itsName, G4double itsEdgesFraction,
			     G4double itsCornersFraction)
  : GateVPulseProcessor(itsChain, itsName),
    m_edgesCrosstalkFraction(itsEdgesFraction),m_cornersCrosstalkFraction(itsCornersFraction)
{
  m_messenger = new GateCrosstalkMessenger(this);
  m_testVolume = 0;
}




GateCrosstalk::~GateCrosstalk()
{
  delete m_messenger;
  delete ArrayFinder;
}

void GateCrosstalk::CheckVolumeName(G4String val)
{
  //Retrieve the inserter store to check if the volume name is valid
  GateObjectStore* m_store = GateObjectStore::GetInstance();
  if (m_store->FindCreator(val)) {
    m_volume = val;
    //Find the array params
    ArrayFinder = new GateArrayParamsFinder(m_store->FindCreator(val),
						 m_nbX, m_nbY, m_nbZ);
    m_testVolume = 1;
  }
  else {
    G4cout << "Wrong Volume Name" << G4endl;
  }
}

void GateCrosstalk::ProcessOnePulse(const GatePulse* inputPulse,GatePulseList& outputPulseList)
{

  if(!m_testVolume)
    {
      G4cerr << 	G4endl << "[GateCrosstalk::ProcessOnePulse]:" << G4endl
	     <<   "Sorry, but you don't have choosen any volume !" << G4endl;

			G4String msg = "You must choose a volume for crosstalk, e.g. crystal:\n"
      "\t/gate/digitizer/Singles/crosstalk/chooseCrosstalkVolume VOLUME NAME\n"
      "or disable the crosstalk using:\n"
      "\t/gate/digitizer/Singles/crosstalk/disable\n";

			G4Exception( "GateCrosstalk::ProcessOnePulse", "ProcessOnePulse", FatalException, msg );
    }

  //Find the pulse position in the array
  m_depth = (size_t)(inputPulse->GetVolumeID().GetCreatorDepth(m_volume));
  ArrayFinder->FindInputPulseParams(inputPulse->GetVolumeID().GetCopyNo(m_depth), m_i, m_j, m_k);

  //Numbers of edge and corner neighbors for the pulses
  G4int countE = 0;
  G4int countC = 0;

  // Find the possible neighbors
  if (m_edgesCrosstalkFraction != 0) {
    if (m_i != 0) {
      outputPulseList.push_back(CreatePulse(m_edgesCrosstalkFraction, inputPulse, m_i - 1, m_j, m_k));
      countE++;
    }
    if (m_i != m_nbX - 1) {
      outputPulseList.push_back(CreatePulse(m_edgesCrosstalkFraction, inputPulse, m_i + 1, m_j, m_k));
      countE++;
    }
    if (m_j != 0) {
      outputPulseList.push_back(CreatePulse(m_edgesCrosstalkFraction, inputPulse, m_i, m_j - 1, m_k));
      countE++;
    }
    if (m_j != m_nbY - 1) {
      outputPulseList.push_back(CreatePulse(m_edgesCrosstalkFraction, inputPulse, m_i, m_j + 1, m_k));
      countE++;
    }
    if (m_k != 0) {
      outputPulseList.push_back(CreatePulse(m_edgesCrosstalkFraction, inputPulse, m_i, m_j, m_k - 1));
      countE++;
    }
    if (m_k != m_nbZ - 1) {
      outputPulseList.push_back(CreatePulse(m_edgesCrosstalkFraction, inputPulse, m_i, m_j, m_k + 1));
      countE++;
    }
  }

  if (m_cornersCrosstalkFraction != 0) {
    if ((m_i != 0) & (m_j != 0)) {
      outputPulseList.push_back(CreatePulse(m_cornersCrosstalkFraction, inputPulse, m_i - 1, m_j - 1, m_k));
      countC++;
    }
    if ((m_i != 0) & (m_j != m_nbY - 1)) {
      outputPulseList.push_back(CreatePulse(m_cornersCrosstalkFraction, inputPulse, m_i - 1, m_j + 1, m_k));
      countC++;
    }
    if ((m_i != 0) & (m_k != 0)) {
      outputPulseList.push_back(CreatePulse(m_cornersCrosstalkFraction, inputPulse, m_i - 1, m_j, m_k - 1));
      countC++;
    }
    if ((m_i != 0) & (m_k != m_nbZ - 1)) {
      outputPulseList.push_back(CreatePulse(m_cornersCrosstalkFraction, inputPulse, m_i - 1, m_j, m_k + 1));
      countC++;
    }
    if ((m_i != m_nbX - 1) & (m_j != 0)) {
      outputPulseList.push_back(CreatePulse(m_cornersCrosstalkFraction, inputPulse, m_i + 1, m_j - 1, m_k));
      countC++;
    }
    if ((m_i != m_nbX - 1) & (m_j != m_nbY - 1)) {
      outputPulseList.push_back(CreatePulse(m_cornersCrosstalkFraction, inputPulse, m_i + 1, m_j + 1, m_k));
      countC++;
    }
    if ((m_i != m_nbX - 1) & (m_k != 0)) {
      outputPulseList.push_back(CreatePulse(m_cornersCrosstalkFraction, inputPulse, m_i + 1, m_j, m_k - 1));
      countC++;
    }
    if ((m_i != m_nbX - 1) & (m_k != m_nbZ - 1)) {
      outputPulseList.push_back(CreatePulse(m_cornersCrosstalkFraction, inputPulse, m_i + 1, m_j, m_k + 1));
      countC++;
    }
    if ((m_j != 0) & (m_k != 0)) {
      outputPulseList.push_back(CreatePulse(m_cornersCrosstalkFraction, inputPulse, m_i, m_j - 1, m_k - 1));
      countC++;
    }
    if ((m_j != 0) & (m_k != m_nbZ - 1)) {
      outputPulseList.push_back(CreatePulse(m_cornersCrosstalkFraction, inputPulse, m_i, m_j - 1, m_k + 1));
      countC++;
    }

    if ((m_j != m_nbY - 1) & (m_k != 0)) {
      outputPulseList.push_back(CreatePulse(m_cornersCrosstalkFraction, inputPulse, m_i, m_j + 1, m_k - 1));
      countC++;
    }
    if ((m_j != m_nbY - 1) & (m_k != m_nbZ - 1)) {
      outputPulseList.push_back(CreatePulse(m_cornersCrosstalkFraction, inputPulse, m_i, m_j + 1, m_k + 1));
      countC++;
    }
  }

  // Check if the energy of neighbors is not higher than the energy of the incident pulse
  G4double energytot = inputPulse->GetEnergy()*((countE*m_edgesCrosstalkFraction)+(countC*m_cornersCrosstalkFraction));
  if(energytot>=inputPulse->GetEnergy())
    {
      G4cerr << 	G4endl << "[GateCrosstalk::ProcessOnePulse]:" << G4endl
	     <<   "Sorry, but you have too much energy !" << G4endl;

			G4String msg = "You must change your fractions of energy for the close crystals :\n"
      "\t/gate/digitizer/Singles/crosstalk/setSidesFraction NUMBER\n"
      "\t/gate/digitizer/Singles/crosstalk/setCornersFraction NUMBER\n"
      "or disable the crosstalk using:\n"
      "\t/gate/digitizer/Singles/crosstalk/disable\n";
			G4Exception( "GateCrosstalk::ProcessOnePulse", "ProcessOnePulse", FatalException,msg);
    }
  // Add the incident pulse in the pulse list with less energy
  GatePulse* outputPulse = new GatePulse(*inputPulse);
  m_XtalkpCent = (1-(4*m_edgesCrosstalkFraction+4*m_cornersCrosstalkFraction));
  outputPulse->SetEnergy((inputPulse->GetEnergy())*m_XtalkpCent);
  outputPulseList.push_back(outputPulse);
  if (nVerboseLevel>1)
    G4cout << "the input pulse created " << countE+countC << " pulses around it"
	   << G4endl;
}


GateVolumeID GateCrosstalk::CreateVolumeID(const GateVolumeID* aVolumeID, G4int i, G4int j, G4int k)
{
  GateVolumeID aVolumeIDOut;
  for (size_t n = 0; n < aVolumeID->size(); n++)
    if (n != m_depth)
      aVolumeIDOut.push_back(GateVolumeSelector(aVolumeID->GetVolume(n)));
    else {
      GateVVolume* anInserter =  aVolumeID->GetCreator(m_depth);
      G4VPhysicalVolume* aVolume = anInserter->GetPhysicalVolume(i + m_nbX * j + m_nbX * m_nbY * k);
      aVolumeIDOut.push_back(GateVolumeSelector(aVolume));
    }
  return aVolumeIDOut;
}

GateOutputVolumeID GateCrosstalk::CreateOutputVolumeID(const GateVolumeID aVolumeID)
{
  GateDetectorConstruction* aDetectorConstruction = GateDetectorConstruction::GetGateDetectorConstruction();
  GateOutputVolumeID anOutputVolumeID = aDetectorConstruction->GetCrystalSD()->GetSystem()->ComputeOutputVolumeID(aVolumeID);
  return anOutputVolumeID;
}

GatePulse* GateCrosstalk::CreatePulse(G4double val, const GatePulse* pulse, G4int i, G4int j, G4int k)
{
  GatePulse* apulse = new GatePulse(pulse);
  apulse->SetLocalPos(G4ThreeVector(0,0,0));
  apulse->SetVolumeID(CreateVolumeID(&pulse->GetVolumeID(), i, j, k));
  apulse->SetGlobalPos(apulse->GetVolumeID().MoveToAncestorVolumeFrame(apulse->GetLocalPos()));
  apulse->SetOutputVolumeID(CreateOutputVolumeID(apulse->GetVolumeID()));
  apulse->SetEnergy(pulse->GetEnergy()*val);
  return apulse;
}

void GateCrosstalk::DescribeMyself(size_t indent)
{
  G4cout << GateTools::Indent(indent) << "Optical crosstalk for " << m_volume << ":\n"
	 << GateTools::Indent(indent+1) << "fraction of energy for side crystals: " << m_edgesCrosstalkFraction << "\n"
	 << GateTools::Indent(indent+1) << "fraction of energy for corner crystals: " << m_cornersCrosstalkFraction << G4endl;
}
