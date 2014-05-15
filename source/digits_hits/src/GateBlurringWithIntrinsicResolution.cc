/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateBlurringWithIntrinsicResolution.hh"
#include "GateBlurringWithIntrinsicResolutionMessenger.hh"
#include "GateObjectStore.hh"

#include "GateTools.hh"
#include "GateVolumeID.hh"
#include "GateCrosstalk.hh"
#include "GateTransferEfficiency.hh"
#include "GateQuantumEfficiency.hh"
#include "GateLightYield.hh"
#include "Randomize.hh"

#include "G4UnitsTable.hh"


// Constructor
GateBlurringWithIntrinsicResolution::GateBlurringWithIntrinsicResolution(GatePulseProcessorChain* itsChain,
									 const G4String& itsName)
  : GateVPulseProcessor(itsChain, itsName)
{
  m_messenger = new GateBlurringWithIntrinsicResolutionMessenger(this);
}

GateBlurringWithIntrinsicResolution::~GateBlurringWithIntrinsicResolution()
{
  delete m_messenger;
}

G4int GateBlurringWithIntrinsicResolution::ChooseVolume(G4String val)
{
  GateObjectStore* m_store = GateObjectStore::GetInstance();

  if (m_store->FindCreator(val)!=0) {
    m_param.resolution = -1;
    m_param.eref = -1;
    m_table[val] = m_param;
    return 1;
  }
  else {
    G4cout << "Wrong Volume Name" << G4endl;
    return 0;
  }
}

void GateBlurringWithIntrinsicResolution::ProcessOnePulse(const GatePulse* inputPulse,GatePulseList& outputPulseList)
{
  im=m_table.find(((inputPulse->GetVolumeID()).GetBottomCreator())->GetObjectName());
  GatePulse* outputPulse = new GatePulse(*inputPulse);
  if(im != m_table.end())
    {
      if((*im).second.resolution < 0 ) {
	G4cerr << 	G4endl << "[GateBlurringWithIntrinsicResolution::ProcessOnePulse]:" << G4endl
	       <<   "Sorry, but the resolution (" << (*im).second.resolution << ") for " << (*im).first << " is invalid" << G4endl;
	G4String msg = "You must set the energy of reference AND the resolution:\n\t/gate/digitizer/Singles/intrinsicResolutionBlurring/" + (*im).first + "/setEnergyOfReference ENERGY\n or disable the intrinsic resolution blurring using:\n\t/gate/digitizer/Singles/intrinsicResolutionBlurring/disable";
	G4Exception( "GateBlurringWithIntrinsicResolution::ProcessOnePulse", "ProcessOnePulse", FatalException, msg );
      }
      else if((*im).second.eref < 0) {
	G4cerr <<   G4endl << "[GateBlurringWithIntrinsicResolution::ProcessOnePulse]:" << G4endl
	       <<   "Sorry, but the energy of reference (" << G4BestUnit((*im).second.eref,"Energy") << ") for "
	       << (*im).first <<" is invalid" << G4endl;
	G4String msg = "You must set the resolution AND the energy of reference:\n\t/gate/digitizer/Singles/intrinsicResolutionBlurring/" + (*im).first + "/setEnergyOfReference ENERGY\n or disable the intrinsic resolution blurring using:\n\t/gate/digitizer/Singles/intrinsicResolutionBlurring/disable";
	G4Exception( "GateBlurringWithIntrinsicResolution::ProcessOnePulse", "ProcessOnePulse", FatalException, msg );
	}
      else {
	G4String LayerName = ((inputPulse->GetVolumeID()).GetBottomCreator())->GetObjectName();

	G4double XtalkpCent = (GateCrosstalk::GetInstance(NULL,"name",0.,0.)) ?
	  GateCrosstalk::GetInstance(NULL,"name",0.,0.)->GetXTPerCent() : 1.;

	G4double TECoef = (GateTransferEfficiency::GetInstance(NULL,"name")) ?
	  GateTransferEfficiency::GetInstance(NULL,"name")->GetTECrystCoeff(LayerName) : 1.;

	G4double QECoef;
	if (GateQuantumEfficiency::GetInstance(NULL,"name"))
	  {
	    m_volumeName = GateQuantumEfficiency::GetInstance(NULL,"name")->GetVolumeName();
	    FindInputPulseParams(&inputPulse->GetVolumeID());
	    G4int level2No = GateQuantumEfficiency::GetInstance(NULL,"name")->Getlevel2No();
	    G4int level3No = GateQuantumEfficiency::GetInstance(NULL,"name")->Getlevel3No();
	    G4int tableNB = m_k + m_j*level3No + m_i*level3No*level2No;
	    QECoef = GateQuantumEfficiency::GetInstance(NULL,"name")->GetQECoeff(tableNB, m_volumeIDNo);
	  }
	else
	  QECoef = 1.;

	G4double LightOutput = (GateLightYield::GetInstance(NULL,"name")) ?
	  GateLightYield::GetInstance(NULL,"name")->GetLightOutput(LayerName) : 1.;

	G4double mu = inputPulse->GetEnergy();
	G4double intrinsicResol = (*im).second.resolution
	                        * sqrt(((*im).second.eref * XtalkpCent * QECoef  * TECoef * LightOutput)
	                        / inputPulse->GetEnergy());

	G4double resol = sqrt((1.1/mu)*(2.35*2.35) + intrinsicResol*intrinsicResol);

	outputPulse->SetEnergy(G4RandGauss::shoot(mu,(resol * mu)/2.35));
      }
    }
  outputPulseList.push_back(outputPulse);
}

void GateBlurringWithIntrinsicResolution::FindInputPulseParams(const GateVolumeID* aVolumeID)
{
  m_depth = (size_t)(aVolumeID->GetCreatorDepth(m_volumeName));
  m_volumeIDNo = aVolumeID->GetCopyNo(m_depth);
  if (aVolumeID->GetCopyNo(m_depth-1)==-1) {
    m_k=0;
    m_j=0;
    m_i=0;
  }
  else {
    m_k = aVolumeID->GetCopyNo(m_depth-1);
    if (aVolumeID->GetCopyNo(m_depth-2)==-1) {
      m_j=0;
      m_i=0;
    }
    else {
      m_j = aVolumeID->GetCopyNo(m_depth-2);
      if(aVolumeID->GetCopyNo(m_depth-3)==-1)
        m_i=0;
      else
	m_i = aVolumeID->GetCopyNo(m_depth-3);
    }
  }
}

void GateBlurringWithIntrinsicResolution::DescribeMyself(size_t indent)
{
  for (im=m_table.begin(); im!=m_table.end(); im++)
    G4cout << GateTools::Indent(indent) << (*im).first << " :\n"
	 << GateTools::Indent(indent+1) << "Intrinsic resolution : " << (*im).second.resolution <<  "  @ "
	 << G4BestUnit((*im).second.eref,"Energy") <<  G4endl;
}
