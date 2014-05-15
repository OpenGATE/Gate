/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateCalibration.hh"

#include "G4UnitsTable.hh"

#include "GateCalibrationMessenger.hh"
#include "GateTransferEfficiency.hh"
#include "GateQuantumEfficiency.hh"
#include "GateLightYield.hh"
#include "GateTools.hh"
#include "GateVolumeID.hh"


GateCalibration::GateCalibration(GatePulseProcessorChain* itsChain,
				 const G4String& itsName)
  : GateVPulseProcessor(itsChain, itsName)
{
  m_messenger = new GateCalibrationMessenger(this);
  m_calib = 1;
}




GateCalibration::~GateCalibration()
{
  delete m_messenger;
}



void GateCalibration::ProcessOnePulse(const GatePulse* inputPulse,GatePulseList& outputPulseList)
{
  GatePulse* outputPulse = new GatePulse(*inputPulse);
  G4String LayerName = ((inputPulse->GetVolumeID()).GetBottomCreator())->GetObjectName();

  G4double lightOutput = (GateLightYield::GetInstance(NULL,"name")) ?
    GateLightYield::GetInstance(NULL,"name")->GetLightOutput(LayerName) : 1.;

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

  outputPulse->SetEnergy(inputPulse->GetEnergy()/(lightOutput*TECoef*QECoef)*m_calib);
  outputPulseList.push_back(outputPulse);
}


void GateCalibration::FindInputPulseParams(const GateVolumeID* aVolumeID)
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


void GateCalibration::DescribeMyself(size_t indent)
{
  G4cout << GateTools::Indent(indent) << "Calibration Nphotoelectrons->Energy " << G4endl;
}
