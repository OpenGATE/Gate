/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateSigmoidalThresholder.hh"

#include "G4UnitsTable.hh"
#include "Randomize.hh"

#include "GateSigmoidalThresholderMessenger.hh"
#include "GateTools.hh"
#include "GateCrosstalk.hh"
#include "GateTransferEfficiency.hh"
#include "GateQuantumEfficiency.hh"
#include "GateLightYield.hh"


GateSigmoidalThresholder::GateSigmoidalThresholder(GatePulseProcessorChain* itsChain,
						   const G4String& itsName, G4double itsThreshold,
						   G4double itsAlpha, G4double itsAcceptance)
  : GateVPulseProcessor(itsChain,itsName),
    m_threshold(itsThreshold),
    m_perCent(itsAcceptance),
    m_alpha(itsAlpha)
{
  m_messenger = new GateSigmoidalThresholderMessenger(this);
  m_check  = false;
}




GateSigmoidalThresholder::~GateSigmoidalThresholder()
{
  delete m_messenger;
}



void GateSigmoidalThresholder::ProcessOnePulse(const GatePulse* inputPulse,GatePulseList& outputPulseList)
{
  if (!m_check)
    {
      G4double crosstalk = (GateCrosstalk::GetInstance(NULL,"name", 0, 0)) ?
	GateCrosstalk::GetInstance(NULL,"name", 0, 0)->GetXTPerCent() : 1.;

      G4double lightOutput;
      G4String crystalName;
      G4bool checkRi = false;
      if (GateLightYield::GetInstance(NULL,"name")) {
	lightOutput = GateLightYield::GetInstance(NULL,"name")->GetMinLightOutput();
	crystalName = GateLightYield::GetInstance(NULL,"name")->GetMinLightOutputName();
	checkRi = true;
      }
      else
	lightOutput = 1.;

      G4double TECoef;
      if (GateTransferEfficiency::GetInstance(NULL,"name"))
	TECoef = (!checkRi) ?
	  GateTransferEfficiency::GetInstance(NULL,"name")->GetTEMin() :
	  GateTransferEfficiency::GetInstance(NULL,"name")->GetTECrystCoeff(crystalName);
      else
	TECoef = 1.;

      G4double QECoef = (GateQuantumEfficiency::GetInstance(NULL,"name")) ?
	GateQuantumEfficiency::GetInstance(NULL,"name")->GetMinQECoeff() : 1.;

//       G4cout << "crosstalk: " << crosstalk << "\tlightOutput: " << lightOutput << "\tTECoef: " << TECoef
// 	     << "\tQECoef: " << QECoef << G4endl;


      m_centSigm = m_threshold;
//       G4cout << "m_centSigm: " << m_centSigm;
      m_centSigm /= (1-1./m_alpha * log(1./m_perCent - 1));
//       G4cout << "\tm_centSigm: " << m_centSigm;
      m_centSigm *= crosstalk * lightOutput * TECoef * QECoef;
//       G4cout << "\tm_centSigm: " << m_centSigm << G4endl;
      m_check = true;
    }

  if (!inputPulse) {
    if (nVerboseLevel>1)
      	G4cout << "[GateSigmoidalThresholder::ProcessOnePulse]: input pulse was null -> nothing to do\n\n";
    return;
  }
  if (inputPulse->GetEnergy()==0) {
    if (nVerboseLevel>1)
      	G4cout << "[GateSigmoidalThresholder::ProcessOneHit]: energy is null for " << inputPulse << " -> pulse ignored\n\n";
    return;
  }

  G4double sigma=SigmoideFct(m_alpha*(inputPulse->GetEnergy()-m_centSigm)/m_centSigm);
  G4double rmd=(((G4double) rand ()) / 2147483647.0);
  if ( sigma >= rmd )
  {
    GatePulse* outputPulse = new GatePulse(*inputPulse);
    outputPulseList.push_back(outputPulse);
    if (nVerboseLevel>1)
      	G4cout << "Copied pulse to output:" << G4endl
      	       << *outputPulse << G4endl << G4endl ;
  }
  else {
      if (nVerboseLevel>1)
      	G4cout << "Ignored pulse with energy below threshold:" << G4endl
      	       << *inputPulse << G4endl << G4endl ;
  }
  if (nVerboseLevel>1)
    {
      GatePulseIterator iter;
      G4cout << "----Pulse List after Thresholder--------------------------------------------------------" << G4endl;
      G4cout << "----Threshold central: " << m_centSigm << " --------------------------------------------------------" << G4endl;
      for (iter = outputPulseList.begin() ; iter != outputPulseList.end() ; ++iter )
	G4cout << "VolumeID: " << (*iter)->GetOutputVolumeID()
	       << "\tEnergy: " << (*iter)->GetEnergy() << G4endl;
      G4cout << "------------------------------------------------------------------------------------" << G4endl;
    }
}

void GateSigmoidalThresholder::DescribeMyself(size_t indent)
{
  G4cout << GateTools::Indent(indent) << "Sigmoidal threshold: " << G4BestUnit(m_threshold,"Energy") << G4endl;
  G4cout << GateTools::Indent(indent+1) << "with a percentage of acceptance equal to " << m_perCent << G4endl;
  G4cout << GateTools::Indent(indent+1) << "parameter alpha of the sigmoidal function:  " << m_alpha << G4endl;
}
