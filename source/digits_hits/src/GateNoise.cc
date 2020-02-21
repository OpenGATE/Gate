/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/


#include "GateNoise.hh"

#include "GateNoiseMessenger.hh"
#include "GateTools.hh"
#include "GateVSystem.hh"
#include "GateSystemListManager.hh"
#include "GateVDistribution.hh"
#include "GateApplicationMgr.hh"
#include "Randomize.hh"
#include "CLHEP/Random/RandExponential.h"
#include "CLHEP/Random/RandFlat.h"

#include "G4UnitsTable.hh"
#include <fstream>


GateNoise::GateNoise(GatePulseProcessorChain* itsChain,
                     const G4String& itsName)
  : GateVPulseProcessor(itsChain,itsName),
    m_deltaTDistrib(),
    m_energyDistrib(),
    m_createdPulses("transiant")
{
  m_messenger = new GateNoiseMessenger(this);
  m_oldTime = -1;//GateApplicationMgr::GetInstance()->GetTimeStart();
}




GateNoise::~GateNoise()
{
  delete m_messenger;
}

GatePulseList* GateNoise::ProcessPulseList(const GatePulseList* inputPulseList)
{
  const G4int NoiseEvent_flag=-2;
    
  if (inputPulseList->empty()) return 0;

  if (!m_energyDistrib){
    G4cerr<<"GateNoise::ProcessPulseList : no energy distribution given. Nothing's done\n";
    return GateVPulseProcessor::ProcessPulseList(inputPulseList);
  }

  if (!m_deltaTDistrib){
    G4cerr<<"GateNoise::ProcessPulseList : no deltaT distribution given. Nothing's done\n";
    return GateVPulseProcessor::ProcessPulseList(inputPulseList);
  }
  
  G4double t0;
  if (m_oldTime<0) t0 = inputPulseList->ComputeStartTime();
  else t0=m_oldTime;
  
  m_oldTime = inputPulseList->ComputeFinishTime();
  while(t0+=m_deltaTDistrib->ShootRandom(),t0<m_oldTime){
    GatePulse* pulse = new GatePulse();
    pulse->SetTime(t0);
    pulse->SetEnergy(m_energyDistrib->ShootRandom());

    // now define a random outputVolumeID...
    GateVSystem* system = GateSystemListManager::GetInstance()->GetSystem(0);
    size_t depth = system->GetTreeDepth();
    GateOutputVolumeID outputVol(depth);
    //      G4cout<<"Choosing ";
    for (size_t i=0;i<depth;i++) {
      size_t max = system->ComputeNofElementsAtLevel(i);
      long n = CLHEP::RandFlat::shootInt((long int)0,(long int)(max));
      outputVol[i]=n;
      //          G4cout<<n<<' ';
    }
    //      G4cout<< Gateendl;
    pulse->SetOutputVolumeID(outputVol);
    GateVolumeID* volID = system->MakeVolumeID(outputVol);
    pulse->SetVolumeID(*volID);
    pulse->SetGlobalPos(system->ComputeObjectCenter(volID));
    delete volID;
    pulse->SetEventID(NoiseEvent_flag);
    m_createdPulses.push_back(pulse);
  }
  return  GateVPulseProcessor::ProcessPulseList(inputPulseList);
}

void GateNoise::ProcessOnePulse(const GatePulse* inputPulse,GatePulseList& outputPulseList)
{
  if (!m_createdPulses.empty()) {
    GatePulseList::iterator it = m_createdPulses.begin();
    while (it != m_createdPulses.end() && (*it)->GetTime()<=inputPulse->GetTime()){
      outputPulseList.push_back((*it));
      ++it;
    }
    m_createdPulses.erase(m_createdPulses.begin(),it);
  }
  outputPulseList.push_back(new GatePulse(*inputPulse));
}

void GateNoise::DescribeMyself(size_t indent)
{
  G4cout << GateTools::Indent(indent) << "Noise processor "<< Gateendl;
}
