/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateBuffer.hh"
#include "G4UnitsTable.hh"

#include "GateApplicationMgr.hh"
#include "GateBufferMessenger.hh"
#include "GateTools.hh"
#include "GateSystemListManager.hh"
#include "GateVSystem.hh"
#include <math.h>

GateBuffer::GateBuffer(GatePulseProcessorChain* itsChain,
      	      	      	      	 const G4String& itsName)
  : GateVPulseProcessor(itsChain,itsName)
  , m_bufferSize(1)
  , m_bufferPos(1)
  , m_oldClock(0)
  , m_readFrequency(1)
  , m_doModifyTime(false)
  , m_mode(0)
{
  m_messenger = new GateBufferMessenger(this);
  SetDepth(0);
}

GateBuffer::~GateBuffer()
{
  delete m_messenger;
}
void GateBuffer::SetDepth(size_t depth)
{
    GateVSystem* system = GateSystemListManager::GetInstance()->GetSystem(0);
    m_enableList.resize(system->GetTreeDepth());
    if (depth>system->GetTreeDepth()-1) depth=system->GetTreeDepth()-1;
    for (size_t i=0;i<=depth;i++) m_enableList[i]=true;
    for (size_t i=depth+1;i<system->GetTreeDepth();i++) m_enableList[i]=false;
    size_t nofElements = system->ComputeNofSubCrystalsAtLevel(0,m_enableList);
    m_bufferPos.resize(nofElements);
    for (size_t i=0;i<nofElements;i++) m_bufferPos[i]=0;
}
void GateBuffer::ProcessOnePulse(const GatePulse* inputPulse,GatePulseList& outputPulseList)
{
   GateVSystem* system = GateSystemListManager::GetInstance()->GetSystem(0);
   buffer_t clock = (buffer_t)( (inputPulse->GetTime()-GateApplicationMgr::GetInstance()->GetTimeStart())* m_readFrequency);
   buffer_t deltaClocks = (m_oldClock<clock)? clock - m_oldClock : 0;
   size_t iBuf = system->ComputeIdFromVolID(inputPulse->GetOutputVolumeID(),m_enableList);
//   m_bufferPos[iBuf] = m_bufferPos[iBuf]>deltaClocks ? m_bufferPos[iBuf]-deltaClocks : 0;
   switch (m_mode){
      case 0 : m_bufferPos[iBuf] = m_bufferPos[iBuf]>deltaClocks ? m_bufferPos[iBuf]-deltaClocks : 0; break;
      case 1 : if (deltaClocks>0) m_bufferPos[iBuf]=0;break;
   }
//   G4cout<<"Using buffer "<<iBuf<<" for level1="<<inputPulse->GetComponentID(1)<<G4endl;
   if (m_bufferPos[iBuf]+1<=m_bufferSize) {
      GatePulse* pls = new GatePulse(*inputPulse);
      if (m_doModifyTime) {
        G4double tme = GateApplicationMgr::GetInstance()->GetTimeStart()+clock/m_readFrequency;
	if (m_mode==1) tme += 1./m_readFrequency;
      	pls->SetTime(tme);
      }
      outputPulseList.push_back(pls);
      m_bufferPos[iBuf]++;
   }
   m_oldClock = clock;
}

void GateBuffer::DescribeMyself(size_t indent)
{
  G4cout << GateTools::Indent(indent) << "Buffer: " << G4BestUnit(m_bufferSize,"Memory size")
         << "Read @ "<< G4BestUnit(m_readFrequency,"Frequency")<<G4endl;
}
