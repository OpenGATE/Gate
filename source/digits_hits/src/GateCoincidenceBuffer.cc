/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateCoincidenceBuffer.hh"
#include "G4UnitsTable.hh"
#include "GateCoincidenceBufferMessenger.hh"
#include "GateTools.hh"
#include "GateVolumeID.hh"
#include "GateOutputVolumeID.hh"
#include "GateVVolume.hh"
#include "GateDetectorConstruction.hh"
#include "GateCrystalSD.hh"
#include "GateVSystem.hh"
#include "GateObjectChildList.hh"
#include "GateMaps.hh"
#include "GateApplicationMgr.hh"
#include <math.h>





GateCoincidenceBuffer::GateCoincidenceBuffer(GateCoincidencePulseProcessorChain* itsChain,
			   const G4String& itsName)
  : GateVCoincidencePulseProcessor(itsChain,itsName)
  , m_bufferSize(1)
  , m_bufferPos(0)
  , m_oldClock(0)
  , m_readFrequency(1)
//  , m_doModifyTime(false)
  , m_mode(0)
{
  m_messenger = new GateCoincidenceBufferMessenger(this);
}




GateCoincidenceBuffer::~GateCoincidenceBuffer()
{
  delete m_messenger;
}




GateCoincidencePulse* GateCoincidenceBuffer::ProcessPulse(GateCoincidencePulse* inputPulse,G4int )
{
   GateCoincidencePulse* ans=0;
   buffer_t clock = (buffer_t)( (inputPulse->GetTime()-GateApplicationMgr::GetInstance()->GetTimeStart())* m_readFrequency);
   buffer_t deltaClocks = (m_oldClock<clock)? clock - m_oldClock : 0;
   switch (m_mode){
      case 0 : m_bufferPos = m_bufferPos>deltaClocks ? m_bufferPos-deltaClocks : 0; break;
      case 1 : if (deltaClocks>0) m_bufferPos=0;break;
   }

   if (m_bufferPos+1<=m_bufferSize) {
      ans = new GateCoincidencePulse(*inputPulse);
//       if (m_doModifyTime) {
//         G4double tme = GateApplicationMgr::GetInstance()->GetTimeStart()+clock/m_readFrequency;
// 	if (m_mode==1) tme += 1./m_readFrequency;
//       	ans->SetTime(tme);
//       }
      m_bufferPos++;
   }
   m_oldClock = clock;
   return ans;
}


void GateCoincidenceBuffer::DescribeMyself(size_t indent)
{
  G4cout << GateTools::Indent(indent) << "Buffer: " << G4BestUnit(m_bufferSize,"Memory size")
         << "Read @ "<< G4BestUnit(m_readFrequency,"Frequency")<<G4endl;
}
