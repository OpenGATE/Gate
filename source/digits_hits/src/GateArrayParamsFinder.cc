/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateArrayParamsFinder.hh"

#include "GateVVolume.hh"
#include "GateVGlobalPlacement.hh"
#include "GateArrayRepeater.hh"
#include "GateObjectRepeaterList.hh"


GateArrayParamsFinder::GateArrayParamsFinder(GateVVolume* anInserter, size_t& nbX, size_t& nbY, size_t& nbZ)
{
  GateArrayRepeater* anArray = GetArrayRepeater(GetRepeater(anInserter));

  nbX = anArray->GetRepeatNumberX();
  nbY = anArray->GetRepeatNumberY();
  nbZ = anArray->GetRepeatNumberZ();
  m_nbX = nbX;
  m_nbY = nbY;
  m_nbZ = nbZ;
}

GateVGlobalPlacement* GateArrayParamsFinder::GetRepeater(GateVVolume* anInserter)
{
  GateVVolume *autoCrIn = dynamic_cast<GateVVolume*>(anInserter);
  if(!autoCrIn) {
    G4cerr << 	G4endl << "[GateArrayParamsFinder::GetRepeater]:" << G4endl
	   <<   "Sorry, but your Inserter isn't a GateVVolume" << G4endl;
    return 0;
  }
  if (!autoCrIn->GetRepeaterList()) {
    G4cerr << 	G4endl << "[GateArrayParamsFinder::GetRepeaterFindArrayParams]:" << G4endl
	   <<   "Sorry, but you don't have a repeater list" << G4endl;
    return 0;
  }
  if (autoCrIn->GetRepeaterList()->size()== 1)
    return autoCrIn->GetRepeaterList()->GetRepeater(0);
  else {
    G4cerr << 	G4endl << "[GateArrayParamsFinder::GetRepeater]:" << G4endl
	   <<   "Sorry, but you have several repeaters" << G4endl;
    return 0;
  }
}

GateArrayRepeater* GateArrayParamsFinder::GetArrayRepeater(GateVGlobalPlacement* aRepeater)
{
  GateArrayRepeater* anArray = dynamic_cast<GateArrayRepeater*>(aRepeater);
  if (!anArray) {
    G4cerr <<   G4endl << "[GateArrayParamsFinder::GetArrayRepeater]:" << G4endl
	   <<   "Sorry, but you haven't an array repeater" << G4endl;
    return 0;
  }
  return anArray;
}

void GateArrayParamsFinder::FindInputPulseParams(const size_t copyNb, size_t& i, size_t& j, size_t& k)
{
  i = copyNb % (m_nbX * m_nbY) % m_nbX;
  j = copyNb % (m_nbX * m_nbY) / m_nbX;
  k = copyNb / (m_nbX * m_nbY);
//   G4cout << "Number: " << copyNb
// 	 << "\tm_nbX: " << m_nbX << "\tm_nbY: " << m_nbY << "\tm_nbZ: " << m_nbZ
// 	 << "\ti: " << i << "\tj: " << j << "\tk: " << k  << G4endl;
}
