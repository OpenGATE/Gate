/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateLevelsFinder.hh"

#include "GateVVolume.hh"
#include "GateVolumeID.hh"
#include "GateObjectChildList.hh"


// GateLevelsFinder::GateLevelsFinder(GateVVolume* anInserter)
//   : std::vector<size_t>()
// {
//   this.push_back(anInserter->GetVolumeNumber());
//   while (anInserter->GetMotherList())
//     {
//       anInserter = anInserter->GetMotherList()->GetCreator()->GetInserter();
//       this.push_back(anInserter->GetVolumeNumber());
//     }
//   m_size = this.size();
// }

GateLevelsFinder::GateLevelsFinder(GateVVolume* anInserter, std::vector<size_t>& levels)
{
  levels.push_back(anInserter->GetVolumeNumber());
  while (anInserter->GetMotherList())
    {
      anInserter = anInserter->GetMotherList()->GetCreator();
      levels.push_back(anInserter->GetVolumeNumber());
    }
  m_size = levels.size();
}


std::vector<size_t> GateLevelsFinder::FindInputPulseParams(const GateVolumeID* aVolumeID,
							      const size_t depth)
{
  std::vector<size_t> pulseLevels;
  size_t temp;
  for(size_t i = 0; i < m_size; i++)
    {
      temp = (aVolumeID->GetCopyNo(depth - i) != -1)
	? aVolumeID->GetCopyNo(depth - i) : 0;
      pulseLevels.push_back(temp);
    }
  return pulseLevels;
}
