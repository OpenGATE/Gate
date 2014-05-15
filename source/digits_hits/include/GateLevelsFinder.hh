/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateLevelsFinder_h
#define GateLevelsFinder_h 1

#include "globals.hh"
#include <iostream>
#include <vector>

class GateVVolume;
class GateVolumeID;

/*! \class  GateLevelsFinder
    \brief

    - GateLevelsFinder - by Martin.Rey@epfl.ch (sept. 2003)

    - Find the number of copies for an Inserter
      and then go up the chain to find the number of copies of the Inserters above
      It allows also to find the levels params of a pulse
      (Give the copy number of all the volume where is the pulse)

      \sa GateVPulseProcessor
*/
class GateLevelsFinder//  : public std::vector<size_t>
{
public:

  //! Constructs a new GateLevelsFinder; m_nbX, m_nbY and m_nbZ are the parameters of the matrix of detection
  GateLevelsFinder(GateVVolume*, std::vector<size_t>&);
  //! Destructor
  ~GateLevelsFinder() {};

  //! Find the different parameters of the input Pulse :
  //! e.g. the position in this array of the hit
  std::vector<size_t> FindInputPulseParams(const GateVolumeID* aVolumeID, const size_t);

private:
  size_t m_size;
};


#endif
