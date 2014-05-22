/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateArrayParamsFinder_h
#define GateArrayParamsFinder_h 1

#include "globals.hh"
#include <iostream>


class GateVVolume;
class GateVGlobalPlacement;
class GateArrayRepeater;

/*! \class  GateArrayParamsFinder
    \brief

    - GateArrayParamsFinder - by Martin.Rey@epfl.ch

    - The arrayParamsFinder allows to find the different parameters of the array of detection :
    The numbers of rows in x, y and z
    The position in this matrix of the hit

      \sa GateVPulseProcessor
*/
class GateArrayParamsFinder
{
public:

  //! Constructs a new GateArrayParamsFinder; m_nbX, m_nbY and m_nbZ are the parameters of the matrix of detection
  GateArrayParamsFinder(GateVVolume*, size_t&, size_t&, size_t&);

  //! Destructor
  ~GateArrayParamsFinder() {};

  //! Find the different parameters of the input Pulse :
  //! e.g. the position in this array of the hit
  void FindInputPulseParams(const size_t, size_t&, size_t&, size_t&);

private:
  size_t m_nbX, m_nbY, m_nbZ;                                     //!< Parameters of the matrix of detection

  //! Get the VObjectReapeater from an VObjectInserter (if it isn't the right VObjectInserter return 0)
  GateVGlobalPlacement* GetRepeater(GateVVolume*);

  //! Get the ArrayRepeater from an VObjectReapeater (if it isn't an ArrayRepeater return 0)
  GateArrayRepeater* GetArrayRepeater(GateVGlobalPlacement*);
};


#endif
