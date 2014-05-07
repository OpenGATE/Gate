/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GatePlacementQueue_h
#define GatePlacementQueue_h 1

#include "globals.hh"
#include <queue>
#include "GateMessageManager.hh"
#include "G4ThreeVector.hh"
#include "G4RotationMatrix.hh"

typedef std::pair<G4RotationMatrix,G4ThreeVector> GatePlacement;

//-----------------------------------------------------------
/*! \class  GatePlacementQueue
    \brief  Class for storing a series of physical volumes' positions
    
    - GatePlacementQueue - by Daniel.Strul@iphe.unil.ch
    
    - This class stores a series of physical volumes' positions as a queue of
      GatePlacement, where each GatePlacement combines a rotation matrix and a 
      translation vector
*/      
class GatePlacementQueue : public std::queue<GatePlacement>
{
  public:  

  GatePlacementQueue() {}
  virtual ~ GatePlacementQueue() {}
  
  virtual inline void push_back(const GatePlacement& aPlacement)
    { std::queue<GatePlacement>::push(aPlacement) ; }
  virtual inline void push_back(const G4RotationMatrix& rotationMatrix,const G4ThreeVector& position)
    { push_back(GatePlacement(rotationMatrix,position)) ; }
  virtual inline GatePlacement pop_front()
  { GatePlacement placement = front();
    std::queue<GatePlacement>::pop() ; 
    return placement;
  }
};
//-----------------------------------------------------------

#endif

