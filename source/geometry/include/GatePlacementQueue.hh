/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
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

// Do we need an iterable queue for fast traveling?? Solution is easy and retains queue purposes
//Example class definition for iterable queue
/*#include <queue>
#include <deque>
#include <iostream>

template<typename T, typename Container=std::deque<T> >
class iterable_queue : public std::queue<T,Container>
{
public:
    typedef typename Container::iterator iterator;
    typedef typename Container::const_iterator const_iterator;

    iterator begin() { return this->c.begin(); }
    iterator end() { return this->c.end(); }
    const_iterator begin() const { return this->c.begin(); }
    const_iterator end() const { return this->c.end(); }
};*/

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

