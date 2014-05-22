/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


#include "GateVGlobalPlacement.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "GateVVolume.hh"

/* Constructor
   itsObjectInserter: the inserter to which the repeater is appended
   itsName:       	  the name chosen for this repeater
*/    
//----------------------------------------------------------------------------------------------------
GateVGlobalPlacement::GateVGlobalPlacement(GateVVolume* itsObjectInserter,
      	      	      	      	      	   const G4String& itsName)
  :GateClockDependent( itsName), m_objectInserter(itsObjectInserter)
{
}
//----------------------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------------------
// Destructor
GateVGlobalPlacement::~GateVGlobalPlacement() 
{
  // Normally the placement queue should be empty, but we empty it as a security measure
  if (!m_placementQueue.empty()) {
    G4cout << "\n\n!!![GateVGlobalPlacement::~GateVGlobalPlacement]:\n"
      "An object repeater was destroyed with its placement queue not empty!\n"
      "This should not happen. There may be a problem somewhere.\n\n";
    while (m_placementQueue.size()) m_placementQueue.pop();
    //DS I suppress a ';' just after the 'size())' (while did nothing)
  }
}
//----------------------------------------------------------------------------------------------------

//----------------------------------------------------------------------------------------------------
/* Public method to be called to repeat a series of placements placed into an input queue
    
   motherQueue:  the input queue to repeat
      
   If the repeater is enabled, it repeats each placement in the input queue using the user-defined
   virtual method PushMyPlacements. The series thus obtained are all placed into the repeater placement 
   queue, which is returned to the caller. 
   If the repeater is disabled, it directly places a copy of each input placement into the
   output queue, so that the output queue is a direct copy of the input queue.
   In both cases (enabled or disabled), the input queue is emptied as it is processed.
*/
GatePlacementQueue* GateVGlobalPlacement::ComputePlacements(GatePlacementQueue* motherQueue)
{
 
  // Loop until the input-queue is empty
  while (motherQueue->size()) {

    // Extract a placement from the input queue
    GatePlacement placement = motherQueue->pop_front();
    
    // Extract the rotation matrix and translation vector from this placement
    G4RotationMatrix rotationMatrix = placement.first;
    G4ThreeVector position = placement.second ;

    // If the repeater is enabled, process the placement through PushMyPlacements
    // If it is disabled, place a copy of the placement into the output queue
    
    if (IsEnabled()) {
      PushMyPlacements(rotationMatrix,position,GetCurrentTime());}
    else {
      PushBackPlacement(rotationMatrix,position);
    }
  }

  // Return a pointer to the repeater's placement queue (output queue)
  return &m_placementQueue;
}
//----------------------------------------------------------------------------------------------------



//----------------------------------------------------------------------------------------------------
/* Implementation of the pure virtual method defined by the base-class.
   Calls DescribeMyself() to print-out a description of the repeater

   indent: the print-out indentation (cosmetic parameter)
*/
void GateVGlobalPlacement::Describe(size_t indent)
{
  // Call the base-class Describe() method to print-out the parameters
  // that are common to all clock-dependant objects 
  GateClockDependent::Describe(indent);
    
  // Call the pure-virtual method DescribeMyself() to print-out the 
  // parameters that are specific to each repeater
  DescribeMyself(indent);
    
  G4cout << G4endl;
}
//----------------------------------------------------------------------------------------------------


