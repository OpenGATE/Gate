/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


#ifndef GATEGENERICMOVE_H
#define GATEGENERICMOVE_H 1

#include "globals.hh"
#include "GateVGlobalPlacement.hh"
#include "GatePlacementQueue.hh"

class GateGenericMoveMessenger;

//-------------------------------------------------------------------------------------------------
/*! \class  GateGenericMove
    \brief The GateGenericMove models a motion described with
    several rotations.
*/      
class GateGenericMove  : public GateVGlobalPlacement
{
public:
  
  GateGenericMove(GateVVolume* itsObjectInserter, const G4String& itsName);
  virtual ~GateGenericMove();
  virtual void PushMyPlacements(const G4RotationMatrix& currentRotationMatrix,
                                const G4ThreeVector& currentPosition, G4double aTime);
  virtual void DescribeMyself(size_t indent);
  void SetFilename(G4String filename);
  // void EnableRelativeTranslation(bool b) { mUseRelativeTranslation = b; }
  
public:
  GateGenericMoveMessenger* mMessenger;
  std::vector<GatePlacement> mPlacementsList;
  std::vector<double> mTimeList;
  // int GetIndexFromTime(double aTime);
  bool mUseRotation;
  bool mUseTranslation;
};
//-------------------------------------------------------------------------------------------------

#endif

