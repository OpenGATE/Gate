/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#include "GateGenericMove.hh"
#include "GateGenericMoveMessenger.hh"
#include "G4ThreeVector.hh"
#include "G4Transform3D.hh"
#include "G4RotationMatrix.hh"
#include "G4UnitsTable.hh"
#include "GateTools.hh"
#include "GateMiscFunctions.hh"
#include "GateVVolume.hh"
#include "GateGenericRepeater.hh"

//-------------------------------------------------------------------------------------------------
GateGenericMove::GateGenericMove(GateVVolume* itsObjectInserter, const G4String& itsName)
  : GateVGlobalPlacement(itsObjectInserter,itsName), mMessenger(0)
{
  mPlacementsList.clear();
  mUseRotation = mUseTranslation = false;
  // mUseRelativeTranslation = true;
  mMessenger = new GateGenericMoveMessenger(this);
}
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
GateGenericMove::~GateGenericMove()
{  
  delete mMessenger;
}
//-------------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------------
void GateGenericMove::SetFilename(G4String filename) {
  ReadTimePlacements(filename, mTimeList, mPlacementsList, mUseRotation, mUseTranslation);
}
//-------------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------------
void GateGenericMove::PushMyPlacements(const G4RotationMatrix& currentRotationMatrix,
                                       const G4ThreeVector& currentPosition,
                                       G4double aTime)
{
  // Check
  if (mTimeList.size() ==0) {
    GateError("Please provide a time-placement file with 'setPlacementsFilename'\n.");
  }

  // Get time index
  int i = GetIndexFromTime(mTimeList, aTime);

  GateDebugMessage("Move", 3, "GateGenericMove " << GetObjectName() << G4endl);
  GateDebugMessage("Move", 3, "\t current time " << aTime/s << " sec." << G4endl);
  GateDebugMessage("Move", 3, "\t current index " << i << G4endl);
  GateDebugMessage("Move", 3, "\t pos " << currentPosition << G4endl);
  GateDebugMessage("Move", 3, "\t plac " << mPlacementsList[i].second << G4endl);
  
  // New placement
  G4RotationMatrix newRotationMatrix;
  G4ThreeVector newPosition;
  if (mUseRotation) newRotationMatrix = mPlacementsList[i].first;
  else newRotationMatrix = currentRotationMatrix;
  if (mUseTranslation) 
    {
      // if (mUseRelativeTranslation) {
      //         newPosition = currentPosition + mPlacementsList[i].second;
      //       }
      //       else 
        newPosition = mPlacementsList[i].second;
    }
  else newPosition = currentPosition;

  // Return placement
  PushBackPlacement(GatePlacement(newRotationMatrix,newPosition));
}
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
void GateGenericMove::DescribeMyself(size_t indent)
{
  G4cout << GateTools::Indent(indent) << "Move type:          " << "genericMove\n";
}
//-------------------------------------------------------------------------------------------------
