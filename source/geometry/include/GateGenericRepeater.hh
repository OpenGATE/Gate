/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/


#ifndef GATEGENERICREPEATER_H
#define GATEGENERICREPEATER_H 1

#include "globals.hh"
#include "GateVGlobalPlacement.hh"
#include "GatePlacementQueue.hh"

class GateGenericRepeaterMessenger;

//-------------------------------------------------------------------------------------------------
/*! 
  \class  GateGenericRepeater
  \brief  The GateGenericRepeater create a repetition of an object 
  \brief  according to a given list of translation    
*/      
class GateGenericRepeater : public GateVGlobalPlacement
{
public:
  GateGenericRepeater(GateVVolume* itsObjectInserter,
                      const G4String& itsName="genericRepeater");
  virtual ~GateGenericRepeater();
  virtual void PushMyPlacements(const G4RotationMatrix& currentRotationMatrix,
                                const G4ThreeVector& currentPosition,
                                G4double aTime);
  virtual void DescribeMyself(size_t indent);
  void SetPlacementsFilename(std::string filename);
  void EnableRelativeTranslation(bool b) { mUseRelativeTranslation = b; }
  void SetPlacementList(std::vector<GatePlacement> l);
  G4int GetRepeatNumber() { return (G4int)mPlacementsList.size(); };

protected:
  GateGenericRepeaterMessenger* mMessenger; 
  std::vector<GatePlacement> mPlacementsList;
  bool mUseRotation;
  bool mUseTranslation;
  bool mUseRelativeTranslation;
};
//-------------------------------------------------------------------------------------------------

#endif

