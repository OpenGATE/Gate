/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


#ifndef GATEGENERICREPEATERMOVE_H
#define GATEGENERICREPEATERMOVE_H 1

#include "globals.hh"
#include "GateVGlobalPlacement.hh"
#include "GatePlacementQueue.hh"
#include "GateGenericRepeater.hh"

class GateGenericRepeaterMoveMessenger;

//-------------------------------------------------------------------------------------------------
class GateGenericRepeaterMove  : public GateVGlobalPlacement
{
public:
  
  GateGenericRepeaterMove(GateVVolume* itsObjectInserter, const G4String& itsName);
  virtual ~GateGenericRepeaterMove();
  virtual void PushMyPlacements(const G4RotationMatrix& currentRotationMatrix,
                                const G4ThreeVector& currentPosition, G4double aTime);
  virtual void DescribeMyself(size_t indent);
  void SetFilename(G4String filename);
  void EnableRelativeTranslation(bool b) { mUseRelativeTranslation = b; mGenericRepeater->EnableRelativeTranslation(b); }
  
public:
  GateGenericRepeater * mGenericRepeater;
  GateGenericRepeaterMoveMessenger* mMessenger;
  std::vector<std::vector<GatePlacement> >  mListOfPlacementsList;
  std::vector<double> mTimeList;
  //void InsertLineOfTimePlacement(std::string & line, double timeUnit, double translationUnit, double angleUnit);
  void GetPlacementListFromTime(double aTime, std::vector<GatePlacement> & list);
  // bool mUseTranslation;
//   bool mUseRotation;
  bool mUseRelativeTranslation;
};
//-------------------------------------------------------------------------------------------------

#endif

