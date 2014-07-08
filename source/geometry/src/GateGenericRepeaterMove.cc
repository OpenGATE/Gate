/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


#include "GateGenericRepeaterMove.hh"
#include "GateGenericRepeaterMoveMessenger.hh"
#include "G4ThreeVector.hh"
#include "G4Transform3D.hh"
#include "G4RotationMatrix.hh"
#include "G4UnitsTable.hh"
#include "GateTools.hh"
#include "GateVVolume.hh"
#include "GateObjectRepeaterList.hh"
#include "GateMiscFunctions.hh"

//-------------------------------------------------------------------------------------------------
GateGenericRepeaterMove::GateGenericRepeaterMove(GateVVolume* itsObjectInserter,
                                                 const G4String& itsName)
  : GateVGlobalPlacement(itsObjectInserter,itsName)
{
  mMessenger = new GateGenericRepeaterMoveMessenger(this);
  
  // Create internal random repeater
  mGenericRepeater = new GateGenericRepeater(itsObjectInserter, itsName+"GenericRepeater");
  itsObjectInserter->GetRepeaterList()->AppendObjectRepeater(mGenericRepeater);  
  mGenericRepeater->EnableRelativeTranslation(true);
}
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
GateGenericRepeaterMove::~GateGenericRepeaterMove()
{  
  delete mMessenger;
}
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
void GateGenericRepeaterMove::SetFilename(G4String filename) {
  // Reset list
  mListOfPlacementsList.clear();
  ReadTimePlacementsRepeat(filename, mTimeList, mListOfPlacementsList);
  // mGenericRepeater->SetPlacementList(list);
}
//-------------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------------
// void GateGenericRepeaterMove::InsertLineOfTimePlacement(std::string & line, 
//                                                         double timeUnit, 
//                                                         double translationUnit,
//                                                         double angleUnit) {  
//   std::istringstream is(line);
//   // Read time
//   double time = ReadDouble(is);
//   // Read X Y Z
//   std::vector<GatePlacement> l;
//   while (is) {
//     // Read placement
//     GatePlacement p;
//     GateGenericRepeater::ReadPlacement(is, p, mUseRotation, mUseTranslation, angleUnit, translationUnit);
//     if (is) l.push_back(p);
//   }

//   if (l.size() > 0) {
//     // Insert new translation
//     if (mListOfPlacementsList.size() > 0) {
//       if (mListOfPlacementsList[0].size() != l.size()) {
//         GateError("Error, the first line indicates " << mListOfPlacementsList[0].size()
//                   << " placements, while the line has " << l.size() << " placements.");
//       }
//     }
//     // DD(time); DD(l.size());
//     mListOfPlacementsList.push_back(l);
//     mTimeList.push_back(time*timeUnit);
//   }
// }
//-------------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------------
void GateGenericRepeaterMove::PushMyPlacements(const G4RotationMatrix& currentRotationMatrix,
                                               const G4ThreeVector& currentPosition,
                                               G4double aTime)
{
  // Check
  if (mTimeList.size() ==0) {
    GateError("Please provide a time-placement-repeat file with 'setPlacementsFilename'\n.");
  }
  
  //  GateMessage("Core", 0, "GateGenericRepeaterMove::PushMyPlacements " << GetObjectName() << " t=" << aTime/s << " sec." << G4endl);
  std::vector<GatePlacement> list;
  GetPlacementListFromTime(aTime, list);
  mGenericRepeater->SetPlacementList(list);
  // Initial position
  PushBackPlacement(GatePlacement(currentRotationMatrix,currentPosition));
}
//-------------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------------
void GateGenericRepeaterMove::GetPlacementListFromTime(double aTime, 
                                                       std::vector<GatePlacement> & list) {
  // Search for current "time"
  int i=0; 
  while ((i < (int)mTimeList.size()) && (aTime >= mTimeList[i])) {
    i++;
  }
  i--;
  if ((i < 0) && (aTime < mTimeList[0])) {
    GateError("The time list for " << GetObjectName() << " begin with " << mTimeList[0]/s
              << " sec, so I cannot find the time" << aTime/s << " sec." << G4endl);
  }
  list = mListOfPlacementsList[i];
}
//-------------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------------
void GateGenericRepeaterMove::DescribeMyself(size_t indent)
{
  G4cout << GateTools::Indent(indent) << "Move type:          " << "randomTranslationRepeater"   << "\n";
}
//-------------------------------------------------------------------------------------------------
