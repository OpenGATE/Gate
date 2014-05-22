/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


#include "GateGenericRepeater.hh"
#include "GateGenericRepeaterMessenger.hh"

#include "G4ThreeVector.hh"
#include "G4Transform3D.hh"
#include "G4RotationMatrix.hh"
#include "G4UnitsTable.hh"
#include "GateTools.hh"
#include "GateMiscFunctions.hh"

//--------------------------------------------------------------------------------------------
GateGenericRepeater::GateGenericRepeater(GateVVolume* itsObjectInserter,
                                         const G4String& itsName)
  : GateVGlobalPlacement(itsObjectInserter, itsName), mMessenger(0) {
  mPlacementsList.clear();
  mUseRotation = mUseTranslation = false;
  mUseRelativeTranslation = true;
  mMessenger = new GateGenericRepeaterMessenger(this);
}
//--------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------
GateGenericRepeater::~GateGenericRepeater() {  
  delete mMessenger;
}
//--------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------
void GateGenericRepeater::SetPlacementList(std::vector<GatePlacement> list) {
  mPlacementsList.clear();
  mPlacementsList.resize(list.size());
  std::copy(list.begin(), list.end(), mPlacementsList.begin());
  mUseTranslation = mUseRotation = true;
}
//--------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------
// void GateGenericRepeater::ReadPlacementsFromFile(std::string filename, 
//                                                  std::vector<GatePlacement> & placementsList// , 
// //                                                  bool & mUseTranslation,
// //                                                  bool & mUseRotation
//                                                  ) {
//   std::vector<double> bidon;
//   ReadTimePlacements(filename, bidon, mPlacementsList);
//   /*
//   std::vector<double> bidon;
//   ReadPlacementsFromFile(filename, placementsList, bidon, false, mUseTranslation, mUseRotation);
//   */
// }
//--------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------
// void GateGenericRepeater::ReadPlacementsFromFile(std::string filename, 
//                                                  std::vector<GatePlacement> & placementsList, 
//                                                  std::vector<double> & timesList, 
//                                                  bool readTimeList, 
//                                                  bool & mUseTranslation,
//                                                  bool & mUseRotation) {
//   // Reset list
//   placementsList.clear();
//   // Open file
//   std::ifstream is;
//   OpenFileInput(filename, is);
//   skipComment(is);
//   // Read if rotation and/or translation must be used
//   mUseRotation = ReadBool(is, "UseRotation", filename);
//   mUseTranslation = ReadBool(is, "UseTranslation", filename);
//   // Read time / angle units
//   double timeUnit=0, angleUnit=0, translationUnit=0;
//   if (readTimeList) timeUnit = ReadUnit(is, "TimeUnit", filename);
//   if (mUseRotation) angleUnit = ReadUnit(is, "AngleUnit", filename);
//   if (mUseTranslation) translationUnit = ReadUnit(is, "TranslationUnit", filename);
//   // Read values
//   std::string s;
//   while (is) {
//     double time=0;
//     if (readTimeList) { // Read time
//       time = ReadDouble(is)*timeUnit;
//     }
//     // Read placement
//     GatePlacement p;
//     ReadPlacement(is, p, mUseRotation, mUseTranslation, angleUnit, translationUnit);
//     // Insert
//     if (readTimeList) timesList.push_back(time);
//     placementsList.push_back(p);
//   } 
//   is.close();
// }           

//--------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------
// void GateGenericRepeater::ReadPlacement(std::istream & is, 
//                                         GatePlacement & p, 
//                                         bool mUseRotation, bool mUseTranslation, 
//                                         double angleUnit, double translationUnit) {
//   double angle=0, x=0,y=0,z=0, tx=0,ty=0,tz=0;
//   std::string s;
//   // skipComment(is);
//   if (mUseRotation) {
//     // Read angle
//     angle = ReadDouble(is);
//     // Read axis
//     x = ReadDouble(is);
//     y = ReadDouble(is);
//     z = ReadDouble(is);
//   }
//   // Read translation
//   if (mUseTranslation) {
//     tx = ReadDouble(is)*translationUnit;
//     ty = ReadDouble(is)*translationUnit;
//     tz = ReadDouble(is)*translationUnit;
//   }
//   // Insert
//   if (mUseRotation) {
//     G4RotationMatrix r;
//     r.rotate(angle*angleUnit, G4ThreeVector(x,y,z));
//     p.first = r;
//   }
//   else {
//     G4RotationMatrix r;
//     r.rotate(0, G4ThreeVector(0,0,0));
//     p.first = r;
//   }
//   if (mUseTranslation) 
//     p.second = G4ThreeVector(tx,ty,tz);
//   else 
//     p.second = G4ThreeVector(0,0,0);
//   GateMessage("Geometry", 8, "I read placement " << tx << " " << ty << " " << tz 
//               << " \t rot=" << angle << " \t axis=" << x << " " << y << " " << z << G4endl);
// }
//--------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------
void GateGenericRepeater::SetPlacementsFilename(std::string filename) {
  std::vector<double> bidon;
  ReadTimePlacements(filename, bidon, mPlacementsList, mUseTranslation, mUseRotation);
}
//--------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------
void GateGenericRepeater::PushMyPlacements(const G4RotationMatrix& currentRotationMatrix,
                                           const G4ThreeVector& currentPosition,
                                           G4double ) {
  // Check
  if (mPlacementsList.size() ==0) {
    GateError("Please provide a placement file with 'setPlacementsFilename'\n.");
  }
  
  GateDebugMessage("Repeater", 3, "GateGenericRepeater " << GetObjectName() << G4endl);
  GateDebugMessage("Repeater", 3, "\t current position " << currentPosition << G4endl);
  GateDebugMessage("Repeater", 3, "\t current rotation " << currentRotationMatrix << G4endl);
  
  for(unsigned int i=0; i<mPlacementsList.size(); i++) {
    GateDebugMessage("Repeater", 3, "\t translation " << i << " = " << mPlacementsList[i].second << G4endl);
    GateDebugMessage("Repeater", 3, "\t final " << i << " = " << currentPosition+mPlacementsList[i].second << G4endl);
    
    // New position
    G4ThreeVector newPosition;
    if (mUseTranslation) {
      if (mUseRelativeTranslation) {
        newPosition = currentPosition + mPlacementsList[i].second;
      }
      else newPosition = mPlacementsList[i].second;
     }
    else newPosition = currentPosition;
    
    // New rotation
    G4RotationMatrix newRotationMatrix;
    if (mUseRotation) {
      newRotationMatrix = mPlacementsList[i].first;
     }
    else newRotationMatrix = currentRotationMatrix;

    // Set placement
    PushBackPlacement(newRotationMatrix, newPosition);
  }
}
//--------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------
void GateGenericRepeater::DescribeMyself(size_t indent)
{
  G4cout << GateTools::Indent(indent) << "Repetition type: " << "genericRepeater \n";
  G4cout << GateTools::Indent(indent) << "Nb of copies   : " << mPlacementsList.size() << "\n";
}
//--------------------------------------------------------------------------------------------
