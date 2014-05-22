/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


/*
  \brief Miscellaneous functions
  \brief By david.sarrut@creatis.insa-lyon.fr

 */
#ifndef GATEMISCFUNCTIONS_HH
#define GATEMISCFUNCTIONS_HH

#include "globals.hh"
#include <iostream>
#include <fstream>
#include <string>

#include "G4UIcommand.hh"
#include "G4VSolid.hh"
#include "G4Element.hh"
#include "G4UnitsTable.hh"
#include "GateMessageManager.hh"
#include "GatePlacementQueue.hh"

//-----------------------------------------------------------------------------
double rad2deg(const double anglerad);
double deg2rad(const double angledeg);

//-----------------------------------------------------------------------------
void skipComment(std::istream & is);

//-----------------------------------------------------------------------------
std::string getExtension(const std::string & filename);

//-----------------------------------------------------------------------------
std::string removeExtension(const std::string & filename);

//-----------------------------------------------------------------------------
void setExtension(std::string & filename, const std::string& extension);

//-----------------------------------------------------------------------------
void OpenFileInput(G4String filename, std::ifstream & is);

//-----------------------------------------------------------------------------
void OpenFileOutput(G4String filename, std::ofstream & os);

//-----------------------------------------------------------------------------
G4String DoubletoString(G4double a);

//-----------------------------------------------------------------------------
void Get2StringsFromCommand(const G4UIcommand * command, const G4String & newValues, G4String & string1, G4String& string2);

//-----------------------------------------------------------------------------
void GetStringAndValueFromCommand(const G4UIcommand * command, const G4String & newValues, G4String & stringName, double & value);

//-----------------------------------------------------------------------------
void GetStringAndValueWithUnitFromCommand(const G4UIcommand * command, const G4String & newValues, G4String & stringName, double & value);

//-----------------------------------------------------------------------------
void GetStringAnd3ValuesFromCommand(const G4UIcommand * command, const G4String & newValues, G4String & stringName, G4ThreeVector& value);

//-----------------------------------------------------------------------------
G4double norm(const G4ThreeVector & v);
void normalize(G4ThreeVector & v);

//-----------------------------------------------------------------------------
template<typename T>
bool ConvertFromString(const std::string & Str, T & Dest);

//-----------------------------------------------------------------------------
struct GateSolidExtend{
  G4ThreeVector pMin;
  G4ThreeVector pMax;
};

//-----------------------------------------------------------------------------
GateSolidExtend GetSolidExtend(G4VSolid * solid);

//-----------------------------------------------------------------------------
void MoveExtend(const G4ThreeVector & position, GateSolidExtend & ex);

//-----------------------------------------------------------------------------
G4ThreeVector GetExtendHalfBoundingBox(GateSolidExtend & ex);

//-----------------------------------------------------------------------------
GateSolidExtend GetMaxExtend(GateSolidExtend & ex1, GateSolidExtend & ex2);

//-----------------------------------------------------------------------------
G4ThreeVector ComputeBoundingBox(G4VSolid * solid);

//-----------------------------------------------------------------------------
void waitSomeSeconds(int seconds);

//-----------------------------------------------------------------------------
G4double LinearInterpolation(G4double x, std::vector<G4double> & X, std::vector<G4double> & Y);

//-----------------------------------------------------------------------------
G4Element* GetElementBySymbol(G4String elementSymbol, bool warning);

//-----------------------------------------------------------------------------
double ReadUnit(std::istream & is, std::string unit, std::string filename);

//-----------------------------------------------------------------------------
bool ReadBool(std::istream & is, std::string word, std::string filename);

//-----------------------------------------------------------------------------
double ReadDouble(std::istream & is);

//-----------------------------------------------------------------------------
bool ReadColNameAndUnit(std::istream & is, std::string name, double & unit);

//-----------------------------------------------------------------------------
int ReadColNameAndInteger(std::istream & is, std::string name);

//-----------------------------------------------------------------------------
GatePlacement & ReadPlacement(std::istream & is,
                              bool mUseTranslation, bool mUseRotation, 
                              double translationUnit, double angleUnit);

//-----------------------------------------------------------------------------
void ReadTimePlacements(std::string filename, 
                        std::vector<double> & timesList,
                        std::vector<GatePlacement> & placementsList, 
                        bool & mUseRotation, bool & mUseTranslation);

//-----------------------------------------------------------------------------
void ReadTimePlacementsRepeat(std::string filename, 
                              std::vector<double> & timesList,
                              std::vector<std::vector<GatePlacement> > & listOfPlacementsList);

//-----------------------------------------------------------------------------
void ReadTimeDoubleValue(std::string filename, std::string name, 
                         std::vector<double> & timesList,
                         std::vector<double> & nameList);

//-----------------------------------------------------------------------------
int GetIndexFromTime(std::vector<double> & mTimeList, double aTime);

//-----------------------------------------------------------------------------
G4String GetSaveCurrentFilename(G4String & mSaveFilename);


#include "GateMiscFunctions.icc"

#endif // GATEMISCFUNCTIONS_HH

