/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

#ifndef GATEMISCFUNCTIONS_CC
#define GATEMISCFUNCTIONS_CC

#include "GateMiscFunctions.hh"
#include "GateActorManager.hh"

#include "G4SystemOfUnits.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3Vector.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4VoxelLimits.hh"
#include "G4AffineTransform.hh"
#include "G4VSolid.hh"
#include "G4ThreeVector.hh"
#include "GateRunManager.hh"
#include "G4Run.hh"

#include <sys/types.h>
#include <sys/file.h>
#include <unistd.h>
#include <fcntl.h>

#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"


//-----------------------------------------------------------------------------
void skipComment(std::istream & is)
{
  char c;
  char line[1024];
  if (is.eof()) return;
  is >> c;
  while (is && (c == '#')) {
    is.getline (line, 1024);
    is >> c;
    if (is.eof()) return;
  }
  is.unget();
} ////
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
std::string getExtension(const std::string & filename) {
  unsigned int position = filename.find_last_of(".");
  //if (position == filename.npos) return std::string("");
  return filename.substr(position+1,filename.size()-position);
} ////
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
std::string removeExtension(const std::string & filename) {
  unsigned int position = filename.find_last_of(".");
  //if (position == filename.npos) return std::string("");
  return filename.substr(0, position);
} ////
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void setExtension(std::string & filename, const std::string& extension) {
  unsigned int position = filename.find_last_of(".");
  //if (position == filename.npos) {
  //  filename += ".";
  //}
  //else {
  filename = filename.substr(0,position+1);
  //}
  filename += extension;

} ////
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void OpenFileInput(G4String filename, std::ifstream & is)
{
  is.open(filename.data());
  if (!is) {
    GateError("Error while opening " << filename
	      << " for reading.");
  }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void OpenFileOutput(G4String filename, std::ofstream & os)
{
  os.open(filename.data());
  if (!os) {
    GateError("Error while opening " << filename
	      << " for writing.");
  }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
G4String DoubletoString(G4double a) {
  std::ostringstream os;
  os << a;
  return G4String(os.str());
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void Get2StringsFromCommand(const G4UIcommand * /*command*/,
			    const G4String & newValues,
			    G4String & string1,
			    G4String& string2)
{
  std::string valueString;
  int i = newValues.find(" ");
  if (i == (int)std::string::npos) {
    //  GateError( "Command '" << command->GetCommandPath()
    //	   << "' expects 2 arguments"
    //	   << " but I read '" << newValues << "'."
    //	   << Gateendl);
  }
  string1 = newValues.substr(0,i);
  //G4cout << "$K2 stringName= |" << stringName << "|\n");
  string2 = newValues.substr(i+1,newValues.length());
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GetStringAndValueFromCommand(const G4UIcommand * /*command*/,
				  const G4String & newValues,
				  G4String & stringName,
				  double & value)
{
  std::string valueString;
  int i = newValues.find(" ");
  if (i == (int)std::string::npos) {
    //  GateError( "Command '" << command->GetCommandPath()
    //	   << "' expects 2 arguments (a 'string' and a 'value+unit')"
    //	   << " but I read '" << newValues << "'."
    //	   << Gateendl);
  }
  stringName = newValues.substr(0,i);
  //G4cout << "$K2 stringName= |" << stringName << "|\n");
  valueString = newValues.substr(i+1,newValues.length());
  //G4cout << "$K2 val =  " << valueString << Gateendl);
  value = G4UIcmdWithADoubleAndUnit::GetNewDoubleValue(valueString.c_str());
  if (value == 0) {
    //  GateError( "Could not convert the second arg ('"
    //	       << valueString << "') of the command '"
    //	       << command->GetCommandPath()
    //	       << "' to a 'double' value. Try something like '2.0' ..."
    //	       << Gateendl);
  }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GetStringAndValueWithUnitFromCommand(const G4UIcommand * /*command*/,
					  const G4String & newValues,
					  G4String & stringName,
					  double & value)
{
  std::string valueString;
  int i = newValues.find(" ");
  if (i == (int)std::string::npos) {
    //  GateError( "Command '" << command->GetCommandPath()
    //	       << "' expects 2 arguments (a 'string' and a 'value+unit')"
    //	       << " but I read '" << newValues << "'."
    //	       << Gateendl);
  }
  stringName = newValues.substr(0,i);
  //G4cout << "$K2 stringName= |" << stringName << "|\n");
  valueString = newValues.substr(i+1,newValues.length());
  //G4cout << "$K2 val =  " << valueString << Gateendl);
  value = G4UIcmdWithADoubleAndUnit::GetNewDoubleValue(valueString.c_str());
  if (value == 0) {
    //  GateError( "Could not convert the second arg ('"
    //	       << valueString << "') of the command '"
    //	       << command->GetCommandPath()
    //	       << "' to a 'double with unit' value. Try something like '2.0 mm' ...\n");
  }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GetStringAnd3ValuesFromCommand(const G4UIcommand * /*command*/,
				    const G4String & newValues,
				    G4String & stringName,
				    G4ThreeVector& value)
{
  std::string valueString;
  int i = newValues.find(" ");
  if (i == (int)std::string::npos) {
    //  GateError( "Command '" << command->GetCommandPath()
    //	       << "' expects 4 arguments (a 'string' and three 'values')"
    //	       << " but I read '" << newValues << "'."
    //	       << Gateendl);
  }
  stringName = newValues.substr(0,i);
  //G4cout << "$K2 stringName= |" << stringName << "|\n");
  valueString = newValues.substr(i+1,newValues.length());
  //G4cout << "$K2 val =  " << valueString << Gateendl);
  value = G4UIcmdWith3Vector::GetNew3VectorValue(valueString.c_str());
  //if (value == 0) {
    //  GateError( "Could not convert the second arg ('"
    //	       << valueString << "') of the command '"
    //	       << command->GetCommandPath()
    //	       << "' to a 'three vector' value. Try something like '1.0 1.0 1.0' ...\n");
  //}
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
G4double norm(const G4ThreeVector & v)
{
  return G4double (v.x()*v.x()+v.y()*v.y()+v.z()*v.z());
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void normalize(G4ThreeVector & v)
{
  G4double n = norm(v);
  v.setX(v.x()/n);
  v.setY(v.y()/n);
  v.setZ(v.z()/n);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
GateSolidExtend GetSolidExtend(G4VSolid * solid)
{
  //GateDebugMessage("Manager",6, "GetSolidExtend of " << solid->GetName() << Gateendl);
  G4double pMin;
  G4double pMax;
  G4VoxelLimits pVoxelLimit;
  G4ThreeVector zero;
  G4AffineTransform pTransform(zero);
  GateSolidExtend ex;

  EAxis pAxis=kXAxis;
  solid->CalculateExtent(pAxis, pVoxelLimit, pTransform, pMin, pMax);
  // GateDebugMessage("Manager",7, "pMinMax x = " << pMin << " " << pMax << Gateendl);
  ex.pMin.setX(pMin);
  ex.pMax.setX(pMax);

  pAxis=kYAxis;
  solid->CalculateExtent(pAxis, pVoxelLimit, pTransform, pMin, pMax);
  // GateDebugMessage("Manager",7, "pMinMax y = " << pMin << " " << pMax << Gateendl);
  ex.pMin.setY(pMin);
  ex.pMax.setY(pMax);

  pAxis=kZAxis;
  solid->CalculateExtent(pAxis, pVoxelLimit, pTransform,  pMin, pMax);
  // GateDebugMessage("Manager",7, "pMinMax z = " << pMin << " " << pMax << Gateendl);
  ex.pMin.setZ(pMin);
  ex.pMax.setZ(pMax);

  return ex;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
G4ThreeVector ComputeBoundingBox(G4VSolid * solid)
{
  //GateDebugMessage("Manager",6, "ComputeBoundingBox of " << solid->GetName() << Gateendl);
  G4double pMin;
  G4double pMax;
  G4VoxelLimits pVoxelLimit;
  G4ThreeVector zero;
  G4AffineTransform pTransform(zero);
  G4ThreeVector halfBoundingBox;

  EAxis pAxis=kXAxis;
  solid->CalculateExtent(pAxis, pVoxelLimit, pTransform, pMin, pMax);
  halfBoundingBox.setX((pMax-pMin)/2.0);

  pAxis=kYAxis;
  solid->CalculateExtent(pAxis, pVoxelLimit, pTransform, pMin, pMax);
  halfBoundingBox.setY((pMax-pMin)/2.0);

  pAxis=kZAxis;
  solid->CalculateExtent(pAxis, pVoxelLimit, pTransform, pMin, pMax);
  halfBoundingBox.setZ((pMax-pMin)/2.0);

  return halfBoundingBox;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
G4ThreeVector GetExtendHalfBoundingBox(GateSolidExtend & ex)
{
  //GateDebugMessage("Manager",6, "GetExtendBoundingBox\n");
  G4ThreeVector v;
  v.setX((ex.pMax.x()-ex.pMin.x())/2.0);
  v.setY((ex.pMax.y()-ex.pMin.y())/2.0);
  v.setZ((ex.pMax.z()-ex.pMin.z())/2.0);
  return v;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void MoveExtend(const G4ThreeVector & position, GateSolidExtend & ex)
{
  // GateDebugMessage("Manager",6, "MoveExtend " << position << Gateendl);
  ex.pMax.setX( ex.pMax.x() - position.x());
  ex.pMin.setX( ex.pMin.x() - position.x());
  ex.pMax.setY( ex.pMax.y() - position.y());
  ex.pMin.setY( ex.pMin.y() - position.y());
  ex.pMax.setZ( ex.pMax.z() - position.z());
  ex.pMin.setZ( ex.pMin.z() - position.z());
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
GateSolidExtend GetMaxExtend(GateSolidExtend & ex1, GateSolidExtend & ex2)
{
  //GateDebugMessage("Manager",6, "GetMaxExtend\n");
  GateSolidExtend ex;

  ex.pMax.setX(std::max(ex1.pMax.x(),ex2.pMax.x()));
  ex.pMin.setX(std::min(ex1.pMin.x(),ex2.pMin.x()));

  ex.pMax.setY(std::max(ex1.pMax.y(),ex2.pMax.y()));
  ex.pMin.setY(std::min(ex1.pMin.y(),ex2.pMin.y()));

  ex.pMax.setZ(std::max(ex1.pMax.z(),ex2.pMax.z()));
  ex.pMin.setZ(std::min(ex1.pMin.z(),ex2.pMin.z()));

  return ex;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void waitSomeSeconds(int seconds)
{
  clock_t endwait;
  endwait = clock () + seconds * CLOCKS_PER_SEC ;
  while (clock() < endwait) {}
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
G4double LinearInterpolation(G4double x, std::vector<G4double> & X, std::vector<G4double> & Y) {
  int i = 0;
  int n = X.size();

  while (i<n && x>X[i]) i++;
  //  ThISMessage("Event", 1, "Lin x=" << x << " i=" << i << ThISendl);
  i--;

  if (i<0) return Y[0]; // first value
  if (i>=n-1) return Y[n-1]; // last value

  //ThISMessage("Event", 1, "X = " << X[i] << " " << X[i+1] << ThISendl);
  //ThISMessage("Event", 1, "Y = " << Y[i] << " " << Y[i+1] << ThISendl);
  G4double value = ((x-X[i])/(X[i+1]-X[i])) * (Y[i+1]-Y[i]) + Y[i];
  //ThISMessage("Event", 1, "value =  " << value << ThISendl);
  return value;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
G4Element* GetElementBySymbol(G4String elementSymbol, bool warning)
{
  const G4ElementTable & theElementTable = *G4Element::GetElementTable();

  // search the element by its symbol
  DD(theElementTable.size());
  for (size_t J=0 ; J<theElementTable.size() ; J++)
    {
      DD(theElementTable[J]->GetSymbol());
      if (theElementTable[J]->GetSymbol() == elementSymbol)
        return theElementTable[J];
    }

  // the element does not exist in the table
  if (warning) {
    GateWarning("\n---> warning from GetElementBySymbol. The element: "
		<< elementSymbol << " does not exist in the table. Return NULL pointer."
		<< Gateendl);
  }
  else {
    GateError("\n---> error from G4Element::GetElement. The element: "
	      << elementSymbol << " does not exist in the table. Abort."
	      << Gateendl);
  }
  return 0;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
double ReadUnit(std::istream & is, std::string unit, std::string filename) {
  std::string s;
  is >> s;
  if (s == unit) is >> s;
  else {
    GateError("Error while searching for string '" << unit << "' in the file <" << filename << ">.");
  }
  double v = G4UnitDefinition::GetValueOf(s);
  if (v<=0) {
    GateError("Error : the unit '" << s << "' is not recognized for '" << unit << "'. Please check this line. ");
   }
  return v;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
bool ReadBool(std::istream & is, std::string word, std::string filename) {
  skipComment(is);
  std::string s;
  is >> s;
  if (s == word) is >> s;
  else {
    GateError("Error, I should read '" << word << "' in the file <" << filename
              << ">, but I read '" << s << "'.");
  }
  bool b = ((s!="0") && (s != "false"));
  skipComment(is);
  return b;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
double ReadDouble(std::istream & is) {
  skipComment(is);
  std::string s;
  is >> s;
  if (s.size() == 0) return 0.0;
  char *endptr;
  const char *nptr = s.c_str();
  double n = strtod(nptr, &endptr);
  if ((n==0) && (nptr == endptr)) {
    GateError("Error I should read a double and I read '" << nptr << "'\n");
  }
  skipComment(is);
  return n;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
bool ReadColNameAndUnit(std::istream & is, std::string name, double & unit) {
  skipComment(is);
  // Read name
  std::string s;
  is >> s;
  if (s != name) {
    // DD(s);
    for(unsigned int i=0; i<s.size(); i++) is.unget();
    return false;
  }
  // Read unit name and convert
  is >> s;
  unit = G4UnitDefinition::GetValueOf(s);
  if (unit<=0) {
    GateError("Error : the unit '" << s << "' is not recognized. Abort.\n");
   }
  // DD(unit);
  return true;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
int ReadColNameAndInteger(std::istream & is, std::string name) {
  skipComment(is);
  // Read name
  std::string s;
  is >> s;
  if (s != name) {
    GateError("I try to read '" << name << "' but I read '" << s << "'. Abort.\n");
  }
  // Read int
  int n = lrint(ReadDouble(is));
  return n;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GatePlacement & ReadPlacement(std::istream & is,
                              bool mUseTranslation, bool mUseRotation,
                              double translationUnit, double angleUnit) {
  double angle=0, x=0,y=0,z=0, tx=0,ty=0,tz=0;
  GatePlacement * p = new GatePlacement;
  if (mUseRotation) {
    // Read angle
    angle = ReadDouble(is);
    // Read axis
    x = ReadDouble(is);
    y = ReadDouble(is);
    z = ReadDouble(is);
  }
  // Read translation
  if (mUseTranslation) {
    tx = ReadDouble(is)*translationUnit;
    ty = ReadDouble(is)*translationUnit;
    tz = ReadDouble(is)*translationUnit;
  }
  // Insert rotation
  if (mUseRotation) {
    G4RotationMatrix r;
    r.rotate(angle*angleUnit, G4ThreeVector(x,y,z));
    p->first = r;
  }
  else {
    G4RotationMatrix r;
    r.rotate(0, G4ThreeVector(0,0,0));
    p->first = r;
  }
  // Insert translation
  if (mUseTranslation)
    p->second = G4ThreeVector(tx,ty,tz);
  else
    p->second = G4ThreeVector(0,0,0);
  GateMessage("Geometry", 8, "I read placement " << tx << " " << ty << " " << tz
              << " \t rot=" << angle << " \t axis=" << x << " " << y << " " << z << Gateendl);
  return *p;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void ReadTimePlacements(std::string filename,
                        std::vector<double> & timesList,
                        std::vector<GatePlacement> & placementsList,
                        bool & mUseRotation, bool & mUseTranslation) {
  // Open file
  std::ifstream is;
  OpenFileInput(filename, is);
  skipComment(is);

  // Use R et/or T ? Time s  | Translation mm | Rotation deg
  double timeUnit=0;
  if (!ReadColNameAndUnit(is, "Time", timeUnit)) {
    GateError("The file '" << filename << "' need to begin with 'Time'\n");
  }
  double angleUnit=0;
  mUseRotation = ReadColNameAndUnit(is, "Rotation", angleUnit);
  double translationUnit=0;
  mUseTranslation = ReadColNameAndUnit(is, "Translation", translationUnit);

  // Loop line
  skipComment(is);
  while (is) {
    // Read time, translation and rotation
    timesList.push_back(ReadDouble(is)*timeUnit);
    placementsList.push_back(ReadPlacement(is, mUseTranslation, mUseRotation, translationUnit, angleUnit));
    skipComment(is);
  }

  // End
  is.close();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void ReadTimePlacementsRepeat(std::string filename,
                              std::vector<double> & timesList,
                              std::vector<std::vector<GatePlacement> > & listOfPlacementsList) {
  // Open file
  std::ifstream is;
  OpenFileInput(filename, is);
  skipComment(is);

  // Use R et/or T ? Time s  | Translation mm | Rotation deg
  double timeUnit=0.;
  if (!ReadColNameAndUnit(is, "Time", timeUnit)) {
    GateError("The file '" << filename << "' need to begin with 'Time'\n");
  }
  int nbOfPlacements = ReadColNameAndInteger(is, "NumberOfPlacements");
  bool mUseRotation; double angleUnit=0;
  mUseRotation = ReadColNameAndUnit(is, "Rotation", angleUnit);
  bool mUseTranslation; double translationUnit=0;
  mUseTranslation = ReadColNameAndUnit(is, "Translation", translationUnit);

  // Loop line
  skipComment(is);
  while (is) {
    // Read time, translation and rotation
    timesList.push_back(ReadDouble(is)*timeUnit);
    std::vector<GatePlacement> * l = new std::vector<GatePlacement>;
    for(int i=0; i<nbOfPlacements; i++) {
      l->push_back(ReadPlacement(is, mUseTranslation, mUseRotation, translationUnit, angleUnit));
    }
    listOfPlacementsList.push_back(*l);
    skipComment(is);
  }

  // End
  is.close();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void ReadTimeDoubleValue(std::string filename, std::string name,
                         std::vector<double> & timesList,
                         std::vector<double> & nameList) {
  // Open file
  std::ifstream is;
  OpenFileInput(filename, is);
  skipComment(is);

  // Use Time and Name
  double timeUnit=0.;
  if (!ReadColNameAndUnit(is, "Time", timeUnit)) {
    GateError("The file '" << filename << "' need to begin with 'Time'\n");
  }
  double nameUnit=0;
  if (!ReadColNameAndUnit(is, name, nameUnit)) {
    GateError("The file '" << filename << "' need to continue with '" << name << "'\n");
  }

  // Loop line
  skipComment(is);
  while (is) {
    // Read time
    timesList.push_back(ReadDouble(is)*timeUnit);
    // Read name
    nameList.push_back(ReadDouble(is)*nameUnit);
    skipComment(is);
  }

  // End
  is.close();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
int GetIndexFromTime(std::vector<double> & mTimeList, double aTime) {
  // Search for current "time"
  int i=0;
  while ((i < (int)mTimeList.size()) && (aTime >= mTimeList[i])) {
    i++;
  }

  // Return the preceding (last) time update before aTime
  if(i > 0) {
    i--;
  }

  if ((i < 0) && (aTime < mTimeList[0])) {
    GateError("The time list for  begin with " << mTimeList[0]/s
              << " sec, so I cannot find the time" << aTime/s << " sec.\n");
  }
  return i;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
G4String GetSaveCurrentFilename(G4String & mSaveFilename) {
  int nr=0;
  // int ne=0;
  const G4Run * run = GateRunManager::GetRunManager()->GetCurrentRun();
  if (run) nr = run->GetRunID();
  else {
    nr = 0;
  }
  // ne = GateActorManager::GetInstance()->GetCurrentEventId();

  G4String extension = "."+getExtension(mSaveFilename);

  /*std::ostringstream oss;
  oss << "_R" << std::setfill('0') << std::setw(4) << nr;
  oss << "_E" << std::setfill('0') << std::setw(15) << ne;
  */
  std::ostringstream oss;
  oss << "_" << nr;
  G4String mSaveCurrentFilename = G4String(removeExtension(mSaveFilename))+oss.str()+extension;
  return mSaveCurrentFilename;
}
//------------------------------------------------------------------------------------------------------

//------------------------------------------------------------------------------------------------------
std::string ReadNextContentLine( std::istream& input, int& lineno, const std::string& fname ) {
  while ( input ){
    std::string line;
    std::getline(input,line);
    ++lineno;
    if (line.empty()) continue;
    if (line[0]=='#') continue;
    return line;
  }
  throw std::runtime_error(std::string("reached end of file '")+fname+std::string("' unexpectedly."));
}
//-----------------------------------------------------------------------------


// ----------------------------------------------------------------------------
//http://stackoverflow.com/questions/216823/whats-the-best-way-to-trim-stdstring
// trim from start
std::string & ltrim(std::string &s)
{
  //s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(std::function<int(int)>(isspace))));
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
  return s;
}

// trim from end
std::string & rtrim(std::string &s)
{
  //s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::function<int(int)>(isspace))).base(), s.end());
  s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), s.end());
  return s;
}

// trim from both ends
std::string & trim(std::string &s)
{
  return ltrim(rtrim(s));
}
// ----------------------------------------------------------------------------


// ----------------------------------------------------------------------------
void GetWords(std::vector<std::string> & words, const std::string & phrase) {
  std::istringstream iss(phrase); // consider all words in 'phrase'
  do {
    std::string s;
    iss >> s;
    s = trim(s);
    if (s != "") words.push_back(s);
  } while (iss);
}
// ---------------------------------------------------------------------------

#endif // GATEMISCFUNCTIONS_CC
