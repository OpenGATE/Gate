/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateUIcmdWithAVectorAndUnit_H
#define GateUIcmdWithAVectorAndUnit_H 1

#include "G4UIcommand.hh"
#include <vector>
#include "GateTokenizer.hh"
//LF
//#include <strstream>
#include <sstream>
//LF
// class description:
//  A concrete class of G4UIcommand. The command defined by this class
// takes a vector<> and a unit.
//  General information of G4UIcommand is given in G4UIcommand.hh.

template<typename vContentType> class GateUIcmdWithAVectorAndUnit : public G4UIcommand
{
public: // with description
  GateUIcmdWithAVectorAndUnit(const char * theCommandPath, G4UImessenger * theMessenger);
  //  Constructor. The command string with full path directory
  // and the pointer to the messenger must be given.

//    G4int DoIt(G4String parameterList) {  
//      messenger->SetNewValue( this, parameterList );
//      return 0;
//    };

  std::vector<vContentType> GetNewVectorValue(G4String paramString);
  //  Convert string the vector values and a unit to
  //  the Vector
  G4String ConvertToString(std::vector<vContentType> vec);
  //  Convert the Vector and the unit to a string which represents it
};

template<typename vContentType>
GateUIcmdWithAVectorAndUnit<vContentType>::GateUIcmdWithAVectorAndUnit
(const char * theCommandPath, G4UImessenger * theMessenger)
:G4UIcommand(theCommandPath,theMessenger)
{
  //  G4UIparameter * intParam = new G4UIparameter('i');
  G4UIparameter * intParam = new G4UIparameter('s');
  SetParameter(intParam);
}

template<typename vContentType>
std::vector<vContentType> GateUIcmdWithAVectorAndUnit<vContentType>::GetNewVectorValue(G4String paramString)
{
  G4cout << "GateUIcmdWithAVectorAndUnit::GetNewVectorValue : paramString <" << paramString << ">" << G4endl;

  //const char* t = paramString;
  //LF 21/12/2005
  //std::istrstream is(paramString);
  std::istringstream is(paramString);
  //LF
  std::vector<vContentType> vec;
    G4int nElem;
    is >> nElem;
    G4cout << "GateUIcmdWithAVectorAndUnit::GetNewVectorValue : nElem "<< nElem << G4endl;
  vContentType value;
//    G4cout << "GateUIcmdWithAVectorAndUnit::GetNewVectorValue : ";
  G4String aToken;
  GateTokenizer parameterToken( paramString );
  G4bool isGood = true;
  do {
    aToken = parameterToken(); 
    G4cout << " aToken = " << aToken << G4endl;
    
    const char* charToken = aToken; 
    //LF
    //std::istrstream istrToken(charToken);
    std::istringstream istrToken(charToken);
    //LF
    istrToken >> value;
    
    G4cout << " istrToken.good() = "  << istrToken.good() << G4endl;
    
    if (istrToken.good()) {
            G4cout << " is.eof() " << is.eof() << " is.fail() " << is.fail() << " is.bad() " << is.bad() << " ";
      vec.push_back(value);
            G4cout << value << ", ";
    } else {
      isGood = false;
    }
  } while (isGood);
  //  G4cout << G4endl;
  G4cout << "GateUIcmdWithAVectorAndUnit::GetNewVectorValue: vec.size()  " << vec.size() << G4endl;
  if ( aToken.length() == 0 ) {
    G4cout << "GateUIcmdWithAVectorAndUnit::GetNewVectorValue : ERROR : missing Unit" << G4endl;
  } else {
    //    char unts[30];
    const char* charToken = aToken; 
    //LF
    //std::istrstream istrToken(charToken);
    std::istringstream istrToken(charToken);
    //LF
    G4String unt = "";
    istrToken >> unt;
    //    G4cout << " unts : <" << unts << ">" << G4endl;
    G4cout << " Unit " << unt << G4endl;
    G4double unitValue = ValueOf(unt);
    for (size_t i=0; i<vec.size(); i++) {
      vec[i] *= unitValue;
    }
  }

  return vec;
}

template<typename vContentType>
G4String GateUIcmdWithAVectorAndUnit<vContentType>::ConvertToString
(std::vector<vContentType> vec)
{
  char st[100];
  //LF
  //std::ostrstream os(st,100);
  std::ostringstream os(st,100);
  //
  for (G4int i=0; i<vec.size(); i++) os << vec[i] << " "; 
  os << '\0';
  G4String vl = st;
  G4cout << "GateUIcmdWithAVectorAndUnit::ConvertToString : " << vl << G4endl;
  return vl;
}

#endif





