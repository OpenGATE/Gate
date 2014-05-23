/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateUIcmdWithAVector_H
#define GateUIcmdWithAVector_H 1

#include "G4UIcommand.hh"
#include <vector>
#include "GateTokenizer.hh"

//LF
//#include <strstream>
#include <sstream>
//LF

// Class description:
//  A concrete class of G4UIcommand. The command defined by this class
// takes a vector<> and a unit.
//  General information of G4UIcommand is given in G4UIcommand.hh.

template<typename vContentType> class GateUIcmdWithAVector : public G4UIcommand
{
public: // with description
  GateUIcmdWithAVector(const char * theCommandPath, G4UImessenger * theMessenger);
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
GateUIcmdWithAVector<vContentType>::GateUIcmdWithAVector
(const char * theCommandPath, G4UImessenger * theMessenger)
:G4UIcommand(theCommandPath,theMessenger)
{
  //  G4UIparameter * intParam = new G4UIparameter('i');
  G4UIparameter * intParam = new G4UIparameter('s');
  SetParameter(intParam);
}

template<typename vContentType>
std::vector<vContentType> GateUIcmdWithAVector<vContentType>::GetNewVectorValue(G4String paramString)
{
//    G4cout << "GateUIcmdWithAVector::GetNewVectorValue : paramString <" << paramString << ">" << G4endl;

  //const char* t = paramString;
  //LF 21/12/2005
  // std::istrstream is(paramString);
  std::istringstream is(paramString);
  //LF
  std::vector<vContentType> vec;
//    G4int nElem;
//    is >> nElem;
  //  G4cout << "GateUIcmdWithAVector::GetNewVectorValue : nElem "<< nElem << G4endl;
  vContentType value;
  // TEST  G4int value;
//    G4cout << "GateUIcmdWithAVector::GetNewVectorValue : ";
  G4String aToken;
  GateTokenizer parameterToken( paramString );
  //G4bool isGood = true;
  std::vector<G4String> inputVec;
  G4int length = 0;
  do {
    aToken = parameterToken();
    length = aToken.length();
    if (length>0) {
      inputVec.push_back(aToken);
    }
  } while (length>0);

//  G4cout << "GateUIcmdWithAVector::GetNewVectorValue : inputVec.size() : " << inputVec.size() << G4endl;

  for (size_t iw=0; iw<inputVec.size(); iw++) {
    aToken = inputVec[iw] + G4String(" dummy ") ;
    const char* charToken = aToken; 
    //LF 21/12/2005
    //std::istrstream istrToken(charToken); 
    std::istringstream istrToken(charToken); 
    //LF
    istrToken >> value;
//      G4cout << "GateUIcmdWithAVector::GetNewVectorValue : aToken : " << aToken << " value : " << value << G4endl;
//      G4cout << " is.eof() "         << istrToken.eof() 
//  	   << " istrToken.fail() " << istrToken.fail() 
//  	   << " istrToken.good() " << istrToken.good() 
//  	   << " istrToken.bad() "  << istrToken.bad() << G4endl;
    if (istrToken.good() == 0) {
      ;//isGood = false;
    } else {
      vec.push_back(value);
//        G4cout << value << ", ";
    }
  }
//    G4cout << G4endl;

//  G4cout << "GateUIcmdWithAVector::GetNewVectorValue : vec.size() " << vec.size() << G4endl;

  return vec;
}

template<typename vContentType>
G4String GateUIcmdWithAVector<vContentType>::ConvertToString
(std::vector<vContentType> vec)
{
  std::ostringstream os;
  for (G4int i=0; i<vec.size(); i++) os << vec[i] << " "; 
  os << '\0';
  G4String vl = os.str();
  G4cout << "GateUIcmdWithAVector::ConvertToString : " << vl << G4endl;
  return vl;
}

#endif





