/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateUIcontrolMessenger.hh"

#include "G4UIdirectory.hh"
#include "GateUIcmdWithTwoDouble.hh"

#include "G4UImanager.hh"
#include "G4UIbatch.hh"

#include "GateTools.hh"


// Constructor
//LF
GateUIcmdWithTwoDouble::GateUIcmdWithTwoDouble
//(const char * theCommandPath,G4UImessenger * theMessenger)
(G4String theCommandPath,G4UImessenger * theMessenger)
: G4UIcommand(theCommandPath,theMessenger)
{
   G4UIparameter * dblParam = new G4UIparameter('d');
   SetParameter(dblParam);
   dblParam->SetParameterName("Value1");
   dblParam = new G4UIparameter('d');
   SetParameter(dblParam);
   dblParam->SetParameterName("Value2");
}
//_____________________________________________________________________________
//LF
//G4double GateUIcmdWithTwoDouble::GetNewDoubleValue(G4int num,const char* paramString)
G4double GateUIcmdWithTwoDouble::GetNewDoubleValue(G4int num,G4String paramString)
//LF
{
  G4double vl[2];
  //std::istrstream is((char*)paramString);
  std::istringstream is(paramString);
  is >>  vl[0] >>vl[1];
  
  return vl[num];
}
//_____________________________________________________________________________
G4String GateUIcmdWithTwoDouble::ConvertToString(G4int intValue,G4double dblValue)
{
  //char st[100];
  //std::ostrstream os(st,100);
  std::ostringstream os;
  os<<intValue<<' '<<dblValue<< '\0';
  //G4String vl = st;
  G4String vl = os.str();
  return vl;
}
