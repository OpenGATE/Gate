/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GATEUICMDWITHADOUBLEWITHUNITANDINTEGER_CC
#define GATEUICMDWITHADOUBLEWITHUNITANDINTEGER_CC

#include "GateUIcmdWithADoubleWithUnitAndInteger.hh"

//---------------------------------------------------------------------------
GateUIcmdWithADoubleWithUnitAndInteger::GateUIcmdWithADoubleWithUnitAndInteger(const char * theCommandPath,G4UImessenger * theMessenger)
:G4UIcommand(theCommandPath,theMessenger)
{
  G4UIparameter * strParam1 = new G4UIparameter('d');
  SetParameter(strParam1);
  G4UIparameter * strParam2 = new G4UIparameter('s');
  SetParameter(strParam2);
  G4UIparameter * strParam3 = new G4UIparameter('i');
  SetParameter(strParam3);
 
}
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
void GateUIcmdWithADoubleWithUnitAndInteger::SetParameterName
(const char * theName1, const char * theName2, const char * theName3,
 G4bool omittable1, G4bool omittable2, G4bool omittable3, G4bool currentAsDefault)
{
  G4UIparameter * theParam = GetParameter(0);
  theParam->SetParameterName(theName1);
  theParam->SetOmittable(omittable1);
  theParam->SetCurrentAsDefault(currentAsDefault);
  
  theParam = GetParameter(1);
  theParam->SetParameterName(theName2);
  theParam->SetOmittable(omittable2);
  theParam->SetCurrentAsDefault(currentAsDefault);

  theParam = GetParameter(2);
  theParam->SetParameterName(theName3);
  theParam->SetOmittable(omittable3);
  theParam->SetCurrentAsDefault(currentAsDefault);

}
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
void GateUIcmdWithADoubleWithUnitAndInteger::SetCandidates(const char * candidateList1, const char * candidateList2,
								  const char * candidateList3)
{
  G4UIparameter * theParam = GetParameter(0);
  G4String canList = candidateList1;
  theParam->SetParameterCandidates(canList);

  theParam = GetParameter(1);
  canList = candidateList2;
  theParam->SetParameterCandidates(canList);

  theParam = GetParameter(2);
  canList = candidateList3;
  theParam->SetParameterCandidates(canList);
}
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
void GateUIcmdWithADoubleWithUnitAndInteger::SetDefaultValue(const char * defVal1, const char * defVal2,
								    const char * defVal3 )
{
  G4UIparameter * theParam = GetParameter(0);
  theParam->SetDefaultValue(defVal1);

  theParam = GetParameter(1);
  theParam->SetDefaultValue(defVal2);

  theParam = GetParameter(2);
  theParam->SetDefaultValue(defVal3);

}
//---------------------------------------------------------------------------

#endif

