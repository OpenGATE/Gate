/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GATEUICMDWITHADOUBLEAND3STRING_CC
#define GATEUICMDWITHADOUBLEAND3STRING_CC

#include "GateUIcmdWithADoubleAnd3String.hh"

//---------------------------------------------------------------------------
GateUIcmdWithADoubleAnd3String::GateUIcmdWithADoubleAnd3String(const char * theCommandPath,G4UImessenger * theMessenger)
:G4UIcommand(theCommandPath,theMessenger)
{
  G4UIparameter * strParam1 = new G4UIparameter('d');
  SetParameter(strParam1);
  G4UIparameter * strParam2 = new G4UIparameter('s');
  SetParameter(strParam2);
  G4UIparameter * strParam3 = new G4UIparameter('s');
  SetParameter(strParam3);
  G4UIparameter * strParam4 = new G4UIparameter('s');
  SetParameter(strParam4);
}
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
void GateUIcmdWithADoubleAnd3String::SetParameterName
(const char * theName1, const char * theName2, const char * theName3, const char * theName4, 
 G4bool omittable1, G4bool omittable2, G4bool omittable3, G4bool omittable4, G4bool currentAsDefault)
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

  theParam = GetParameter(3);
  theParam->SetParameterName(theName4);
  theParam->SetOmittable(omittable4);
  theParam->SetCurrentAsDefault(currentAsDefault);

}
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
void GateUIcmdWithADoubleAnd3String::SetCandidates(const char * candidateList1, const char * candidateList2,
                                                     const char * candidateList3, const char * candidateList4)
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

  theParam = GetParameter(3);
  canList = candidateList4;
  theParam->SetParameterCandidates(canList);
}
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
void GateUIcmdWithADoubleAnd3String::SetDefaultValue(const char * defVal1, const char * defVal2,
                                                        const char * defVal3, const char * defVal4  )
{
  G4UIparameter * theParam = GetParameter(0);
  theParam->SetDefaultValue(defVal1);

  theParam = GetParameter(1);
  theParam->SetDefaultValue(defVal2);

  theParam = GetParameter(2);
  theParam->SetDefaultValue(defVal3);

  theParam = GetParameter(3);
  theParam->SetDefaultValue(defVal4);
}
//---------------------------------------------------------------------------

#endif

