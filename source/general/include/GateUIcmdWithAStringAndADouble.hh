/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*
  \class  GateUIcmdWithAStringAndADouble
  \author thibault.frisson@creatis.insa-lyon.fr
          laurent.guigues@creatis.insa-lyon.fr
	  david.sarrut@creatis.insa-lyon.fr
*/

#ifndef GATEUICMDWITHASTRINGANDADOUBLE_HH
#define GATEUICMDWITHASTRINGANDADOUBLE_HH

#include "G4UIcommand.hh"

class GateUIcmdWithAStringAndADouble : public G4UIcommand
{
  public: // with description
    GateUIcmdWithAStringAndADouble(const char * theCommandPath,G4UImessenger * theMessenger);
    //  Constructor. The command string with full path directory
    // and the pointer to the messenger must be given.
  void SetParameterName(const char * theName1, const char * theName2,
			G4bool omittable1, G4bool omittable2, G4bool currentAsDefault=false);
    //  Set the parameter name.
    //  If "omittable" is set as true, the user of this command can ommit
    // the value when he/she applies the command. If "omittable" is false,
    // the user must supply the parameter string.
    //  "currentAsDefault" flag is valid only if "omittable" is true. If this
    // flag is true, the current value is used as the default value when the 
    // user ommit the parameter. If this flag is false, the value given by the 
    // next SetDefaultValue() method is used.
  void SetCandidates(const char * candidateList1, const char * candidateList2);

    //  Defines the candidates of the parameter string. Candidates listed in
    // the argument must be separated by space(s).
  void SetDefaultValue(const char * defVal1, const char * defVal2);
    //  Set the default value of the parameter. This default value is used
    // when the user of this command ommits the parameter value, and
    // "ommitable" is true and "curreutAsDefault" is false.
};

#endif
