/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateUIcmdWithTwoADouble_H
#define GateUIcmdWithTwoADouble_H 1

#include "G4UIcommand.hh"
#include "G4UImessenger.hh"
#include "GateTokenizer.hh"
//LF
//#include <strstream>
#include <sstream>
//LF

// class description:
//  A concrete class of G4UIcommand. The command defined by this class
// takes an integer and a double.
//  General information of G4UIcommand is given in G4UIcommand.hh.
class GateUIcmdWithTwoDouble : public G4UIcommand
{

  public: // with description
    GateUIcmdWithTwoDouble
    //LF
	//(const char* theCommandPath,G4UImessenger* theMessenger);
	(G4String theCommandPath,G4UImessenger* theMessenger);

    //  Constructor. The command string with full path directory
    // and the pointer to the messenger must be given.
    //G4double GetNewDoubleValue(G4int num,const char* paramString);
	G4double GetNewDoubleValue(G4int num,G4String paramString);
    //  Convert string which represents a double to a double.
    G4String ConvertToString(G4int intValue,G4double dblValue);
    //  Convert an integer value to a string. This method must be used by 
    // the messenger for its GetCurrentValues() method.

};

#endif





