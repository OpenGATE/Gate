/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#include "GateNamedObject.hh"

#include "GateTools.hh"

//----------------------------------------------------------------------------------------------------
// Print-out a description of the object
void GateNamedObject::Describe(size_t indent)
{
  G4cout << Gateendl << GateTools::Indent(indent) << "GATE object:        '" << mName << "'\n";
}
//----------------------------------------------------------------------------------------------------

