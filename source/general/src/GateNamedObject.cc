/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateNamedObject.hh"

#include "GateTools.hh"

//----------------------------------------------------------------------------------------------------
// Print-out a description of the object
void GateNamedObject::Describe(size_t indent)
{
  G4cout << G4endl << GateTools::Indent(indent) << "GATE object:        '" << mName << "'" << G4endl;
}
//----------------------------------------------------------------------------------------------------

