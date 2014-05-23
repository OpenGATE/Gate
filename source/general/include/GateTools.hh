/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateTools_h
#define GateTools_h 1

#include "globals.hh"
#include "GateConfiguration.h"

#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"

/*! \namespace  GateTools
    \brief  Namespace to provide general-purpose utility functions to GATE
    
    - GateTools - by Daniel.Strul@iphe.unil.ch
    
    - The GateTools namespace is a collection of functions providing some general-purpose
      utility functions to GATE:
      - Indent() to compute a string made of tabulations
      - GetBaseName() to extract the beginning of a string terminated by a tag
      - FindGateFile() to find a file in the GATE directories 
    
*/      
namespace GateTools
{
  //! Returns a string composed of a number of tabulations
  G4String Indent(size_t indent);

  //! Checks if a tag can be found within a string.
  //! If the tag is found, returns everything before the tag.
  //! If the tag is not found, returns the whole string.
  G4String GetBaseName(const G4String& name,const G4String& tag);

  //! Looks for a GATE file: if the file can not be found in the current 
  //! directory, FindGateFile() looks for it in the directory $GATEHOME.
  //! The function returns either the full file path or "" (file not found).
  G4String FindGateFile(const G4String& fileName);

}

#endif

