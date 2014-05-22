/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

//LF
//#include <strstream>
#include <sstream>
//LF
#include "GateUIterminal.hh"

//////////////////////////////////////////////////////
//G4int GateUIterminal::ReceiveG4cout(G4String coutString)
#ifdef Geant496_COMPATIBILITY
G4int GateUIterminal::ReceiveG4cout(const G4String& coutString)
//////////////////////////////////////////////////////
{
  if ( coutString == "G4PhysicalVolumeModel::Validate() called.\n" )
    return 0;
  if ( coutString == "  Volume of the same name and copy number (\"world_P\", copy 0) still exists and is being used.\n" )
    return 0;
  if ( coutString == "  Be warned that this does not necessarily guarantee it's the same\n" )
    return 0;
  if ( coutString == "  volume you originally specified in /vis/scene/add/.\n" )
    return 0;

  std::cout << coutString << std::flush;
  return 0;
}
#else
G4int GateUIterminal::ReceiveG4cout(G4String coutString)
{
  if ( coutString == "G4PhysicalVolumeModel::Validate() called.\n" )
    return 0;
  if ( coutString == "  Volume of the same name and copy number (\"world_P\", copy 0) still exists and is being used.\n" )
    return 0;
  if ( coutString == "  Be warned that this does not necessarily guarantee it's the same\n" )
    return 0;
  if ( coutString == "  volume you originally specified in /vis/scene/add/.\n" )
    return 0;

  std::cout << coutString << std::flush;
  return 0;
} 
#endif

//////////////////////////////////////////////////////
//G4int GateUIterminal::ReceiveG4cerr(G4String cerrString)
#ifdef Geant496_COMPATIBILITY
G4int GateUIterminal::ReceiveG4cerr(const G4String& cerrString)
{
  std::cerr << cerrString << std::flush;
  return 0;
}
#else
G4int GateUIterminal::ReceiveG4cerr(G4String cerrString)
{
  std::cerr << cerrString << std::flush;
  return 0;
}
#endif

