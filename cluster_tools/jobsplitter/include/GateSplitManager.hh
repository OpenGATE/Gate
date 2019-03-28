/*----------------------
   GATE version name: gate_v...

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateSplitManager_h
#define GateSplitManager_h 1
#include "globals.hh"
#include "GateToPlatform.hh"
#include "GateMacfileParser.hh"

/*use this class as a manager class for the whole program*/

class GateSplitManager
{
public:

  GateSplitManager(G4int nAliases,G4String* aliases,G4String platform,G4String pbsscript,G4String slurmscript,G4String condorscript,G4String macfile,G4int nSplits,G4int time);
  ~GateSplitManager();
  void SetVerboseLevel(G4int value) { m_verboseLevel = value; };
  void StartSplitting();

protected:

  GateToPlatform*    toPlatform;
  GateMacfileParser* macParser;
  G4String      m_name;       
  G4int         m_verboseLevel;  
  void CleanAbort();
  void CheckEnvironment();
  G4int numberOfSplits;
};
#endif


