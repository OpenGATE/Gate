/*----------------------
   GATE version name: gate_v...

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateToPlatform_h
#define GateToPlatform_h 1
#include "globals.hh"

/*use this class to generate submit files for different platforms
based on output from GateMacfileParser*/

class GateToPlatform
{
public:
  GateToPlatform();
  GateToPlatform(G4int numberOfSplits, G4String thePlatform, G4String pbsscript,G4String slurmscript,G4String theCondorScript,G4String outputMacName,G4int time);
  ~GateToPlatform();
  void SetVerboseLevel(G4int value) { m_verboseLevel = value; };
  int GenerateSubmitfile(G4String outputMacDir);

protected: 
  int GenerateOpenMosixSubmitfile();
  int GenerateOpenPBSSubmitfile();
  int GenerateOpenPBSScriptfile();
  int GenerateSlurmSubmitfile();
  int GenerateSlurmScriptfile();
  int GenerateCondorSubmitfile();
  int GenerateXgridSubmitfile();    
  G4int m_verboseLevel;  
  G4int nSplits;
  G4String platform;
  G4String pbsScript;
  G4String slurmScript;
  G4String condorScript;
  G4String outputMacfilename;
  G4String outputDir;
  G4int useTiming;
};
#endif


