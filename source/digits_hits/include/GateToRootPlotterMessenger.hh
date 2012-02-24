/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateToRootPlotterMessenger_h
#define GateToRootPlotterMessenger_h 1

//e #ifdef G4ANALYSIS_USE_ROOT
#ifdef G4ANALYSIS_USE_ROOT_PLOTTER

#include "GateOutputModuleMessenger.hh"

class GateToRootPlotter;

class G4UIdirectory;
class G4UIcmdWithAString;
class G4UIcmdWithABool;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADouble;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3VectorAndUnit;
class G4UIcmdWithoutParameter;

#include "GateUIcmdWithAVector.hh"
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

class GateToRootPlotterMessenger: public GateOutputModuleMessenger
{
  public:
    GateToRootPlotterMessenger(GateToRootPlotter* gateToRootPlotter);
   ~GateToRootPlotterMessenger();
    
    void SetNewValue(G4UIcommand*, G4String);
    
  protected:
    GateToRootPlotter*               m_gateToRootPlotter;
    
    G4UIcmdWithoutParameter*         ResetCmd;
    GateUIcmdWithAVector<G4String>*  AddPlotCmd;
    G4UIcmdWithAnInteger*            RemovePlotCmd;
    G4UIcmdWithoutParameter*         RemoveAllPlotsCmd;
    G4UIcmdWithoutParameter*         ListPlotsCmd;
    G4UIcmdWithAnInteger*            PlotWidthCmd;
    G4UIcmdWithAnInteger*            PlotHeightCmd;
    G4UIcmdWithAnInteger*            NColumnsCmd;
    G4UIcmdWithoutParameter*         PlotAllCmd;
    G4UIcmdWithoutParameter*         ShowPlotterCmd;
    G4UIcmdWithoutParameter*         ClosePlotterCmd;
};

#endif
//e #endif
#endif

