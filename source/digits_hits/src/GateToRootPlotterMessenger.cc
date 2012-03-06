/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "GateConfiguration.h"

#ifdef G4ANALYSIS_USE_ROOT
#ifdef G4ANALYSIS_USE_ROOT_PLOTTER

#include "GateToRootPlotterMessenger.hh"
#include "GateToRootPlotter.hh"
#include "GateOutputMgr.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithoutParameter.hh"


//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

GateToRootPlotterMessenger::GateToRootPlotterMessenger(GateToRootPlotter* gateToRootPlotter)
  : GateOutputModuleMessenger(gateToRootPlotter)
  , m_gateToRootPlotter(gateToRootPlotter)
{ 
  G4String cmdName;

  cmdName = GetDirectoryName()+"reset";
  ResetCmd = new G4UIcmdWithoutParameter(cmdName,this);
  ResetCmd->SetGuidance("Reset the output");

  cmdName = GetDirectoryName()+"addPlot";
  AddPlotCmd = new GateUIcmdWithAVector<G4String>(cmdName,this);
  AddPlotCmd->SetGuidance("Add a plot to the plotter");
  AddPlotCmd->SetGuidance("1. type ('hist' / 'tree')");
  AddPlotCmd->SetGuidance("2. name of hist / tree");
  AddPlotCmd->SetGuidance("3. in case of 'tree', name of the variable");
  AddPlotCmd->SetGuidance("4. in case of 'tree', cut expression [opt.]");

  cmdName = GetDirectoryName()+"removePlot";
  RemovePlotCmd = new G4UIcmdWithAnInteger(cmdName,this);
  RemovePlotCmd->SetGuidance("Remove a plot from the plotter");
  RemovePlotCmd->SetGuidance("1. progressive number of the plot to remove");

  cmdName = GetDirectoryName()+"removeAllPlots";
  RemoveAllPlotsCmd = new G4UIcmdWithoutParameter(cmdName,this);
  RemoveAllPlotsCmd->SetGuidance("Remove all plots from the plotter");

  cmdName = GetDirectoryName()+"listPlots";
  ListPlotsCmd = new G4UIcmdWithoutParameter(cmdName,this);
  ListPlotsCmd->SetGuidance("Remove all plots from the plotter");

  cmdName = GetDirectoryName()+"setPlotWidth";
  PlotWidthCmd = new G4UIcmdWithAnInteger(cmdName,this);
  PlotWidthCmd->SetGuidance("Set the width of the single plot");
  PlotWidthCmd->SetGuidance("1. Width of the plot (integer)");

  cmdName = GetDirectoryName()+"setPlotHeight";
  PlotHeightCmd = new G4UIcmdWithAnInteger(cmdName,this);
  PlotHeightCmd->SetGuidance("Set the width of the single plot");
  PlotHeightCmd->SetGuidance("1. Width of the plot (integer)");

  cmdName = GetDirectoryName()+"setNColumns";
  NColumnsCmd = new G4UIcmdWithAnInteger(cmdName,this);
  NColumnsCmd->SetGuidance("Set the width of the single plot");
  NColumnsCmd->SetGuidance("1. Width of the plot (integer)");
  NColumnsCmd->SetParameterName("N",false);
  NColumnsCmd->SetRange("N>0");

  cmdName = GetDirectoryName()+"plotAll";
  PlotAllCmd = new G4UIcmdWithoutParameter(cmdName,this);
  PlotAllCmd->SetGuidance("Plot all plots in the plotter");

  cmdName = GetDirectoryName()+"showPlotter";
  ShowPlotterCmd = new G4UIcmdWithoutParameter(cmdName,this);
  ShowPlotterCmd->SetGuidance("Open (if needed) the plotter window");

  cmdName = GetDirectoryName()+"closePlotter";
  ClosePlotterCmd = new G4UIcmdWithoutParameter(cmdName,this);
  ClosePlotterCmd->SetGuidance("Open (if needed) the plotter window");

}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

GateToRootPlotterMessenger::~GateToRootPlotterMessenger()
{
  delete ClosePlotterCmd;
  delete ShowPlotterCmd;
  delete PlotAllCmd;
  delete NColumnsCmd;
  delete PlotHeightCmd;
  delete PlotWidthCmd;
  delete ListPlotsCmd;
  delete RemoveAllPlotsCmd;
  delete RemovePlotCmd;
  delete AddPlotCmd;
  delete ResetCmd;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void GateToRootPlotterMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{ 
  if( command == ResetCmd ) { 
    m_gateToRootPlotter->Reset();
  } else if ( command ==  AddPlotCmd) {
    m_gateToRootPlotter->AddPlot(AddPlotCmd->GetNewVectorValue(newValue));
  } else if ( command ==  RemovePlotCmd) {
    m_gateToRootPlotter->RemovePlot(RemovePlotCmd->GetNewIntValue(newValue));
  } else if ( command ==  RemoveAllPlotsCmd) {
    m_gateToRootPlotter->RemoveAllPlots();
  } else if ( command ==  ListPlotsCmd) {
    m_gateToRootPlotter->ListPlots();
  } else if ( command ==  PlotWidthCmd) {
    m_gateToRootPlotter->SetPlotWidth(PlotWidthCmd->GetNewIntValue(newValue));
  } else if ( command ==  PlotHeightCmd) {
    m_gateToRootPlotter->SetPlotHeight(PlotHeightCmd->GetNewIntValue(newValue));
  } else if ( command ==  NColumnsCmd) {
    m_gateToRootPlotter->SetNColumns(NColumnsCmd->GetNewIntValue(newValue));
  } else if ( command ==  PlotAllCmd) {
    m_gateToRootPlotter->PlotAll();
  } else if ( command ==  ShowPlotterCmd) {
    m_gateToRootPlotter->ShowPlotter();
  } else if ( command ==  ClosePlotterCmd) {
    m_gateToRootPlotter->ClosePlotter();
  } else {
    GateOutputModuleMessenger::SetNewValue(command,newValue);
  }
   
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

#endif
#endif
