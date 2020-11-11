/** ----------------------
  Copyright (C): OpenGATE Collaboration
  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/
#ifndef GateExtendedAnalysisMessenger_h
#define GateExtendedAnalysisMessenger_h 1

#include "GateOutputModuleMessenger.hh"

class GateExtendedAnalysis;
class G4UIdirectory;
class G4UIcmdWithAString;
class G4UIcmdWithABool;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADouble;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3VectorAndUnit;
class G4UIcmdWithoutParameter;

/** Author: Mateusz Bała
 *  Email: bala.mateusz@gmail.com
 *  Organization: J-PET (http://koza.if.uj.edu.pl/pet/)
 *  About class: Messenger for GateExtendedAnalysis class
 **/
class GateExtendedAnalysisMessenger: public GateOutputModuleMessenger
{
  public:
    GateExtendedAnalysisMessenger(GateExtendedAnalysis* gateExtendedAnalysis);
    virtual ~GateExtendedAnalysisMessenger() = default;

    virtual void SetNewValue(G4UIcommand*, G4String) override;

  protected:
    GateExtendedAnalysis* m_GateExtendedAnalysis = nullptr;
};

#endif
