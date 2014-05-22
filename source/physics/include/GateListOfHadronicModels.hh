/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*
  \class  GateListOfHadronicModels
  \author thibault.frisson@creatis.insa-lyon.fr
          laurent.guigues@creatis.insa-lyon.fr
	  david.sarrut@creatis.insa-lyon.fr
*/


#ifndef GATELISTOFHADMODEL_HH
#define GATELISTOFHADMODEL_HH

#include "globals.hh"
#include "G4String.hh"

#include <iomanip>
#include <vector>

#include "G4Material.hh"
#include "G4MaterialTable.hh"
#include "G4Element.hh"
#include "G4ElementTable.hh"

#include "G4UnitsTable.hh"

class GateListOfHadronicModels
{
public:
  GateListOfHadronicModels(G4String model);
  ~GateListOfHadronicModels();

  void SetEmin(double val, G4String opt = "NoOption");
  void SetEmax(double val, G4String opt = "NoOption");
  G4double GetEmin(G4String opt = "NoOption");
  G4double GetEmax(G4String opt = "NoOption");
  void ClearERange();

  void Print(G4int level=1, G4String symbol = "*", G4String symbol2 = "-");
  void Print(G4String file,G4int level=1, G4String symbol = "*", G4String symbol2 = "-");

  G4Material * GetMaterial(G4String materialName);
  G4Element * GetElement(G4String elementName);

  std::vector<G4String> GetTheListOfOptions() {return theListOfOptions  ;}
  std::vector<double> GetTheListOfEmin() {return theListOfEmin  ;}
  std::vector<double> GetTheListOfEmax() {return theListOfEmax  ;}
  G4String GetModelName(){return modelName;}
  
  bool IsEnergyRangeDefined() {if(theListOfOptions.size()!=0) return true; return false;}

protected:
  G4String modelName;
  std::vector<G4String> theListOfOptions;
  std::vector<double> theListOfEmin;
  std::vector<double> theListOfEmax;
};

#endif
