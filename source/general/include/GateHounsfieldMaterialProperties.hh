/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


/*! 
  \class  GateHounsfieldMaterialProperties.hh
  \brief  
  \author david.sarrut@creatis.insa-lyon.fr
*/
 
#ifndef __GateHounsfieldMaterialProperties__hh__
#define __GateHounsfieldMaterialProperties__hh__

#include "G4Material.hh"

class GateHounsfieldMaterialProperties
{
public:
  GateHounsfieldMaterialProperties();
  ~GateHounsfieldMaterialProperties();

  int GetNumberOfElements();
  G4Element * GetElements(int i);
  double GetElementsFraction(int i);
  G4String GetName();
  double GetH();
  void Read(std::ifstream & is, std::vector<G4String> & el);

protected:
  void ReadAndStoreElementFraction(std::ifstream & is, G4String name);
  std::vector<G4Element*> mElementsList;
  std::vector<double> mElementsFractionList;
  G4String mName;
  double mH;
};
#endif
