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
  
  struct mElementCompound
  {
    G4Element* Element;
    double Fraction;
  };
  typedef std::vector<mElementCompound> mElementVector;
  typedef mElementVector::iterator iterator;
  typedef mElementVector::const_iterator const_iterator;
  iterator begin(){ return mElementsList.begin(); }
  iterator end(){ return mElementsList.end(); }

  int GetNumberOfElements();
  inline G4Element * GetElements(int i) { return mElementsList[i].Element; }
  inline double GetElementsFraction(int i) { return mElementsList[i].Fraction; }
  G4String GetName();
  double GetH();
  void Read(std::ifstream & is, std::vector<G4String> & el);

protected:
  void ReadAndStoreElementFraction(std::ifstream & is, G4String name);
  mElementVector mElementsList;
  G4String mName;
  double mH;
};
#endif
