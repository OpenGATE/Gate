/** ----------------------
  Copyright (C): OpenGATE Collaboration
  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/
#ifndef GateExtendedVSourceManager_hh
#define GateExtendedVSourceManager_hh

#include <iostream>
#include <map>
#include "GateGammaSourceModel.hh"

/** Author: Mateusz Bała
 *  Email: bala.mateusz@gmail.com
 *  Organization: J-PET (http://koza.if.uj.edu.pl/pet/)
 *  About class: The purpose of this class is to store information about available generation of gamma quanta models and sharing it.
 **/
class GateExtendedVSourceManager
{

public:
 ~GateExtendedVSourceManager();

 static GateExtendedVSourceManager* GetInstance();

 void AddGammaSourceModel( GateGammaSourceModel* model );

 GateGammaSourceModel* GetGammaSourceModelByName( const G4String& modelName ) const;

 G4String GetGammaSourceModelsNames() const;

protected:
 GateExtendedVSourceManager();
 static GateExtendedVSourceManager* ptrSingletonJPETSourceManager;
 std::map<G4String,GateGammaSourceModel*> fGammaSourceModels;
 G4String fGammaSourceModelsNames = "";
};


#endif


