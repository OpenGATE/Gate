/**
 *  @copyright Copyright 2016 The J-PET Gate Authors. All rights reserved.
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  @file GateExtendedVSourceManager.hh
 */
#ifndef GateExtendedVSourceManager_hh
#define GateExtendedVSourceManager_hh

#include <iostream>
#include <map>
#include "GateGammaSourceModel.hh"

/**Author: Mateusz Bała
 * Email: bala.mateusz@gmail.com
 * About class: The purpose of this class is to store information about available generation of gamma quanta models and sharing it.
 */
class GateExtendedVSourceManager
{

public:
 ~GateExtendedVSourceManager();

 static GateExtendedVSourceManager* GetInstance();

 void AddGammaSourceModel( GateGammaSourceModel* model );

 GateGammaSourceModel* GetGammaSourceModelByName( const G4String& modelName ) const;

 G4String GetGammaSourceModelsNames() const;

private:
 GateExtendedVSourceManager();
 static GateExtendedVSourceManager* ptrSingletonJPETSourceManager;
 std::map<G4String,GateGammaSourceModel*> fGammaSourceModels;
 G4String fGammaSourceModelsNames = "";
};


#endif


