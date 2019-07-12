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
#include "GateExtendedVSourceManager.hh"

GateExtendedVSourceManager* GateExtendedVSourceManager::ptrSingletonJPETSourceManager = 0;

GateExtendedVSourceManager::GateExtendedVSourceManager() {}

GateExtendedVSourceManager::~GateExtendedVSourceManager() {}

GateExtendedVSourceManager* GateExtendedVSourceManager::GetInstance()
{
 if ( !ptrSingletonJPETSourceManager) { ptrSingletonJPETSourceManager = new GateExtendedVSourceManager; }

 return ptrSingletonJPETSourceManager;
}

void GateExtendedVSourceManager::AddGammaSourceModel( GateGammaSourceModel* model )
{
 if ( model )
 {
  if ( !GetGammaSourceModelByName( model->GetModelName() ) )
  {
   G4cout << "GateExtendedVSourceManager::AddGammaSourceModel - Added model : '"<<model->GetModelName() << "' " << G4endl;
   fGammaSourceModelsNames += "<" + model->GetModelName() + "> ";
   fGammaSourceModels.emplace( model->GetModelName(), model );
  }
 } else { G4cout << "GateExtendedVSourceManager::AddGammaSourceModel - null pointer to GateGammaSourceModel model." << G4endl; }
}

GateGammaSourceModel* GateExtendedVSourceManager::GetGammaSourceModelByName( const G4String& modelName ) const
{
 std::map<G4String, GateGammaSourceModel*>::const_iterator found = fGammaSourceModels.find( modelName );

 if ( found != fGammaSourceModels.end() ) { return found->second; }
 
 return nullptr;
}

G4String GateExtendedVSourceManager::GetGammaSourceModelsNames() const { return fGammaSourceModelsNames; }
