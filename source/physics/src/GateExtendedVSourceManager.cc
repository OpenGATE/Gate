/** ----------------------
  Copyright (C): OpenGATE Collaboration
  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

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
