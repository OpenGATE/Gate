/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/


#ifndef GATEVPROCESS_CC
#define GATEVPROCESS_CC

#include "GateVProcess.hh"
#include "GateVProcessMessenger.hh"
#include "GateConfiguration.h"

#include "G4Material.hh"
#include "G4Element.hh"
#include "G4VEnergyLossProcess.hh"
#include "G4VMultipleScattering.hh"
#include "G4HadronicParameters.hh"

//-----------------------------------------------------------------------------
GateVProcess::GateVProcess(G4String name)
{
  pProcess = 0;
  pFinalProcess = 0;
  pMessenger = 0;
  mIsWrapperActive = false;
  theHandler = 0;
  mG4ProcessName = name;
  mKeepSec=false;

  //fix for G4 11.2.1
  G4HadronicParameters::Instance()->SetTimeThresholdForRadioactiveDecay( 3.171e+10*CLHEP::year );
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateVProcess::~GateVProcess()
{
  GateDebugMessageInc("Physic",4,"~GateVProcess -- begin");

  theListOfDefaultParticles.clear();
  theListOfDataSets.clear();
  theListOfModels.clear();
  theListOfSelectedDataSets.clear();

  for (std::vector<G4ParticleDefinition*>::iterator it = theListOfEnabledParticles.begin();
       it != theListOfEnabledParticles.end(); ) {
    it = theListOfEnabledParticles.erase(it);
  }
  for (std::vector<G4ParticleDefinition*>::iterator it = theListOfParticlesWithSelectedDS.begin();
       it != theListOfParticlesWithSelectedDS.end(); ) {
    it = theListOfParticlesWithSelectedDS.erase(it);
  }
  for (std::vector<G4ParticleDefinition*>::iterator it = theListOfParticlesWithSelectedModels.begin();
       it != theListOfParticlesWithSelectedModels.end(); ) {
    it = theListOfParticlesWithSelectedModels.erase(it);
  }

  for (std::vector<GateListOfHadronicModels*>::iterator it = theListOfSelectedModels.begin();
       it != theListOfSelectedModels.end(); ) {
    delete (*it);
    it = theListOfSelectedModels.erase(it);
  }

  for(std::list<G4HadronicInteraction*>::iterator i = theListOfG4HadronicModels.begin();
      i!=theListOfG4HadronicModels.end(); i++) {
    delete (*i);
    i = theListOfG4HadronicModels.erase(i);
  }

  delete theHandler;
  delete pMessenger;

  for(std::list<G4VProcess*>::iterator i = theListOfG4Processes.begin();
      i!=theListOfG4Processes.end(); i++) {
    i = theListOfG4Processes.erase(i);
  }

  pFinalProcess=0;pProcess=0;
  GateDebugMessageDec("Physic",4,"~GateVProcess -- end");
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateVProcess::Delete()
{
  for (std::vector<GateVProcess*>::iterator it = GetTheListOfProcesses()->begin();
       it != GetTheListOfProcesses()->end(); ) {
    (*it)->kill();
    it = GetTheListOfProcesses()->erase(it);
  }
}
//-----------------------------------------------------------------------------



//-----------------------------------------------------------------------------
void GateVProcess::Register()
{
  GetTheListOfProcesses()->push_back(this);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateVProcess::ConstructProcess()
{
  G4ProcessManager * manager = 0;
  G4ProcessVector * processvector = 0;
  G4String pname;

  bool alreadyDefined;

  for(unsigned int k=0; k<theListOfEnabledParticles.size(); k++)
    {//particle
      manager = theListOfEnabledParticles[k]->GetProcessManager();
      processvector = manager->GetProcessList();
      alreadyDefined = false;
      for(int j=0;j<manager->GetProcessListLength();j++)
        {
          pname = (*processvector)[j]->GetProcessName();
          if(pname==mG4ProcessName) alreadyDefined = true;
        }
      if(!alreadyDefined)
        {
          pProcess = CreateProcess(theListOfEnabledParticles[k] );
          theListOfG4Processes.push_back(pProcess);

          if(theListOfParticlesWithSelectedDS.size()!=0)
            {
              for(unsigned int j=0; j<theListOfParticlesWithSelectedDS.size(); j++)
                if(theListOfParticlesWithSelectedDS[j]==theListOfEnabledParticles[k])
                  AddDataSet(theListOfSelectedDataSets[j]);
            }

          if(theListOfParticlesWithSelectedModels.size()!=0)
            {
              for(unsigned int j=0; j<theListOfParticlesWithSelectedModels.size(); j++)
                if(theListOfParticlesWithSelectedModels[j]==theListOfEnabledParticles[k])
                  {
                    GateDebugMessage("Physic",2,"AddModel start - "<< theListOfSelectedModels[j]->GetModelName()
                                     <<"  -   "<< theListOfEnabledParticles[k]->GetParticleName()   << Gateendl);

                    AddModel(theListOfSelectedModels[j]);
                    GateDebugMessage("Physic",2,"AddModel end\n");
                  }
            }

          G4String particle = theListOfEnabledParticles[k]->GetParticleName();

          if(thelistOfFinalRangeForStepFunction.size()!=0)
            {
              if(thelistOfFinalRangeForStepFunction[particle])
                dynamic_cast<G4VEnergyLossProcess*>(pProcess)->SetStepFunction(thelistOfRatioForStepFunction[particle] , thelistOfFinalRangeForStepFunction[particle]);
            }

          if(thelistOfLinearLossLimit.size()!=0)
            {
              if(thelistOfLinearLossLimit[particle]) dynamic_cast<G4VEnergyLossProcess*>(pProcess)->SetLinearLossLimit(thelistOfLinearLossLimit[particle]);
            }

          if(thelistOfMscLimitation.size()!=0)
            {
              if(thelistOfMscLimitation[particle]) dynamic_cast<G4VMultipleScattering*>(pProcess)->SetStepLimitType(thelistOfMscLimitation[particle]);
            }

          if (theListOfWrapperFactor[particle] || theListOfWrapperCSEFactor[particle]) //((GetIsWrapperActive() || GetIsCSEActive()) &&
            {
              //G4cout<<"INIT  "<<pProcess->GetProcessName()<<"  "<<dynamic_cast<G4VEnergyLossProcess*>(pProcess)->IsIonisationProcess()<< Gateendl;
              theListOfWrapper[particle]->RegisterProcess(pProcess);
              if(theListOfWrapperFactor[particle])	   theListOfWrapper[particle]->SetSplitFactor(theListOfWrapperFactor[particle]);
              if(theListOfWrapperCSEFactor[particle])	   theListOfWrapper[particle]->SetCSEFactor(theListOfWrapperCSEFactor[particle]);
              theListOfWrapper[particle]->SetKeepSec(mKeepSec);
              pFinalProcess = theListOfWrapper[particle];
            }
          else pFinalProcess = pProcess;
          ConstructProcess(manager);
        }
    }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateVProcess::PrintEnabledParticles(G4String name)
{
  int nSelectedModel = 0;

  for(unsigned int k=0; k<theListOfEnabledParticles.size(); k++)
    {
      if(name=="All" || name==theListOfEnabledParticles[k]->GetParticleName())
        {
          if(name=="All")
            {
              if(k==0) std::cout<<"\n   ===  Process: "<<mG4ProcessName<<"  ===\n";
              std::cout<<"Particle: "<<theListOfEnabledParticles[k]->GetParticleName()<< Gateendl;
            }
          else std::cout<<"Process: "<<mG4ProcessName<< Gateendl;

          if(theListOfModels.size()!=0)
            {
              std::cout<<"    * Model(s):\n";
              if(theListOfParticlesWithSelectedModels.size()==0)
                std::cout<<"       - <!> *** Warning *** <!> No model selected!\n";
              else
                {
                  nSelectedModel = 0;
                  for(unsigned int i=0; i<theListOfParticlesWithSelectedModels.size(); i++)
                    {
                      if(theListOfEnabledParticles[k]->GetParticleName()==theListOfParticlesWithSelectedModels[i]->GetParticleName())
                        {
                          theListOfSelectedModels[i]->Print(4,"-","+");
                          nSelectedModel++;
                        }
                    }
                  if(nSelectedModel==0)  std::cout<<"       - <!> *** Warning *** <!> No model selected!\n";
                }
            }
          if(theListOfDataSets.size()!=0)
            {
              std::cout<<"    * DataSet(s):\n";
              if(theListOfParticlesWithSelectedDS.size()==0)
                std::cout<<"       - Default\n";

              else
                for(unsigned int i=0; i<theListOfParticlesWithSelectedDS.size(); i++)
                  {
                    if(theListOfEnabledParticles[k]->GetParticleName()==theListOfParticlesWithSelectedDS[i]->GetParticleName())
                      std::cout<<"        - "<<theListOfSelectedDataSets[i] << Gateendl;
                  }
            }
        }
    }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateVProcess::PrintEnabledParticlesToFile(G4String file)
{
  int nSelectedModel = 0;

  std::ofstream os;
  os.open(file.data(), std::ios_base::app);

  for(unsigned int k=0; k<theListOfEnabledParticles.size(); k++)
    {
      if(k==0) os<<"   ===  Process: "<<mG4ProcessName.data()<<"  ===\n";
      os<<"Particle: "<<theListOfEnabledParticles[k]->GetParticleName().data()<<"\n";
      if(theListOfModels.size()!=0)
        {
          os<<"    * Model(s):\n";
          if(theListOfParticlesWithSelectedModels.size()==0)
            os<<"       - <!> *** Warning *** <!> No model selected!\n";
          else
            {
              nSelectedModel = 0;
              os.close();
              for(unsigned int i=0; i<theListOfParticlesWithSelectedModels.size(); i++)
                {
                  if(theListOfEnabledParticles[k]->GetParticleName()==theListOfParticlesWithSelectedModels[i]->GetParticleName())
                    {
                      theListOfSelectedModels[i]->Print(file,4,"-","+");
                      nSelectedModel++;
                    }
                }
              os.open(file.data(), std::ios_base::app);
              if(nSelectedModel==0)  os<<"       - <!> *** Warning *** <!> No model selected!\n";
            }
        }
      if(theListOfDataSets.size()!=0)
        {
          os<<"    * DataSet(s):\n";
          if(theListOfParticlesWithSelectedDS.size()==0)
            os<<"       - Default\n";
          else
            for(unsigned int i=0; i<theListOfParticlesWithSelectedDS.size(); i++)
              {
                if(theListOfEnabledParticles[k]->GetParticleName()==theListOfParticlesWithSelectedDS[i]->GetParticleName())
                  os<<"        - "<<theListOfSelectedDataSets[i].data()<<"\n";
              }
        }
      os<<"\n";
    }
  os.close();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateVProcess::CreateEnabledParticle(G4String par)
{
  std::vector<G4ParticleDefinition*>  theListOfParticles = GetParticles(par);

  G4ParticleDefinition * particle=0;

  for(unsigned int i=0; i<theListOfParticles.size(); i++)
    {
      particle = theListOfParticles[i];

      if( IsEnabled(particle) )
        {
          GateWarning(mG4ProcessName<<" already selected for "<< particle->GetParticleName() );
          continue;
        }
      if( IsApplicable(particle) )  theListOfEnabledParticles.push_back(particle);
      else  GateWarning(mG4ProcessName<<" is not applicable to "<< particle->GetParticleName() );
    }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateVProcess::RemoveElementOfParticleList(G4String par )
{
  std::vector<G4ParticleDefinition*>  theListOfParticles = GetParticles(par);

  for(unsigned int j=0; j<theListOfParticles.size(); j++)
    {
      std::vector<G4ParticleDefinition *>::iterator lIt;

      for (lIt=theListOfEnabledParticles.begin();lIt !=theListOfEnabledParticles.end();)
        {

          if( (*lIt)==theListOfParticles[j] ) lIt = theListOfEnabledParticles.erase(lIt);
          else ++lIt;
        }
    }

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GateVProcess::IsEnabled(G4ParticleDefinition * par)
{
  for(unsigned int k=0; k<theListOfEnabledParticles.size(); k++)
    {
      if(theListOfEnabledParticles[k]==par) return true;
    }

  return false;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
std::vector<G4ParticleDefinition*> GateVProcess::GetParticles(G4String param)
{
  std::vector<G4ParticleDefinition*> theListOfG4Particles;
  std::vector<G4String> theListOfParticles;

  if(param=="Default")
    {
      for(unsigned int i=0; i<theListOfDefaultParticles.size(); i++)
        {
          std::vector<G4String> tmplist;
          tmplist = FindParticleName(theListOfDefaultParticles[i]);
          for(unsigned int j=0; j<tmplist.size(); j++)
            theListOfParticles.push_back(tmplist[j]);
        }
    }
  else
    {
      theListOfParticles = FindParticleName(param);
    }
  if(theListOfParticles.size()==0) G4cout<< "\n  <!> *** Warning *** <!> Unknown particle: "<<param<< Gateendl;

  G4ParticleDefinition* particle = 0;
  G4ParticleTable* theParticleTable = 0;

  theParticleTable = G4ParticleTable::GetParticleTable();

  for(unsigned int i=0; i<theListOfParticles.size(); i++)
    {
      particle = theParticleTable->FindParticle(theListOfParticles[i]);
      theListOfG4Particles.push_back(particle);
    }

  return theListOfG4Particles;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
std::vector<G4String> GateVProcess::FindParticleName(G4String name)
{
  std::vector<G4String> theListOfParticles;

  G4ParticleTable* theParticleTable = 0;
  theParticleTable = G4ParticleTable::GetParticleTable();

  G4ParticleTable::G4PTblDicIterator & particleIterator(*theParticleTable->GetIterator());
  particleIterator.reset();

  while(particleIterator())
    {
      G4ParticleDefinition * particle(particleIterator.value());
      if(particle->GetParticleName() == name)
        {
          theListOfParticles.push_back(name);
          return theListOfParticles;
        }
    }

  if(name=="EM" || name=="em")
    {
      theListOfParticles.push_back("gamma");
      theListOfParticles.push_back("e+");
      theListOfParticles.push_back("e-");
    }
  else if(name=="charged" || name=="Charged" )
    {
      particleIterator.reset();

      while (particleIterator())
        {
          G4ParticleDefinition * particle(particleIterator.value());
          if(particle->GetPDGCharge() != 0.0)
            theListOfParticles.push_back(particle->GetParticleName());
        }
    }
  return theListOfParticles;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/*void GateVProcess::MessengerInitialization()
  {

  }*/

//-----------------------------------------------------------------------------
void GateVProcess::SetDataSet(G4String cs ,G4String par)
{
  std::vector<G4ParticleDefinition*>  theListOfParticles = GetParticles(par);
  G4ParticleDefinition * particle=0;

  bool alreadySet = false;

  for(unsigned int i=0; i<theListOfParticles.size(); i++)
    {
      alreadySet = false;
      particle = theListOfParticles[i];

      if( !IsDatasetApplicable(cs, particle) ){
        GateWarning("DataSet ("<<cs<<") is not applicable for "<<particle->GetParticleName());
        continue;
      }

      for(unsigned int j=0; j<theListOfParticlesWithSelectedDS.size(); j++)
        if(theListOfParticlesWithSelectedDS[j]==particle && theListOfSelectedDataSets[j]==cs)
          {
            alreadySet=true;
            GateWarning("DataSet ("<<cs<<") already selected for "<<particle->GetParticleName());
          }

      if(alreadySet == false)
        {
          theListOfParticlesWithSelectedDS.push_back(particle);
          theListOfSelectedDataSets.push_back(cs );
        }

    }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateVProcess::UnSetDataSet(G4String cs ,G4String par )
{
  std::vector<G4String>::iterator lIt;
  std::vector<G4ParticleDefinition *>::iterator lItpart=theListOfParticlesWithSelectedDS.begin();


  if(par=="All")
    {
      for(lIt=theListOfSelectedDataSets.begin();lIt !=theListOfSelectedDataSets.end();)
        {
          if( (*lIt) == cs )
            {
              lIt = theListOfSelectedDataSets.erase(lIt);
              lItpart = theListOfParticlesWithSelectedDS.erase(lItpart);
            }
          else {++lIt;++lItpart;}
        }
    }
  else
    {
      std::vector<G4ParticleDefinition*>  theListOfParticles = GetParticles(par);

      G4ParticleDefinition * particle=0;

      for(lIt=theListOfSelectedDataSets.begin();lIt !=theListOfSelectedDataSets.end();)
        {
          for(unsigned int j=0; j<theListOfParticles.size(); j++)
            {
              particle=theListOfParticles[j];
              if( (*lIt) == cs && particle == (*lItpart))
                {
                  lIt = theListOfSelectedDataSets.erase(lIt);
                  lItpart = theListOfParticlesWithSelectedDS.erase(lItpart);
                }
              else {++lIt;++lItpart;}
            }
        }
    }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateVProcess::DataSetList(G4String par, G4int level, G4String symbol, G4String symbol2)
{
  G4String space = " ";
  for(G4int i = 1;i<level;i++) space += "  ";
  G4String space2 = space + "  " + symbol2 + " ";
  space += symbol + " ";

  if(theListOfDataSets.size()==0)
    {
      G4cout<< "No DataSet for this process ("<<mG4ProcessName<<")\n";
      return;
    }

  std::vector<G4ParticleDefinition*>  theListOfParticles = GetParticles(par);
  G4ParticleDefinition * particle=0;

  for(unsigned int i=0; i<theListOfParticles.size(); i++)
    {
      particle = theListOfParticles[i];
      G4cout<<space<<"DataSet(s) for "<<particle->GetParticleName()<<" :\n";

      for(unsigned int j=0; j<theListOfDataSets.size(); j++)
        if( IsDatasetApplicable(theListOfDataSets[j],particle) )
          G4cout<<space2<<theListOfDataSets[j]<< Gateendl;
    }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateVProcess::SetModel(G4String model,G4String par)
{
  std::vector<G4ParticleDefinition*>  theListOfParticles = GetParticles(par);
  G4ParticleDefinition * particle=0;

  bool alreadySet = false;

  for(unsigned int i=0; i<theListOfParticles.size(); i++)
    {
      alreadySet = false;
      particle = theListOfParticles[i];

      if( !IsModelApplicable(model, particle) ){
        GateWarning("Model ("<<model<<") is not applicable for "<<particle->GetParticleName() );
        continue;
      }

      for(unsigned int j=0; j<theListOfParticlesWithSelectedModels.size(); j++)
        if(theListOfParticlesWithSelectedModels[j]==particle && theListOfSelectedModels[j]->GetModelName()==model)
          {
            alreadySet=true;
            GateWarning("Model ("<<model<<") already selected for "<<particle->GetParticleName() );
          }

      if(alreadySet == false)
        {
          theListOfParticlesWithSelectedModels.push_back(particle);
          theListOfSelectedModels.push_back(new GateListOfHadronicModels(model) );
        }

    }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateVProcess::UnSetModel(G4String model,G4String par)
{
  std::vector<GateListOfHadronicModels *>::iterator lIt;
  std::vector<G4ParticleDefinition *>::iterator lItpart=theListOfParticlesWithSelectedModels.begin();

  if(par=="All")
    {

      for(lIt=theListOfSelectedModels.begin();lIt !=theListOfSelectedModels.end();)
        {
          if( (*lIt)->GetModelName()==model )
            {
              lIt = theListOfSelectedModels.erase(lIt);
              lItpart = theListOfParticlesWithSelectedModels.erase(lItpart);
            }
          else {++lIt;++lItpart;}
        }
    }
  else
    {
      std::vector<G4ParticleDefinition*>  theListOfParticles = GetParticles(par);

      G4ParticleDefinition * particle=0;
      for(unsigned int j=0; j<theListOfParticles.size(); j++)
        {
          particle=theListOfParticles[j];

          lItpart=theListOfParticlesWithSelectedModels.begin();
          for(lIt=theListOfSelectedModels.begin();lIt !=theListOfSelectedModels.end();)
            {
              if( (*lIt)->GetModelName() == model && particle == (*lItpart))
                {
                  lIt = theListOfSelectedModels.erase(lIt);
                  lItpart = theListOfParticlesWithSelectedModels.erase(lItpart);
                }
              else {++lIt;++lItpart;}
            }
        }
    }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateVProcess::ModelList(G4String par, G4int level, G4String symbol, G4String symbol2)
{
  G4String space = " ";
  for(G4int i = 1;i<level;i++) space += "  ";
  G4String space2 = space + "  " + symbol2 + " ";
  space += symbol + " ";

  if(theListOfModels.size()==0)
    {
      G4cout<< "No model for this process ("<<mG4ProcessName<<")\n";
      return;
    }

  std::vector<G4ParticleDefinition*>  theListOfParticles = GetParticles(par);
  G4ParticleDefinition * particle=0;

  for(unsigned int i=0; i<theListOfParticles.size(); i++)
    {
      particle = theListOfParticles[i];
      //GateMessage("Physic",0,space<<"Model(s) for "<<particle->GetParticleName()<<" :\n");
      G4cout<<space<<"Model(s) for "<<particle->GetParticleName()<<" :\n";

      for(unsigned int j=0; j<theListOfModels.size(); j++)
        if( IsModelApplicable(theListOfModels[j],particle) )
          G4cout<<space2<<theListOfModels[j]<< Gateendl;
    }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateVProcess::SetModelEnergyMax(G4String model, G4double energy, G4String par,G4String opt)
{
  std::vector<G4ParticleDefinition*>  theListOfParticles = GetParticles(par);

  G4ParticleDefinition * particle=0;

  bool alreadySet = false;

  for(unsigned int i=0; i<theListOfParticles.size(); i++)
    {
      alreadySet = false;
      particle = theListOfParticles[i];

      if( !IsModelApplicable(model, particle) ) continue;

      for(unsigned int j=0; j<theListOfParticlesWithSelectedModels.size(); j++)
        if(theListOfParticlesWithSelectedModels[j]==particle && theListOfSelectedModels[j]->GetModelName()==model )
          {
            theListOfSelectedModels[j]->SetEmax(energy, opt);
            alreadySet = true;
          }
      if(alreadySet == false)
        GateWarning("Model "<<model<<" not selected for "<<particle->GetParticleName() );
    }

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateVProcess::SetModelEnergyMin(G4String model, G4double energy, G4String par,G4String opt)
{
  std::vector<G4ParticleDefinition*>  theListOfParticles = GetParticles(par);

  G4ParticleDefinition * particle=0;

  bool alreadySet = false;

  for(unsigned int i=0; i<theListOfParticles.size(); i++)
    {
      alreadySet = false;
      particle = theListOfParticles[i];

      if( !IsModelApplicable(model, particle) ) continue;

      for(unsigned int j=0; j<theListOfParticlesWithSelectedModels.size(); j++)

        if(theListOfParticlesWithSelectedModels[j]==particle && theListOfSelectedModels[j]->GetModelName()==model )
          {
            theListOfSelectedModels[j]->SetEmin(energy, opt);
            alreadySet = true;
          }
      if(alreadySet == false)
        GateWarning("Model "<<model<<" not selected for "<<particle->GetParticleName() );

    }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateVProcess::ClearModelEnergyRange(G4String model,G4String par)
{

  if(par=="All") {
    for(unsigned int i=0; i<theListOfParticlesWithSelectedModels.size(); i++)
      if(theListOfSelectedModels[i]->GetModelName()==model) theListOfSelectedModels[i]->ClearERange();
  }
  else
    {
      std::vector<G4ParticleDefinition*>  theListOfParticles = GetParticles(par);

      G4ParticleDefinition * particle=0;

      //bool alreadySet = false;

      for(unsigned int i=0; i<theListOfParticles.size(); i++)
        {
          //alreadySet = false;
          particle = theListOfParticles[i];

          for(unsigned int j=0; j<theListOfParticlesWithSelectedModels.size(); j++)
            {
              if(theListOfParticlesWithSelectedModels[j]==particle && theListOfSelectedModels[j]->GetModelName()==model)
                theListOfSelectedModels[j]->ClearERange();
            }
        }
    }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateVProcess::AddDataSet(G4String csection)
{
  // Elastic --> Default
  if(csection=="G4HadronElasticDataSet")
    {
      //G4HadronElasticDataSet* cs = new G4HadronElasticDataSet();
      //dynamic_cast<G4HadronicProcess*>(pProcess)->AddDataSet(cs);
    }
  /*else if(csection=="G4QElasticCrossSection")
    {
    //G4QElasticCrossSection* cs = new G4QElasticCrossSection();
    dynamic_cast<G4HadronicProcess*>(pProcess)->AddDataSet(G4QElasticCrossSection::GetPointer());
    }*/
  else if(csection=="G4NeutronHPElasticData")// Cross section data set for high precision neutron elastic scattering (user must first download high precision neutron data files from Geant4 web page)
    {
      G4NeutronHPElasticData* cs = new G4NeutronHPElasticData();
      dynamic_cast<G4HadronicProcess*>(pProcess)->AddDataSet(cs);
    }


  // Inelastic
  if(csection=="G4HadronInelasticDataSet")
    {
      //G4HadronInelasticDataSet * cs = new G4HadronInelasticDataSet();
      //dynamic_cast<G4HadronicProcess*>(pProcess)->AddDataSet(cs);
    }


  else if(csection=="G4PiNuclearCrossSection")  // improved cross section data set for pi+ and pi- inelastic scattering
    {
      //G4PiNuclearCrossSection * cs = new G4PiNuclearCrossSection();
      //dynamic_cast<G4HadronicProcess*>(pProcess)->AddDataSet(cs);
    }
  else if(csection=="G4ProtonInelasticCrossSection") // improved cross section data set for proton inelastic scattering
    {
      //G4ProtonInelasticCrossSection * cs = new G4ProtonInelasticCrossSection();
      //dynamic_cast<G4HadronicProcess*>(pProcess)->AddDataSet(cs);
    }
  else if(csection=="G4NeutronInelasticCrossSection") // improved cross section data set for neutron inelastic scattering
    {
      //G4NeutronInelasticCrossSection * cs = new G4NeutronInelasticCrossSection();
      //dynamic_cast<G4HadronicProcess*>(pProcess)->AddDataSet(cs);
    }
  else if(csection=="G4NeutronHPInelasticData") // Cross section data set for high precision neutron inelastic scattering (user must first download high precision neutron data files from Geant4 web page)
    {
      G4NeutronHPInelasticData * cs = new G4NeutronHPInelasticData();
      dynamic_cast<G4HadronicProcess*>(pProcess)->AddDataSet(cs);
    }

  // ion cross sections
  else if(csection=="G4TripathiCrossSection")
    {
      //G4TripathiCrossSection* cs = new G4TripathiCrossSection();
      //dynamic_cast<G4HadronicProcess*>(pProcess)->AddDataSet(cs);//G4HadronicProcess.hh
    }
  else if(csection=="G4IonsKoxCrossSection")
    {
      //G4IonsKoxCrossSection * cs = new G4IonsKoxCrossSection();
      //dynamic_cast<G4HadronicProcess*>(pProcess)->AddDataSet(cs);
    }
  else if(csection=="G4IonsShenCrossSection")
    {
      //G4IonsShenCrossSection* cs = new G4IonsShenCrossSection();
      //dynamic_cast<G4HadronicProcess*>(pProcess)->AddDataSet(cs);
    }
  else if(csection=="G4IonsSihverCrossSection")
    {
      //G4IonsSihverCrossSection * cs = new G4IonsSihverCrossSection();
      //dynamic_cast<G4HadronicProcess*>(pProcess)->AddDataSet(cs);
    }
  else if(csection=="G4TripathiLightCrossSection")
    {
      //G4TripathiLightCrossSection * cs = new G4TripathiLightCrossSection();
      //dynamic_cast<G4HadronicProcess*>(pProcess)->AddDataSet(cs);
    }

  // G4PhotoNuclearCrossSection not yet added. Is it useful?
  /*  else if(csection=="G4PhotoNuclearCrossSection") // Cross section data set for inelastic photo-nuclear interactions
      G4PhotoNuclearCrossSection * cs = new G4PhotoNuclearCrossSection();
      dynamic_cast<G4HadronicProcess*>(pProcess)->AddDataSet(cs);
      }*/

  // User DataSet
  else AddUserDataSet(csection);

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateVProcess::AddModel(GateListOfHadronicModels *model)
{
  // Elastic
#if (G4VERSION_MAJOR == 9)
  if(model->GetModelName() == "G4LElastic")
    {
      theListOfG4HadronicModels.push_back(new G4LElastic);
      //G4LElastic* g4model = new G4LElastic;
      if(model->IsEnergyRangeDefined()) SetEnergyRange(theListOfG4HadronicModels.back(),model);
      dynamic_cast<G4HadronicProcess*>(pProcess)->RegisterMe(theListOfG4HadronicModels.back());
    }
#endif
  if(model->GetModelName() == "G4ElasticHadrNucleusHE")
    {
      theListOfG4HadronicModels.push_back(new G4ElasticHadrNucleusHE);
      //G4ElasticHadrNucleusHE* g4model = new G4ElasticHadrNucleusHE;
      if(model->IsEnergyRangeDefined()) SetEnergyRange(theListOfG4HadronicModels.back(),model);
      dynamic_cast<G4HadronicProcess*>(pProcess)->RegisterMe(theListOfG4HadronicModels.back());
    }
  else if(model->GetModelName() == "G4LEpp")
    {
      theListOfG4HadronicModels.push_back( new G4LEpp);
      //G4LEpp* g4model = new G4LEpp;
      if(model->IsEnergyRangeDefined()) SetEnergyRange(theListOfG4HadronicModels.back(),model);
      dynamic_cast<G4HadronicProcess*>(pProcess)->RegisterMe(theListOfG4HadronicModels.back());
    }
  else if(model->GetModelName() == "G4LEnp")
    {
      theListOfG4HadronicModels.push_back(new G4LEnp);
      //G4LEnp* g4model = new G4LEnp;
      if(model->IsEnergyRangeDefined()) SetEnergyRange(theListOfG4HadronicModels.back(),model);
      dynamic_cast<G4HadronicProcess*>(pProcess)->RegisterMe(theListOfG4HadronicModels.back());
    }
  else if(model->GetModelName() == "G4HadronElastic")
    {
      theListOfG4HadronicModels.push_back(new G4HadronElastic);
      //G4HadronElastic* g4model = new G4HadronElastic;
      if(model->IsEnergyRangeDefined()) SetEnergyRange(theListOfG4HadronicModels.back(),model);
      dynamic_cast<G4HadronicProcess*>(pProcess)->RegisterMe(theListOfG4HadronicModels.back());
    }

  // Nucleus-nucleus
  else if(model->GetModelName() == "G4BinaryLightIonReaction")
    {
      theListOfG4HadronicModels.push_back(new G4BinaryLightIonReaction);
      //G4BinaryLightIonReaction* g4model = new G4BinaryLightIonReaction;
      if(model->IsEnergyRangeDefined()) SetEnergyRange(theListOfG4HadronicModels.back(),model);
      dynamic_cast<G4HadronicProcess*>(pProcess)->RegisterMe(theListOfG4HadronicModels.back());
    }
#if (G4VERSION_MAJOR == 9)
  else if(model->GetModelName() == "G4LEDeuteronInelastic")
    {
      theListOfG4HadronicModels.push_back(new G4LEDeuteronInelastic);
      //G4LEDeuteronInelastic* g4model  = new G4LEDeuteronInelastic();
      if(model->IsEnergyRangeDefined()) SetEnergyRange(theListOfG4HadronicModels.back(),model);
      dynamic_cast<G4HadronicProcess*>(pProcess)->RegisterMe(theListOfG4HadronicModels.back());
    }
  else if(model->GetModelName() == "G4LETritonInelastic")
    {
      theListOfG4HadronicModels.push_back(new G4LETritonInelastic);
      //G4LETritonInelastic* g4model  = new G4LETritonInelastic();
      if(model->IsEnergyRangeDefined()) SetEnergyRange(theListOfG4HadronicModels.back(),model);
      dynamic_cast<G4HadronicProcess*>(pProcess)->RegisterMe(theListOfG4HadronicModels.back());
    }
  else if(model->GetModelName() == "G4LEAlphaInelastic")
    {
      theListOfG4HadronicModels.push_back(new G4LEAlphaInelastic);
      //G4LEAlphaInelastic* g4model  = new G4LEAlphaInelastic();
      if(model->IsEnergyRangeDefined()) SetEnergyRange(theListOfG4HadronicModels.back(),model);
      dynamic_cast<G4HadronicProcess*>(pProcess)->RegisterMe(theListOfG4HadronicModels.back());
    }
#endif
  else if(model->GetModelName() == "G4WilsonAbrasionModel")
    {
      theListOfG4HadronicModels.push_back(new G4WilsonAbrasionModel);
      //G4WilsonAbrasionModel* g4model  = new G4WilsonAbrasionModel();
      if(model->IsEnergyRangeDefined()) SetEnergyRange(theListOfG4HadronicModels.back(),model);
      dynamic_cast<G4HadronicProcess*>(pProcess)->RegisterMe(theListOfG4HadronicModels.back());
    }
  //
  // // G4EMDissociation not yet added -> is it useful?
  //   else if(model->GetModelName() == "G4EMDissociation")
  //   {
  //      G4EMDissociation* g4model  = new G4EMDissociation();
  //      if(model->IsEnergyRangeDefined()) SetEnergyRange(g4model,model);
  //      dynamic_cast<G4HadronicProcess*>(pProcess)->RegisterMe(g4model);
  //   }
  //

  // Low energy parameterized
#if (G4VERSION_MAJOR == 9)
  else if(model->GetModelName() == "G4LEProtonInelastic")
    {
      theListOfG4HadronicModels.push_back(new G4LEProtonInelastic);
      //G4LEProtonInelastic* g4model = new G4LEProtonInelastic;
      if(model->IsEnergyRangeDefined()) SetEnergyRange(theListOfG4HadronicModels.back(),model);
      dynamic_cast<G4HadronicProcess*>(pProcess)->RegisterMe(theListOfG4HadronicModels.back());
    }
  else if(model->GetModelName() == "G4LEPionPlusInelastic")
    {
      theListOfG4HadronicModels.push_back(new G4LEPionPlusInelastic);
      //G4LEPionPlusInelastic* g4model = new G4LEPionPlusInelastic;
      if(model->IsEnergyRangeDefined()) SetEnergyRange(theListOfG4HadronicModels.back(),model);
      dynamic_cast<G4HadronicProcess*>(pProcess)->RegisterMe(theListOfG4HadronicModels.back());
    }
  else if(model->GetModelName() == "G4LEPionMinusInelastic")
    {
      theListOfG4HadronicModels.push_back(new G4LEPionMinusInelastic);
      //G4LEPionMinusInelastic* g4model = new G4LEPionMinusInelastic;
      if(model->IsEnergyRangeDefined()) SetEnergyRange(theListOfG4HadronicModels.back(),model);
      dynamic_cast<G4HadronicProcess*>(pProcess)->RegisterMe(theListOfG4HadronicModels.back());
    }
  else if(model->GetModelName() == "G4LEKaonPlusInelastic")
    {
      theListOfG4HadronicModels.push_back(new G4LEKaonPlusInelastic);
      //G4LEKaonPlusInelastic* g4model = new G4LEKaonPlusInelastic;
      if(model->IsEnergyRangeDefined()) SetEnergyRange(theListOfG4HadronicModels.back(),model);
      dynamic_cast<G4HadronicProcess*>(pProcess)->RegisterMe(theListOfG4HadronicModels.back());
    }
  else if(model->GetModelName() == "G4LEKaonMinusInelastic")
    {
      theListOfG4HadronicModels.push_back(new G4LEKaonMinusInelastic);
      //G4LEKaonMinusInelastic* g4model = new G4LEKaonMinusInelastic;
      if(model->IsEnergyRangeDefined()) SetEnergyRange(theListOfG4HadronicModels.back(),model);
      dynamic_cast<G4HadronicProcess*>(pProcess)->RegisterMe(theListOfG4HadronicModels.back());
    }
  else if(model->GetModelName() == "G4LEKaonZeroLInelastic")
    {
      theListOfG4HadronicModels.push_back(new G4LEKaonZeroLInelastic);
      //G4LEKaonZeroLInelastic* g4model = new G4LEKaonZeroLInelastic;
      if(model->IsEnergyRangeDefined()) SetEnergyRange(theListOfG4HadronicModels.back(),model);
      dynamic_cast<G4HadronicProcess*>(pProcess)->RegisterMe(theListOfG4HadronicModels.back());
    }
  else if(model->GetModelName() == "G4LEKaonZeroSInelastic")
    {
      theListOfG4HadronicModels.push_back(new G4LEKaonZeroSInelastic);
      //G4LEKaonZeroSInelastic* g4model = new G4LEKaonZeroSInelastic;
      if(model->IsEnergyRangeDefined()) SetEnergyRange(theListOfG4HadronicModels.back(),model);
      dynamic_cast<G4HadronicProcess*>(pProcess)->RegisterMe(theListOfG4HadronicModels.back());
    }
  else if(model->GetModelName() == "G4LENeutronInelastic")
    {
      theListOfG4HadronicModels.push_back(new G4LENeutronInelastic);
      //G4LENeutronInelastic* g4model = new G4LENeutronInelastic;
      if(model->IsEnergyRangeDefined()) SetEnergyRange(theListOfG4HadronicModels.back(),model);
      dynamic_cast<G4HadronicProcess*>(pProcess)->RegisterMe(theListOfG4HadronicModels.back());
    }
  else if(model->GetModelName() == "G4LELambdaInelastic")
    {
      theListOfG4HadronicModels.push_back(new G4LELambdaInelastic);
      //G4LELambdaInelastic* g4model = new G4LELambdaInelastic;
      if(model->IsEnergyRangeDefined()) SetEnergyRange(theListOfG4HadronicModels.back(),model);
      dynamic_cast<G4HadronicProcess*>(pProcess)->RegisterMe(theListOfG4HadronicModels.back());
    }
  else if(model->GetModelName() == "G4LESigmaPlusInelastic")
    {
      theListOfG4HadronicModels.push_back(new G4LESigmaPlusInelastic);
      //G4LESigmaPlusInelastic* g4model = new G4LESigmaPlusInelastic;
      if(model->IsEnergyRangeDefined()) SetEnergyRange(theListOfG4HadronicModels.back(),model);
      dynamic_cast<G4HadronicProcess*>(pProcess)->RegisterMe(theListOfG4HadronicModels.back());
    }
  else if(model->GetModelName() == "G4LESigmaMinusInelastic")
    {
      theListOfG4HadronicModels.push_back(new G4LESigmaMinusInelastic);
      //G4LESigmaMinusInelastic* g4model = new G4LESigmaMinusInelastic;
      if(model->IsEnergyRangeDefined()) SetEnergyRange(theListOfG4HadronicModels.back(),model);
      dynamic_cast<G4HadronicProcess*>(pProcess)->RegisterMe(theListOfG4HadronicModels.back());
    }
  else if(model->GetModelName() == "G4LEXiMinusInelastic")
    {
      theListOfG4HadronicModels.push_back(new G4LEXiMinusInelastic);
      //G4LEXiMinusInelastic* g4model = new G4LEXiMinusInelastic;
      if(model->IsEnergyRangeDefined()) SetEnergyRange(theListOfG4HadronicModels.back(),model);
      dynamic_cast<G4HadronicProcess*>(pProcess)->RegisterMe(theListOfG4HadronicModels.back());
    }
  else if(model->GetModelName() == "G4LEXiZeroInelastic")
    {
      theListOfG4HadronicModels.push_back(new G4LEXiZeroInelastic);
      //G4LEXiZeroInelastic* g4model = new G4LEXiZeroInelastic;
      if(model->IsEnergyRangeDefined()) SetEnergyRange(theListOfG4HadronicModels.back(),model);
      dynamic_cast<G4HadronicProcess*>(pProcess)->RegisterMe(theListOfG4HadronicModels.back());
    }
  else if(model->GetModelName() == "G4LEOmegaMinusInelastic")
    {
      theListOfG4HadronicModels.push_back(new G4LEOmegaMinusInelastic);
      //G4LEOmegaMinusInelastic* g4model = new G4LEOmegaMinusInelastic;
      if(model->IsEnergyRangeDefined()) SetEnergyRange(theListOfG4HadronicModels.back(),model);
      dynamic_cast<G4HadronicProcess*>(pProcess)->RegisterMe(theListOfG4HadronicModels.back());
    }
  else if(model->GetModelName() == "G4LEAntiProtonInelastic")
    {
      theListOfG4HadronicModels.push_back(new G4LEAntiProtonInelastic);
      //G4LEAntiProtonInelastic* g4model = new G4LEAntiProtonInelastic;
      if(model->IsEnergyRangeDefined()) SetEnergyRange(theListOfG4HadronicModels.back(),model);
      dynamic_cast<G4HadronicProcess*>(pProcess)->RegisterMe(theListOfG4HadronicModels.back());
    }
  else if(model->GetModelName() == "G4LEAntiNeutronInelastic")
    {
      theListOfG4HadronicModels.push_back(new G4LEAntiNeutronInelastic);
      //G4LEAntiNeutronInelastic* g4model = new G4LEAntiNeutronInelastic;
      if(model->IsEnergyRangeDefined()) SetEnergyRange(theListOfG4HadronicModels.back(),model);
      dynamic_cast<G4HadronicProcess*>(pProcess)->RegisterMe(theListOfG4HadronicModels.back());
    }
  else if(model->GetModelName() == "G4LEAntiLambdaInelastic")
    {
      theListOfG4HadronicModels.push_back(new G4LEAntiLambdaInelastic);
      //G4LEAntiLambdaInelastic* g4model = new G4LEAntiLambdaInelastic;
      if(model->IsEnergyRangeDefined()) SetEnergyRange(theListOfG4HadronicModels.back(),model);
      dynamic_cast<G4HadronicProcess*>(pProcess)->RegisterMe(theListOfG4HadronicModels.back());
    }
  else if(model->GetModelName() == "G4LEAntiSigmaPlusInelastic")
    {
      theListOfG4HadronicModels.push_back(new G4LEAntiSigmaPlusInelastic);
      //G4LEAntiSigmaPlusInelastic* g4model = new G4LEAntiSigmaPlusInelastic;
      if(model->IsEnergyRangeDefined()) SetEnergyRange(theListOfG4HadronicModels.back(),model);
      dynamic_cast<G4HadronicProcess*>(pProcess)->RegisterMe(theListOfG4HadronicModels.back());
    }
  else if(model->GetModelName() == "G4LEAntiSigmaMinusInelastic")
    {
      theListOfG4HadronicModels.push_back(new G4LEAntiSigmaMinusInelastic);
      //G4LEAntiSigmaMinusInelastic* g4model = new G4LEAntiSigmaMinusInelastic;
      if(model->IsEnergyRangeDefined()) SetEnergyRange(theListOfG4HadronicModels.back(),model);
      dynamic_cast<G4HadronicProcess*>(pProcess)->RegisterMe(theListOfG4HadronicModels.back());
    }
  else if(model->GetModelName() == "G4LEAntiXiMinusInelastic")
    {
      theListOfG4HadronicModels.push_back(new G4LEAntiXiMinusInelastic);
      //G4LEAntiXiMinusInelastic* g4model = new G4LEAntiXiMinusInelastic;
      if(model->IsEnergyRangeDefined()) SetEnergyRange(theListOfG4HadronicModels.back(),model);
      dynamic_cast<G4HadronicProcess*>(pProcess)->RegisterMe(theListOfG4HadronicModels.back());
    }
  else if(model->GetModelName() == "G4LEAntiXiZeroInelastic")
    {
      theListOfG4HadronicModels.push_back(new G4LEAntiXiZeroInelastic);
      //G4LEAntiXiZeroInelastic* g4model = new G4LEAntiXiZeroInelastic;
      if(model->IsEnergyRangeDefined()) SetEnergyRange(theListOfG4HadronicModels.back(),model);
      dynamic_cast<G4HadronicProcess*>(pProcess)->RegisterMe(theListOfG4HadronicModels.back());
    }
  else if(model->GetModelName() == "G4LEAntiOmegaMinusInelastic")
    {
      theListOfG4HadronicModels.push_back(new G4LEAntiOmegaMinusInelastic);
      //G4LEAntiOmegaMinusInelastic* g4model = new G4LEAntiOmegaMinusInelastic;
      if(model->IsEnergyRangeDefined()) SetEnergyRange(theListOfG4HadronicModels.back(),model);
      dynamic_cast<G4HadronicProcess*>(pProcess)->RegisterMe(theListOfG4HadronicModels.back());
    }
#endif

  // Cascade
  else if(model->GetModelName() == "G4BinaryCascade")
    {
      theListOfG4HadronicModels.push_back(new G4BinaryCascade);
      //G4BinaryCascade* g4model = new G4BinaryCascade ;
      if(model->IsEnergyRangeDefined()) SetEnergyRange(theListOfG4HadronicModels.back(),model);
      dynamic_cast<G4HadronicProcess*>(pProcess)->RegisterMe(theListOfG4HadronicModels.back());
    }// else if(model->GetModelName() == "GateBinaryCascade")
  // {
  //   theListOfG4HadronicModels.push_back(new GateBinaryCascade);
  //   if(model->IsEnergyRangeDefined()) SetEnergyRange(theListOfG4HadronicModels.back(),model);
  //   dynamic_cast<G4HadronicProcess*>(pProcess)->RegisterMe(theListOfG4HadronicModels.back());
  // }
  else if(model->GetModelName() == "G4BertiniCascade")
    {
      theListOfG4HadronicModels.push_back(new G4CascadeInterface);
      //G4CascadeInterface* g4model = new G4CascadeInterface;
      if(model->IsEnergyRangeDefined()) SetEnergyRange(theListOfG4HadronicModels.back(),model);
      dynamic_cast<G4HadronicProcess*>(pProcess)->RegisterMe(theListOfG4HadronicModels.back());
    }

  // Precompound
  else if(model->GetModelName() == "PreCompound")
    {
      theHandler = new G4ExcitationHandler();
      theListOfG4HadronicModels.push_back(new G4PreCompoundModel(theHandler));
      //G4PreCompoundModel* g4model  = new G4PreCompoundModel(theHandler);
      if(model->IsEnergyRangeDefined()) SetEnergyRange(theListOfG4HadronicModels.back(),model);
      dynamic_cast<G4HadronicProcess*>(pProcess)->RegisterMe(theListOfG4HadronicModels.back());
    }

  // Leading Particle Bias

  //S.Jan modif 5/11/2010
  //G4Mars5GeV obsolete
  // /  else if(model->GetModelName() == "LeadingParticleBias")
  //   {
  //      theListOfG4HadronicModels.push_back(new G4Mars5GeV);
  //      //G4Mars5GeV* g4model  = new G4Mars5GeV();
  //      if(model->IsEnergyRangeDefined()) SetEnergyRange(theListOfG4HadronicModels.back(),model);
  //      dynamic_cast<G4HadronicProcess*>(pProcess)->RegisterMe(theListOfG4HadronicModels.back());
  //   }
  //
  // Gamma- and Lepto-Nuclear
#if (G4VERSION_MAJOR == 9)
  else if(model->GetModelName() == "G4ElectroNuclearReaction")
    {
      theListOfG4HadronicModels.push_back(new G4ElectroNuclearReaction);
      //G4ElectroNuclearReaction* g4model = new G4ElectroNuclearReaction;
      if(model->IsEnergyRangeDefined()) SetEnergyRange(theListOfG4HadronicModels.back(),model);
      dynamic_cast<G4HadronicProcess*>(pProcess)->RegisterMe(theListOfG4HadronicModels.back());
    }
  else if(model->GetModelName() == "G4GammaNuclearReaction")
    {
      theListOfG4HadronicModels.push_back(new G4GammaNuclearReaction);
      //G4GammaNuclearReaction* g4model = new G4GammaNuclearReaction;
      if(model->IsEnergyRangeDefined()) SetEnergyRange(theListOfG4HadronicModels.back(),model);
      dynamic_cast<G4HadronicProcess*>(pProcess)->RegisterMe(theListOfG4HadronicModels.back());
    }
  else if(model->GetModelName() == "G4NeutronHPorLElastic")
    {
      theListOfG4HadronicModels.push_back(new G4NeutronHPorLElastic);
      //G4NeutronHPorLElastic* g4model = new G4NeutronHPorLElastic;
      if(model->IsEnergyRangeDefined()) SetEnergyRange(theListOfG4HadronicModels.back(),model);
      dynamic_cast<G4HadronicProcess*>(pProcess)->RegisterMe(theListOfG4HadronicModels.back());
    }
  else if(model->GetModelName() == "G4NeutronHPorLEInelastic")
    {
      theListOfG4HadronicModels.push_back(new G4NeutronHPorLEInelastic);
      //G4NeutronHPorLEInelastic* g4model = new G4NeutronHPorLEInelastic;
      if(model->IsEnergyRangeDefined()) SetEnergyRange(theListOfG4HadronicModels.back(),model);
      dynamic_cast<G4HadronicProcess*>(pProcess)->RegisterMe(theListOfG4HadronicModels.back());
    }
  else if(model->GetModelName() == "G4NeutronHPorLFission")
    {
      theListOfG4HadronicModels.push_back(new G4NeutronHPorLFission);
      //G4NeutronHPorLFission* g4model = new G4NeutronHPorLFission;
      if(model->IsEnergyRangeDefined()) SetEnergyRange(theListOfG4HadronicModels.back(),model);
      dynamic_cast<G4HadronicProcess*>(pProcess)->RegisterMe(theListOfG4HadronicModels.back());
    }
#endif

  // Neutron (for high precision models, user must first download high precision neutron data files from Geant4 web page)
#if (G4VERSION_MAJOR == 9)
  else if(model->GetModelName() == "G4LCapture")
    {
      theListOfG4HadronicModels.push_back(new G4LCapture);
      //G4LCapture* g4model = new G4LCapture ;
      if(model->IsEnergyRangeDefined()) SetEnergyRange(theListOfG4HadronicModels.back(),model);
      dynamic_cast<G4HadronicProcess*>(pProcess)->RegisterMe(theListOfG4HadronicModels.back());
    }
  else if(model->GetModelName() == "G4NeutronHPorLCapture")
    {
      theListOfG4HadronicModels.push_back(new G4NeutronHPorLCapture);
      //G4NeutronHPorLCapture* g4model = new G4NeutronHPorLCapture;
      if(model->IsEnergyRangeDefined()) SetEnergyRange(theListOfG4HadronicModels.back(),model);
      dynamic_cast<G4HadronicProcess*>(pProcess)->RegisterMe(theListOfG4HadronicModels.back());
    }
#endif
  else if(model->GetModelName() == "G4NeutronHPCapture")
    {
      theListOfG4HadronicModels.push_back(new G4NeutronHPCapture);
      //G4NeutronHPCapture* g4model = new G4NeutronHPCapture;
      if(model->IsEnergyRangeDefined()) SetEnergyRange(theListOfG4HadronicModels.back(),model);
      dynamic_cast<G4HadronicProcess*>(pProcess)->RegisterMe(theListOfG4HadronicModels.back());
    }
  else if(model->GetModelName() == "G4LFission")
    {
      theListOfG4HadronicModels.push_back(new G4LFission);
      //G4LFission* g4model = new G4LFission;
      if(model->IsEnergyRangeDefined()) SetEnergyRange(theListOfG4HadronicModels.back(),model);
      dynamic_cast<G4HadronicProcess*>(pProcess)->RegisterMe(theListOfG4HadronicModels.back());
    }
  else if(model->GetModelName() == "G4NeutronHPElastic")
    {
      theListOfG4HadronicModels.push_back(new G4NeutronHPElastic);
      //G4NeutronHPElastic* g4model = new G4NeutronHPElastic;
      if(model->IsEnergyRangeDefined()) SetEnergyRange(theListOfG4HadronicModels.back(),model);
      dynamic_cast<G4HadronicProcess*>(pProcess)->RegisterMe(theListOfG4HadronicModels.back());
    }
  else if(model->GetModelName() == "G4NeutronHPInelastic")
    {
      theListOfG4HadronicModels.push_back(new G4NeutronHPInelastic);
      //G4NeutronHPInelastic* g4model = new G4NeutronHPInelastic;
      if(model->IsEnergyRangeDefined()) SetEnergyRange(theListOfG4HadronicModels.back(),model);
      dynamic_cast<G4HadronicProcess*>(pProcess)->RegisterMe(theListOfG4HadronicModels.back());
    }
  else if(model->GetModelName() == "G4NeutronHPFission")
    {
      theListOfG4HadronicModels.push_back(new G4NeutronHPFission);
      //G4NeutronHPFission* g4model = new G4NeutronHPFission;
      if(model->IsEnergyRangeDefined()) SetEnergyRange(theListOfG4HadronicModels.back(),model);
      dynamic_cast<G4HadronicProcess*>(pProcess)->RegisterMe(theListOfG4HadronicModels.back());
    }
  else if(model->GetModelName() == "G4QMDReaction")
    {
      theListOfG4HadronicModels.push_back(new G4QMDReaction);
      //G4QMDReaction* g4model = new G4QMDReaction;
      if(model->IsEnergyRangeDefined()) SetEnergyRange(theListOfG4HadronicModels.back(),model);
      dynamic_cast<G4HadronicProcess*>(pProcess)->RegisterMe(theListOfG4HadronicModels.back());
    }




  // User Model
  else AddUserModel(model);

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateVProcess::SetEnergyRange(G4HadronicInteraction * hadInteraction,GateListOfHadronicModels *model)
{

  std::vector<G4String> theListOfOptions = model->GetTheListOfOptions();
  std::vector<double> theListOfEmin = model->GetTheListOfEmin();
  std::vector<double> theListOfEmax = model->GetTheListOfEmax();

  for(unsigned int j=0; j<theListOfOptions.size(); j++)
    {
      if(theListOfOptions[j]=="NoOption")
        {
          if(theListOfEmin[j] >= 0.0) hadInteraction->SetMinEnergy(theListOfEmin[j]);
          if(theListOfEmax[j] >= 0.0) hadInteraction->SetMaxEnergy(theListOfEmax[j]);
        }
      else if(model->GetMaterial(theListOfOptions[j]))
        {
          G4Material * mat = model->GetMaterial(theListOfOptions[j]);
          if(theListOfEmin[j] >= 0.0) hadInteraction-> SetMinEnergy(theListOfEmin[j] , mat );
          if(theListOfEmax[j] >= 0.0) hadInteraction-> SetMaxEnergy(theListOfEmin[j] , mat );
        }
      else if(model->GetElement(theListOfOptions[j]))
        {
          G4Element * ele = model->GetElement(theListOfOptions[j]);
          if(theListOfEmin[j] >= 0.0) hadInteraction-> SetMinEnergy(theListOfEmin[j] , ele );
          if(theListOfEmax[j] >= 0.0) hadInteraction-> SetMaxEnergy(theListOfEmin[j] , ele );
        }
    }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// *** Not really useful ***
/*void GateVProcess::Verify(G4ParticleDefinition * particule)/
  {
  int nModel = 0;
  for(unsigned int i=0; i<theListOfEnabledParticles.size(); i++)
  if(theListOfEnabledParticles[i]==particule) nModel++;

  if(nModel == 0)
  {
  G4cout<< "\n  <!> *** Warning *** <!> This process "<<mG4ProcessName<<" is selected for "<<particle->GetParticleName()<< Gateendl;
  G4cout <<"                                          but it is not setted in his Process\n";
  return;
  }

  if(theListOfParticlesWithSelectedModels.size()==0)

  int nModelWithERange = 0;
  for(unsigned int i=0; i<theListOfParticlesWithSelectedModels.size(); i++)
  {
  if(theListOfParticlesWithSelectedModels[i] == particule) nModelWithERange++;
  }
  }*/
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
G4bool GateVProcess::AddFilter(G4String filterType, G4String particle )
{
  GateDebugMessageInc("Physic",4,"AddFilter() -- begin\n");
  if(GateActorManager::GetInstance()->theListOfFilterPrototypes[filterType])
    {
      for (std::map<G4String,GenericWrapperProcess*>::const_iterator iter = theListOfWrapper.begin(); iter!=theListOfWrapper.end();++iter)
        {
          if(particle=="primaries"){
            iter->second->GetFilterManagerPrimary()->AddFilter(GateActorManager::GetInstance()->theListOfFilterPrototypes[filterType]("/gate/physics/processes/"+mG4ProcessName+"/primaries/"+filterType));
            iter->second->IncFilterManagerPrimary();
          }
          if(particle=="secondaries"){
            iter->second ->GetFilterManagerSecondary()->AddFilter(GateActorManager::GetInstance()->theListOfFilterPrototypes[filterType]("/gate/physics/processes/"+mG4ProcessName+"/secondaries/"+filterType));
            iter->second->IncFilterManagerSecondary();
          }
        }
    }
  else
    {
      GateWarning("Filter type: "<<filterType<<" does not exist!");
      return false;
    }

  GateDebugMessageDec("Physic",4,"AddFilter() -- end\n");
  return true;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateVProcess::SetWrapperFactor(G4String part,G4double f)
{
  theListOfWrapperFactor[part]=f;
  if(!theListOfWrapper[part]) theListOfWrapper[part] = new GenericWrapperProcess(mG4ProcessName);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateVProcess::SetWrapperCSEFactor(G4String part,G4double f)
{
  theListOfWrapperCSEFactor[part]=f;
  if(!theListOfWrapper[part]) theListOfWrapper[part] = new GenericWrapperProcess(mG4ProcessName);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateVProcess::SetStepFunction(G4String part, G4double ratio, G4double finalRange)
{
  thelistOfRatioForStepFunction[part]=ratio;
  thelistOfFinalRangeForStepFunction[part]=finalRange;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateVProcess::SetLinearlosslimit(G4String part,  G4double limit)
{
  thelistOfLinearLossLimit[part]=limit;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateVProcess::SetMsclimitation(G4String part, G4String  limit)
{
  G4MscStepLimitType limitation;
  if(limit=="safety") limitation = fUseSafety;
  else if(limit=="distanceToBoundary") limitation = fUseDistanceToBoundary;
  else GateError("Candidates for 'setGeometricalStepLimiterType' are safety or distanceToBoundary");
  thelistOfMscLimitation[part]=limitation;
}
//-----------------------------------------------------------------------------

#endif
