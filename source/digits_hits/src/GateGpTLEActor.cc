#include "GateGpTLEActor.hh"
#include "TFile.h"
#include "TAxis.h"

#include "GateConfiguration.h"
#ifdef G4ANALYSIS_USE_ROOT

#include "GateGpTLEActorMessenger.hh"
#include <G4VProcess.hh>
#include <G4ProtonInelasticProcess.hh>
#include <G4CrossSectionDataStore.hh>
#include <G4UnitsTable.hh>

GateGpTLEActor::GateGpTLEActor(G4String name, G4int depth):
	GateVActor(name,depth)
{
        mSaveFilename = "spectrumOUT";
        mUserFileSpectreBaseName = "spectrum";
        pMessenger = new GateGpTLEActorMessenger(this);
}

GateGpTLEActor::~GateGpTLEActor()
{
	delete pMessenger;
}

void GateGpTLEActor::Construct()
{
	GateVActor::Construct();

        // Enable callbacks
	EnableBeginOfRunAction(true);
	EnableBeginOfEventAction(true);
	EnablePreUserTrackingAction(true);
	EnablePostUserTrackingAction(true);
	EnableUserSteppingAction(true);
	EnableEndOfEventAction(true); // for save every n

        pFile = new TFile(mSaveFilename,"RECREATE","ROOT file for GpTLE",9);
        H_SUM = new TH1D( "H_SUM", "H_SUM", 250, 0., 10.);
        Ep = new TH1D("Ep","proton energy",250,0,250);

        //pListeVar = new TH2D("GpTLE","GpTLE histogram"); //ATTENTION ICI JE NE SUIS PAS SUR QUELL de deux on utilise
        //pListeVar = new TTree("GpTLE","GpTLE histogram");
	///REVISAR..... N'IMPORTE QUOI
	//	MaterialMap[ "Air"] = new TH2D("Air ","Air", 250, 0, 250, 250, 0, 10); 
        ResetData();
}


void GateGpTLEActor::ResetData()
{
        H_SUM->Reset();
	Ep->Reset();
}


void GateGpTLEActor::FileSpectreBaseName(G4String fileName)
{
    mUserFileSpectreBaseName = fileName;
}


void GateGpTLEActor::constructSpectrumMaterial(G4String material_Name)
{
 
  double temp;
  //G4String pathName = "/home/eromero/Documents/gate_simu/tiro92/data/";
  G4String nom_complet = mUserFileSpectreBaseName + "_" + material_Name + ".root";
  TFile* tfile = new TFile(nom_complet);
  assert(tfile);
  TH2D *EpEgpNorm = dynamic_cast<TH2D*>(tfile->Get("EpEgpNorm"));
  TH1D *EpInelastic = dynamic_cast<TH1D*>(tfile->Get("EpInelastic"));

  minX = EpEgpNorm -> GetXaxis() -> GetXmin() ;
  maxX = EpEgpNorm -> GetXaxis() -> GetXmax() ;
  minY = EpEgpNorm -> GetYaxis() -> GetXmin() ;
  maxY = EpEgpNorm -> GetYaxis() -> GetXmax() ;
  NbinsX = EpEgpNorm -> GetNbinsX();
  NbinsY = EpEgpNorm -> GetNbinsY();

  // G4cout << "X: " << minX << " " << maxX << " Y: " << minY << " " << maxY << G4endl;

  TH2D *EpEgp_Inelastic = new TH2D("EpEgp_Inelastic","PG normalized by mean free path in [m] and of proton inelastic",NbinsX,minX,maxX,NbinsY,minY,maxY);//JML values should be read
  //TH2D *EpEgp_Inelastic = new TH2D(*EpEgpNorm);


  for( int i=1; i <= EpEgpNorm->GetNbinsX(); i++)
     {
       for( int j = 1; j <= EpEgpNorm->GetNbinsY(); j++)
 	{
          if(EpInelastic->GetBinContent(i) != 0)
            {
	     temp = EpEgpNorm->GetBinContent(i,j)/EpInelastic->GetBinContent(i);
	     // temp = EpEgpNorm->GetBinContent(i,j)/10000000;
            }
          else
            {
	      if (EpEgpNorm->GetBinContent(i,j)> 0.) G4cout << "ERROR in EpEgpNorm->GetBinContent(i,j)" << G4endl;
	      temp = 0.0; //JML EpEgpNorm->GetBinContent(i,j);
            }
 	  EpEgp_Inelastic->SetBinContent(i,j,temp);
 	}
     }
 
  //EpEgp_Inelastic->SetName(  "Wasila");

  MaterialMap[ material_Name] = EpEgp_Inelastic;
  // MaterialMap[ material_Name] = EpEgpNorm;
  // MaterialMap[ material_Name] = new TH2D(*EpEgp_Inelastic));
  // MaterialMap[ material_Name] = (TH2D*)gROOT->FindObject(“EpEgp_Inelastic”);
  //G4cout << "second: " << MaterialMap[ material_Name] << " " << MaterialMap[ material_Name]->GetName() << G4endl;
  //G4cout << "NbinX: " << EpEgp_Inelastic->GetNbinsX() << "NbinY " << EpEgp_Inelastic->GetNbinsY() << G4endl;

      //G4cout << gROOT->FindObject("EpEgp_Inelastic") << G4endl;

}




void GateGpTLEActor::SaveData()
{
    GateVActor::SaveData();
    pFile->Write();
}


void GateGpTLEActor::BeginOfRunAction(const G4Run*)
{
}

void GateGpTLEActor::BeginOfEventAction(const G4Event*)
{
}

void GateGpTLEActor::EndOfEventAction(const G4Event*)
{
}

void GateGpTLEActor::PreUserTrackingAction(const GateVVolume*, const G4Track*)
{
	last_secondaries_size = 0;
	first_step = true;
}

void GateGpTLEActor::PostUserTrackingAction(const GateVVolume*, const G4Track*)
{
}

void GateGpTLEActor::UserSteppingAction(const GateVVolume*, const G4Step* step)
{
  
  //proton call
  const G4String particle_name = step->GetTrack()->GetParticleDefinition()->GetParticleName();
  if (particle_name != "proton") return;
    
  G4double distance = step->GetStepLength();
  const G4double particleEnergy = step->GetPreStepPoint()->GetKineticEnergy();
  Ep->Fill(particleEnergy/MeV);
  //G4cout << "unit: " << G4BestUnit(distance, "Length") << " ou alors en m : " << distance*m << " " << distance/m << G4endl;
  
  
  const G4Material* material = step->GetPreStepPoint()->GetMaterial();
  G4String materialName = material->GetName();
  TH1D  *Py1;
  std::map< G4String, TH2D*>::iterator histoResult = MaterialMap.end();
  histoResult = MaterialMap.find(materialName);


  if( histoResult != MaterialMap.end())
    {
      G4int binX = (G4int) ceil (((G4double) NbinsX) * (particleEnergy - minX) / (maxX - minX));
      // G4cout << "bins " << binX << " " << int(particleEnergy) << G4endl;
      Py1 = histoResult->second->ProjectionY( "PhistoEnergy", binX, binX);
    }
  else
    {
      constructSpectrumMaterial(materialName);
      G4int binX = (G4int) ceil (((G4double) NbinsX) * (particleEnergy - minX) / (maxX - minX));
      // G4cout << "bins " << binX << " " << int(particleEnergy) << G4endl;
      Py1 = MaterialMap[ materialName]->ProjectionY( "PhistoEnergy", binX, binX);
    }

  Py1->Scale(distance);
  H_SUM->Add(Py1);

}



#endif
